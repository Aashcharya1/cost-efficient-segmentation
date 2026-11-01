"""Select a final full/half-width model by paired full-validation CES only.

This script does not open DUTS-TE or captured images. All candidates are timed
on each validation image in rotating order to reduce machine-load/order bias.
"""

from src import runtime

runtime.configure()
from pathlib import Path
import json, time, shutil
import cv2, numpy as np, torch
from src.predict import Predictor
from src.dataset import DUTSDataset
from src.utils import Metrics, ces_score, flatten_checkpoint


def pack(source, mode):
    saved = torch.load(source, map_location="cpu", weights_only=True)
    state = saved.get("state_dict", saved)
    arch = saved.get("architecture", {"width_mult": 1.0})
    packed = {}
    quantized = {}
    for name, tensor in state.items():
        use = (
            tensor.ndim == 4
            and mode != "float16"
            and not (
                mode == "int8_except_head_context"
                and name in ("head.weight", "context.weight")
            )
        )
        if use:
            x = tensor.float()
            scale = x.abs().amax((1, 2, 3), keepdim=True) / 127
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            quantized[name] = {
                "values": (x / scale).round().clamp(-127, 127).to(torch.int8),
                "scale": scale,
            }
        else:
            packed[name] = tensor
    return flatten_checkpoint(
        dict(
            state_dict=packed,
            quantized=quantized,
            architecture=arch,
            inference=dict(
                input_size=160, flip=False, logit_bias=0.0, torchscript=True
            ),
            storage_format=mode,
        )
    )


def main():
    root = Path("experiments/compact_selection")
    root.mkdir(exist_ok=True)
    declaration = dict(
        candidates=[
            "full_width_int8",
            "half_width_float16",
            "half_width_int8_convs",
            "half_width_int8_except_head_context",
            "half_width_int8_flip",
            "threequarter_width_float16",
            "threequarter_width_int8_convs",
            "threequarter_width_int8_except_head_context",
            "threequarter_width_int8_flip",
        ],
        selection="highest paired full-validation CES with CPU <=50ms",
        quality_data="DUTS-TR validation only",
        timing_order="rotate candidate order on each photo",
    )
    (root / "protocol.json").write_text(json.dumps(declaration, indent=2))
    full_path = root / "full_width_int8.pt"
    torch.save(
        flatten_checkpoint(
            torch.load(
                "experiments/full_optimized.pt", map_location="cpu", weights_only=True
            )
        ),
        full_path,
    )
    paths = {"full_width_int8": full_path}
    for mode in ["float16", "int8_convs", "int8_except_head_context"]:
        name = "half_width_" + mode
        paths[name] = root / (name + ".pt")
        torch.save(pack("params/student/model_fp16.pt", mode), paths[name])
    flip = pack("params/student/model_fp16.pt", "int8_convs")
    flip["inference"]["flip"] = True
    paths["half_width_int8_flip"] = root / "half_width_int8_flip.pt"
    torch.save(flip, paths["half_width_int8_flip"])
    for mode in ["float16", "int8_convs", "int8_except_head_context"]:
        name = "threequarter_width_" + mode
        paths[name] = root / (name + ".pt")
        torch.save(pack("params/pruned/model_fp16.pt", mode), paths[name])
    flip = pack("params/pruned/model_fp16.pt", "int8_convs")
    flip["inference"]["flip"] = True
    paths["threequarter_width_int8_flip"] = root / "threequarter_width_int8_flip.pt"
    torch.save(flip, paths["threequarter_width_int8_flip"])
    models = {name: Predictor(path) for name, path in paths.items()}
    meters = {name: Metrics() for name in models}
    times = {name: 0.0 for name in models}
    data = DUTSDataset("data", "val")
    warm = cv2.resize(
        cv2.imread(str(data.pairs[0][0])), (256, 256), interpolation=cv2.INTER_AREA
    )
    for model in models.values():
        model(warm)
    names = list(models)
    for i, (ip, mp) in enumerate(data.pairs):
        image = cv2.resize(
            cv2.imread(str(ip)), (256, 256), interpolation=cv2.INTER_AREA
        )
        gt = (
            cv2.resize(
                cv2.imread(str(mp), 0), (256, 256), interpolation=cv2.INTER_NEAREST
            )
            >= 128
        ).astype(np.uint8)
        order = names[i % len(names) :] + names[: i % len(names)]
        for name in order:
            start = time.process_time()
            pred = models[name](image)
            times[name] += time.process_time() - start
            meters[name].update(pred, gt)
        if (i + 1) % 250 == 0:
            print("Paired validation", i + 1, len(data), flush=True)
    records = []
    for name in names:
        result = meters[name].result()
        result.update(
            candidate=name,
            t_cpu_ms=times[name] * 1000 / len(data),
            s_disk_bytes=paths[name].stat().st_size,
        )
        result["ces"] = ces_score(
            result["q"], result["t_cpu_ms"], result["s_disk_bytes"]
        )
        records.append(result)
    best = max(records, key=lambda r: r["ces"])
    torch.save(
        torch.load(paths[best["candidate"]], map_location="cpu", weights_only=True),
        "params/model_fp16.pt",
    )
    if best["candidate"].startswith("half_width"):
        shutil.copyfile("params/student/training.json", "params/training.json")
    elif best["candidate"].startswith("threequarter_width"):
        shutil.copyfile("params/pruned/training.json", "params/training.json")
    else:
        shutil.copyfile("experiments/baseline/training.json", "params/training.json")
    (root / "results.json").write_text(
        json.dumps(
            dict(protocol=declaration, records=records, selected=best["candidate"]),
            indent=2,
        )
    )
    print("SELECTED", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
