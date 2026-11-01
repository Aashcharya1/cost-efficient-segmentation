"""Validation-only comparison of uncompressed per-output-channel weight storage."""

from src import runtime

runtime.configure()
import json, time, hashlib
from pathlib import Path
import cv2, numpy as np, torch
from src.dataset import DUTSDataset
from src.predict import Predictor
from src.utils import Metrics, ces_score

root = Path("experiments")
state = torch.load(
    root / "baseline/model_fp16.pt", map_location="cpu", weights_only=True
)
config = json.loads((root / "selected_inference.json").read_text())["config"]
val = DUTSDataset("data", "val")
records = []
for mode in ["float16", "int8_convs", "int8_except_head_context"]:
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
            scale = x.abs().amax(dim=(1, 2, 3), keepdim=True) / 127
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            quantized[name] = dict(
                values=(x / scale).round().clamp(-127, 127).to(torch.int8), scale=scale
            )
        else:
            packed[name] = tensor
    saved = dict(
        state_dict=packed, quantized=quantized, inference=config, storage_format=mode
    )
    path = root / (mode + ".pt")
    torch.save(saved, path)
    predictor = Predictor(path)
    image = cv2.resize(
        cv2.imread(str(val.pairs[0][0])), (256, 256), interpolation=cv2.INTER_AREA
    )
    predictor(image)
    metrics = Metrics()
    total = 0.0
    for ip, mp in val.pairs:
        image = cv2.resize(
            cv2.imread(str(ip)), (256, 256), interpolation=cv2.INTER_AREA
        )
        gt = (
            cv2.resize(
                cv2.imread(str(mp), 0), (256, 256), interpolation=cv2.INTER_NEAREST
            )
            >= 128
        ).astype(np.uint8)
        start = time.process_time()
        pred = predictor(image)
        total += time.process_time() - start
        metrics.update(pred, gt)
    result = metrics.result()
    result.update(
        storage=mode, t_cpu_ms=total * 1000 / len(val), s_disk_bytes=path.stat().st_size
    )
    result["ces"] = ces_score(result["q"], result["t_cpu_ms"], result["s_disk_bytes"])
    records.append(result)
    print(json.dumps(result), flush=True)
(root / "storage_search.json").write_text(
    json.dumps(
        dict(
            selection="highest full-validation CES; no DUTS-TE or captured access",
            records=records,
        ),
        indent=2,
    )
)
best = max(records, key=lambda r: r["ces"])
saved = torch.load(
    root / (best["storage"] + ".pt"), map_location="cpu", weights_only=True
)
torch.save(saved, "params/model_fp16.pt")
(root / "selected_storage.json").write_text(json.dumps(best, indent=2))
print("SELECTED", best, flush=True)
