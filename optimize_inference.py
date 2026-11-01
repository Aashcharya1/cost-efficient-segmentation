"""Predeclared validation-only inference grid. No DUTS-TE/captured access."""

from src import runtime

runtime.configure()
import json, time, hashlib
from pathlib import Path
import numpy as np, torch, cv2
from torch.utils.data import DataLoader
from src.dataset import DUTSDataset
from src.model import SmallFCN
from src.utils import load_checkpoint, Metrics, ces_score
from src.predict import Predictor

ROOT = Path("experiments")
state = torch.load(
    ROOT / "baseline/model_fp16.pt", map_location="cpu", weights_only=True
)
GRID = dict(
    input_size=[128, 160, 192, 224, 256],
    flip=[False, True],
    logit_bias=[-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75],
)
(ROOT / "predeclared_grid.json").write_text(json.dumps(GRID, indent=2))
val = DUTSDataset("data", "val")
loader = DataLoader(val, batch_size=16, num_workers=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_checkpoint(SmallFCN(), ROOT / "baseline/model_fp16.pt", device)
records = []
for size in GRID["input_size"]:
    model.input_size = size
    meters = {
        (flip, bias): Metrics() for flip in GRID["flip"] for bias in GRID["logit_bias"]
    }
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            base = model(x)
            flipped = model(x.flip(-1)).flip(-1)
            for flip in GRID["flip"]:
                logits = (base + flipped) * 0.5 if flip else base
                margins = (logits[:, 1] - logits[:, 0]).cpu().numpy()
                for bias in GRID["logit_bias"]:
                    for pred, gt in zip(margins > bias, y.numpy()):
                        meters[(flip, bias)].update(pred.astype(np.uint8), gt)
    for flip in GRID["flip"]:
        config = dict(input_size=size, flip=flip, logit_bias=0.0, torchscript=True)
        torch.save(dict(state_dict=state, inference=config), ROOT / "candidate.pt")
        predictor = Predictor(ROOT / "candidate.pt")
        image = cv2.resize(
            cv2.imread(str(val.pairs[0][0])), (256, 256), interpolation=cv2.INTER_AREA
        )
        predictor(image)
        total = 0.0
        for _ in range(64):
            t = time.process_time()
            predictor(image)
            total += time.process_time() - t
        ms = total * 1000 / 64
        for bias in GRID["logit_bias"]:
            r = meters[(flip, bias)].result()
            r.update(input_size=size, flip=flip, logit_bias=bias, estimated_cpu_ms=ms)
            r["validation_ces_estimate"] = ces_score(
                r["q"], ms, (ROOT / "candidate.pt").stat().st_size
            )
            records.append(r)
    print(
        "Completed",
        size,
        "best",
        max(records, key=lambda r: r["validation_ces_estimate"]),
        flush=True,
    )
    (ROOT / "validation_search.json").write_text(
        json.dumps(
            dict(
                grid=GRID,
                selection_rule="highest validation CES estimate, CPU<50; no test/captured access",
                records=records,
            ),
            indent=2,
        )
    )
best = max(records, key=lambda r: r["validation_ces_estimate"])
config = dict(
    input_size=best["input_size"],
    flip=best["flip"],
    logit_bias=best["logit_bias"],
    torchscript=True,
)
torch.save(dict(state_dict=state, inference=config), "params/model_fp16.pt")
(ROOT / "selected_inference.json").write_text(
    json.dumps(dict(config=config, validation=best), indent=2)
)
print("SELECTED", best, flush=True)
