"""Check delivered measurements, frozen model identity and the real CLI contract."""

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from src import runtime

runtime.configure()
import cv2
import numpy as np
from src.utils import ces_score, miou_from_confusion

root = Path(__file__).resolve().parent
checkpoint = root / "params/model_fp16.pt"
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
manifest = json.loads((root / "captured/manifest.json").read_text())
photos = manifest["photos"]
assert 5 <= len(photos) <= 10
for photo in photos:
    source = root / "captured/original" / photo["renamed_file"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == photo["sha256"]
image_names = {p.stem for p in (root / "captured/images_256").glob("*.png")}
mask_names = {p.stem for p in (root / "captured/masks_256").glob("*.png")}
assert image_names == mask_names == {p["name"] for p in photos}
for split, expected in [("val", 1056), ("test", 5019), ("captured", len(photos))]:
    r = json.loads((root / f"report/{split}.json").read_text())
    assert r["n"] == expected == len(r["per_image"]), split
    assert r["checkpoint_sha256"] == digest, split
    assert r["s_disk_bytes"] == checkpoint.stat().st_size, split
    if split == "test":
        assert r["t_cpu_ms"] <= 50, "DUTS-TE mean CPU time exceeds budget"
    c = np.asarray(r["confusion"])
    assert c.sum() == expected * 256 * 256, split
    assert np.array_equal(
        c, np.sum([x["confusion"] for x in r["per_image"]], axis=0)
    ), split
    assert abs(miou_from_confusion(c) - r["miou"]) < 1e-10, split
    assert abs(np.mean([x["bf1"] for x in r["per_image"]]) - r["bf1"]) < 1e-10, split
    assert (
        abs(np.mean([x["t_cpu_ms"] for x in r["per_image"]]) - r["t_cpu_ms"]) < 1e-8
    ), split
    assert abs(0.7 * r["miou"] + 0.3 * r["bf1"] - r["q"]) < 1e-10, split
    assert (
        abs(ces_score(r["q"], r["t_cpu_ms"], r["s_disk_bytes"]) - r["ces"]) < 1e-10
    ), split

for ip in (root / "captured/images_256").glob("*.png"):
    mask = cv2.imread(str(root / "captured/masks_256" / ip.name), 0)
    assert mask.shape == (256, 256)
    assert set(np.unique(mask)) <= {0, 255}

with tempfile.TemporaryDirectory() as folder:
    out = Path(folder) / "prediction.png"
    p = subprocess.run(
        [
            sys.executable,
            str(root / "predict.py"),
            "--image",
            str(root / "captured/original" / photos[0]["renamed_file"]),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
    )
    assert re.fullmatch(r"T CPU ms: \d+\.\d{2}\nS disk bytes: \d+\n", p.stdout), repr(
        p.stdout
    )
    mask = cv2.imread(str(out), 0)
    assert mask.shape == (256, 256) and set(np.unique(mask)) <= {0, 255}
    assert int(p.stdout.splitlines()[1].split(": ")[1]) == checkpoint.stat().st_size
    print(p.stdout, end="")
print("Submission consistency and CLI checks passed.")
