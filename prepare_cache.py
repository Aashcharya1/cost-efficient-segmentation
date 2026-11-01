"""Optional training-only uint8 cache. Evaluation/timing still starts at 256x256 photo."""

from src import runtime

runtime.configure()
from src.dataset import DUTSDataset
from pathlib import Path
import cv2, numpy as np, json, time

root = Path("data")
d = DUTSDataset(root, "train")
pairs = sorted(d.pairs + DUTSDataset(root, "val").pairs, key=lambda p: p[0].name)
cache = root / "cache"
cache.mkdir(exist_ok=True)
images = np.lib.format.open_memmap(
    cache / "images.npy", mode="w+", dtype=np.uint8, shape=(len(pairs), 256, 256, 3)
)
masks = np.lib.format.open_memmap(
    cache / "masks.npy", mode="w+", dtype=np.uint8, shape=(len(pairs), 256, 256)
)
for i, (ip, mp) in enumerate(pairs):
    im = cv2.imread(str(ip))
    mask = cv2.imread(str(mp), 0)
    images[i] = cv2.cvtColor(
        cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB
    )
    masks[i] = (
        cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST) >= 128
    ).astype(np.uint8)
    if i % 1000 == 0:
        print(i, len(pairs), flush=True)
images.flush()
masks.flush()
(cache / "names.json").write_text(json.dumps([p.name for p, _ in pairs]))
print("Cache complete", flush=True)
