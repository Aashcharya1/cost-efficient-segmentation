"""One-photo-at-a-time streaming CPU evaluation and process_time benchmark."""

try:
    from . import runtime
except ImportError:
    import runtime
import argparse, json, time, platform, hashlib
from pathlib import Path
import cv2, numpy as np

try:
    from .dataset import DUTSDataset
    from .predict import Predictor
    from .utils import Metrics, ces_score
except ImportError:
    from dataset import DUTSDataset
    from predict import Predictor
    from utils import Metrics, ces_score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default="data")
    p.add_argument("--checkpoint", default="params/model_fp16.pt")
    p.add_argument("--split", choices=["val", "test", "captured"], default="val")
    p.add_argument("--captured_root", default="captured")
    p.add_argument("--output", default=None)
    a = p.parse_args()
    runtime.configure()
    predict = Predictor(a.checkpoint)
    if a.split == "captured":
        root = Path(a.captured_root)
        pairs = [
            (ip, root / "masks_256" / (ip.stem + ".png"))
            for ip in sorted((root / "images_256").glob("*.png"))
        ]
        if not pairs:
            raise ValueError("No captured images")
        for _, mp in pairs:
            if not mp.exists():
                raise FileNotFoundError(mp)
    else:
        pairs = DUTSDataset(a.data_root, a.split).pairs
    metrics = Metrics()
    total = 0.0
    records = []
    warm_photo = cv2.imread(str(pairs[0][0]))
    if warm_photo is None:
        raise ValueError(f"Unreadable warm-up image: {pairs[0][0]}")
    predict(cv2.resize(warm_photo, (256, 256), interpolation=cv2.INTER_AREA))
    for i, (ip, mp) in enumerate(pairs):
        image = cv2.imread(str(ip))
        gt = cv2.imread(str(mp), 0)
        if image is None or gt is None:
            raise ValueError(f"Unreadable pair: {ip}, {mp}")
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        gt = (
            cv2.resize(gt, (256, 256), interpolation=cv2.INTER_NEAREST) >= 128
        ).astype(np.uint8)
        start = time.process_time()
        pred = predict(image)
        seconds = time.process_time() - start
        total += seconds
        single = Metrics()
        single.update(pred, gt)
        records.append(dict(image=ip.name, t_cpu_ms=seconds * 1000, **single.result()))
        metrics.update(pred, gt)
        if (i + 1) % 500 == 0:
            print(f"Evaluated {i+1}/{len(pairs)}", flush=True)
    result = metrics.result()
    result.update(
        split=a.split,
        t_cpu_ms=total * 1000 / len(pairs),
        s_disk_bytes=Path(a.checkpoint).stat().st_size,
        cpu=platform.processor(),
        checkpoint=str(Path(a.checkpoint).resolve()),
    )
    result["checkpoint_sha256"] = hashlib.sha256(
        Path(a.checkpoint).read_bytes()
    ).hexdigest()
    result["ces"] = ces_score(result["q"], result["t_cpu_ms"], result["s_disk_bytes"])
    result["per_image"] = records
    dest = Path(a.output or f"report/{a.split}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
