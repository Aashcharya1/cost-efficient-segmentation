"""CPU prediction from an in-memory uint8 BGR photo, including preprocessing."""

try:
    from . import runtime
except ImportError:
    import runtime
import argparse
import os
import time
from pathlib import Path
import cv2
import numpy as np
import torch

try:
    from .model import SmallFCN
    from .utils import load_checkpoint
except ImportError:
    from model import SmallFCN
    from utils import load_checkpoint


class Predictor:
    """Load parameters once; each call is the complete timed CPU pipeline.

    Input is OpenCV-order BGR uint8, exactly 256x256. Output is a 0/1
    uint8 NumPy mask. Conversion to 0/255 is only for untimed PNG export.
    """

    def __init__(self, checkpoint):
        runtime.configure()
        self.model = load_checkpoint(
            SmallFCN(), str(checkpoint), torch.device("cpu")
        ).eval()
        config = self.model.inference_config
        self.flip = bool(config.get("flip", False))
        self.logit_bias = float(config.get("logit_bias", 0.0))
        if config.get("torchscript", False):
            with torch.inference_mode():
                example = torch.zeros(1, 256, 256, 3).permute(0, 3, 1, 2)
                self.model = torch.jit.optimize_for_inference(
                    torch.jit.freeze(torch.jit.trace(self.model, example))
                )

    @torch.inference_mode()
    def __call__(self, bgr):
        if bgr.shape != (256, 256, 3) or bgr.dtype != np.uint8:
            raise ValueError("Expected a 256x256 uint8 BGR photo")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div_(255)
        logits = self.model(x)
        if self.flip:
            logits = (logits + self.model(x.flip(-1)).flip(-1)) * 0.5
        return (
            ((logits[0, 1] - logits[0, 0]) > self.logit_bias).numpy().astype(np.uint8)
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--checkpoint",
        default=str(Path(__file__).resolve().parents[1] / "params/model_fp16.pt"),
    )
    a = p.parse_args()
    runtime.configure()
    model = Predictor(a.checkpoint)
    photo = cv2.imread(a.image)
    if photo is None:
        raise FileNotFoundError(a.image)
    photo = cv2.resize(photo, (256, 256), interpolation=cv2.INTER_AREA)
    mask = model(photo)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(a.out, mask * 255):
        raise OSError(a.out)
    total = 0.0
    for _ in range(20):
        start = time.process_time()
        model(photo)
        total += time.process_time() - start
    print(f"T CPU ms: {total*1000/20:.2f}")
    print(f"S disk bytes: {os.path.getsize(a.checkpoint)}")


if __name__ == "__main__":
    main()
