"""Reproduce hand-traced masks on the resized user photos, without model predictions.

Coordinates were manually traced at 256x256. Bicycle tyres/spokes use manually
placed geometric primitives. Fine edges remain approximate and need human review.
"""

from src import runtime

runtime.configure()
from pathlib import Path
import json, math
import cv2, numpy as np


def polygon(points):
    yy, xx = np.mgrid[:256, :256]
    x = xx + 0.5
    y = yy + 0.5
    mask = np.zeros((256, 256), bool)
    x0, y0 = points[-1]
    for x1, y1 in points:
        if y0 != y1:
            mask ^= ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0)
        x0, y0 = x1, y1
    return mask


def line(points, width):
    yy, xx = np.mgrid[:256, :256]
    x = xx + 0.5
    y = yy + 0.5
    mask = np.zeros((256, 256), bool)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        dx = x1 - x0
        dy = y1 - y0
        t = np.clip(((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy + 1e-12), 0, 1)
        mask |= (x - x0 - t * dx) ** 2 + (y - y0 - t * dy) ** 2 <= (width / 2) ** 2
    return mask


def main():
    root = Path("captured")
    annotations = json.loads(
        (root / "annotations.json").read_text(encoding="utf-8-sig")
    )
    yy, xx = np.mgrid[:256, :256]
    for name, item in annotations.items():
        source = root / "original" / (name + ".jpg")
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(source)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(root / "images_256" / (name + ".png")), image)
        mask = np.zeros((256, 256), bool)
        for points in item["polygons"]:
            mask |= polygon(points)
        for cx, cy, rx, ry, ix, iy in item.get("ellipses", []):
            mask |= (((xx + 0.5 - cx) / rx) ** 2 + ((yy + 0.5 - cy) / ry) ** 2 <= 1) & (
                ((xx + 0.5 - cx) / ix) ** 2 + ((yy + 0.5 - cy) / iy) ** 2 >= 1
            )
        for points, width in item.get("lines", []):
            mask |= line(points, width)
        for cx, cy, rx, ry in item.get("spoke_wheels", []):
            for theta in np.linspace(0, 2 * math.pi, 20, endpoint=False):
                mask |= line(
                    [(cx, cy), (cx + rx * math.cos(theta), cy + ry * math.sin(theta))],
                    0.65,
                )
        for points in item["holes"]:
            mask &= ~polygon(points)
        cv2.imwrite(
            str(root / "masks_256" / (name + ".png")), mask.astype(np.uint8) * 255
        )
    rows = []
    for ip in sorted((root / "images_256").glob("*.png")):
        im = cv2.imread(str(ip))
        m = cv2.imread(str(root / "masks_256" / ip.name), 0) > 0
        overlay = im.copy()
        overlay[m] = (im[m] * 0.55 + np.array([40, 220, 40]) * 0.45).astype(np.uint8)
        rows.append(np.concatenate([im, overlay], 1))
    cv2.imwrite("experiments/captured_annotation_review.png", np.concatenate(rows, 0))


if __name__ == "__main__":
    main()
