"""Deterministic filename split; validates the dataset fingerprint on reuse."""

import json
import hashlib
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DUTSDataset(Dataset):
    """Pair separate image/mask directories and guard a persistent train split.

    The optional lossless uint8 cache is training-only acceleration. Test
    images never enter this cache, augmentation, or the split manifest.
    """

    def __init__(self, root_dir, split="train", augment=False):
        if split not in ("train", "val", "test"):
            raise ValueError(split)
        root = Path(root_dir)
        name = "DUTS-TE" if split == "test" else "DUTS-TR"
        base = root / name
        images = base / (name + "-Image")
        masks = base / (name + "-Mask")
        if not masks.exists():
            masks = base / (name + "-MASK")
        image_paths = sorted(
            p for p in images.glob("*") if p.suffix.lower() in (".jpg", ".png", ".jpeg")
        )
        mask_map = {
            p.stem: p
            for p in masks.glob("*")
            if p.suffix.lower() in (".jpg", ".png", ".jpeg")
        }
        missing = [p.name for p in image_paths if p.stem not in mask_map]
        if missing:
            raise ValueError(f"Missing masks: {missing[:5]}")
        expected = 5019 if split == "test" else 10553
        if len(image_paths) != expected:
            raise ValueError(
                f"{name}: expected {expected} photos, found {len(image_paths)}"
            )
        names = [p.name for p in image_paths]
        if split != "test":
            manifest = root / "split.json"
            fingerprint = hashlib.sha256("\n".join(names).encode()).hexdigest()
            if manifest.exists():
                info = json.loads(manifest.read_text())
                if info["fingerprint"] != fingerprint:
                    raise ValueError("Dataset changed since split was created")
            else:
                idx = np.random.default_rng(42).permutation(len(names))
                n = int(0.9 * len(idx))
                info = dict(
                    seed=42,
                    fingerprint=fingerprint,
                    train=[names[i] for i in idx[:n]],
                    val=[names[i] for i in idx[n:]],
                )
                manifest.write_text(json.dumps(info, indent=2))
            keep = set(info[split])
            image_paths = [p for p in image_paths if p.name in keep]
        self.pairs = [(p, mask_map[p.stem]) for p in image_paths]
        self.augment = augment
        self.cache_images = None
        cache = root / "cache"
        if split != "test" and (cache / "names.json").exists():
            cached_names = json.loads((cache / "names.json").read_text())
            if set(cached_names) != set(names):
                raise ValueError("Stale training cache")
            self.cache_lookup = {name: i for i, name in enumerate(cached_names)}
            self.cache_images = np.load(cache / "images.npy", mmap_mode="r")
            self.cache_masks = np.load(cache / "masks.npy", mmap_mode="r")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        if self.cache_images is not None:
            j = self.cache_lookup[ip.name]
            image = np.array(self.cache_images[j])
            mask = self.cache_masks[j].astype(np.int64)
        else:
            image = cv2.imread(str(ip))
            mask = cv2.imread(str(mp), 0)
            if image is None or mask is None:
                raise ValueError(f"Unreadable pair {ip}, {mp}")
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            mask = (
                cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST) >= 128
            ).astype(np.int64)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augment:
            if np.random.random() < 0.5:
                image = image[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
            image = np.clip(
                image.astype(np.float32) * np.random.uniform(0.8, 1.2)
                + np.random.uniform(-12, 12),
                0,
                255,
            )
        return torch.from_numpy(np.ascontiguousarray(image)).permute(
            2, 0, 1
        ).float() / 255, torch.from_numpy(mask)
