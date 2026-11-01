"""Assignment metrics: accumulated confusion, four-neighbour edges, 5x5 tolerance."""

import os
import numpy as np
import torch
import cv2


def compute_miou(pred, gt):
    p = np.asarray(pred).reshape(-1).astype(np.int64)
    g = np.asarray(gt).reshape(-1).astype(np.int64)
    return miou_from_confusion(np.bincount(2 * g + p, minlength=4).reshape(2, 2))


def miou_from_confusion(c):
    intersection = np.diag(c).astype(float)
    union = c.sum(0) + c.sum(1) - intersection
    return float(
        np.divide(intersection, union, out=np.zeros(2), where=union > 0).mean()
    )


def _boundary_map(mask):
    m = np.asarray(mask, dtype=bool)
    p = np.pad(m, 1)
    return m & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])


def _window(b):
    # Separable logical max over 5x5, implemented directly, no morphology API.
    h, w = b.shape
    p = np.pad(b, ((0, 0), (2, 2)))
    x = np.zeros_like(b)
    for k in range(5):
        x |= p[:, k : k + w]
    p = np.pad(x, ((2, 2), (0, 0)))
    y = np.zeros_like(b)
    for k in range(5):
        y |= p[k : k + h, :]
    return y


def compute_boundary_f1(pred, gt):
    p = _boundary_map(pred)
    g = _boundary_map(gt)
    npred = int(p.sum())
    ngt = int(g.sum())
    if not npred or not ngt:
        return float(npred == ngt)
    precision = (p & _window(g)).sum() / npred
    recall = (g & _window(p)).sum() / ngt
    return (
        float(2 * precision * recall / (precision + recall))
        if precision + recall
        else 0.0
    )


class Metrics:
    """Stream samples without retaining all predictions in memory.

    Rows of confusion are ground truth; columns are predictions. This
    accumulates pixel counts before computing either class IoU.
    """

    def __init__(self):
        self.confusion = np.zeros((2, 2), dtype=np.int64)
        self.bf1 = 0.0
        self.n = 0

    def update(self, p, g):
        self.confusion += np.bincount(
            (2 * g.astype(np.int64) + p).ravel(), minlength=4
        ).reshape(2, 2)
        self.bf1 += compute_boundary_f1(p, g)
        self.n += 1

    def result(self):
        if not self.n:
            raise ValueError("Cannot evaluate an empty dataset")
        miou = miou_from_confusion(self.confusion)
        bf1 = self.bf1 / self.n
        return dict(
            n=self.n,
            miou=miou,
            bf1=bf1,
            q=0.7 * miou + 0.3 * bf1,
            confusion=self.confusion.tolist(),
        )


def ces_score(q, t_cpu_ms, s_disk_bytes):
    return (
        q * (1 - 0.2 * t_cpu_ms / 50 - 0.1 * min(1, s_disk_bytes / 4194304))
        if t_cpu_ms <= 50
        else 0.0
    )


def save_checkpoint(model, path):
    state = {
        k: (v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu())
        for k, v in model.state_dict().items()
    }
    if getattr(model, "width_mult", 1.0) != 1.0:
        torch.save(
            dict(state_dict=state, architecture=dict(width_mult=model.width_mult)), path
        )
    else:
        torch.save(state, path)


def flatten_checkpoint(saved):
    """Concatenate raw same-dtype tensors to reduce archive headers, without compression."""
    if "flat_state" in saved:
        return saved
    chunks = {}
    counts = {}
    entries = []

    def append(tensor):
        dtype = str(tensor.dtype).split(".")[-1]
        offset = counts.get(dtype, 0)
        chunks.setdefault(dtype, []).append(tensor.reshape(-1))
        counts[dtype] = offset + tensor.numel()
        return dict(
            dtype=dtype, offset=offset, numel=tensor.numel(), shape=list(tensor.shape)
        )

    for name, tensor in saved["state_dict"].items():
        entries.append(dict(name=name, value=append(tensor)))
    for name, item in saved.get("quantized", {}).items():
        entries.append(
            dict(name=name, value=append(item["values"]), scale=append(item["scale"]))
        )
    result = {k: v for k, v in saved.items() if k not in ("state_dict", "quantized")}
    result.update(
        flat_state=entries, payload={k: torch.cat(v) for k, v in chunks.items()}
    )
    return result


def load_checkpoint(model, path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    width = saved.get("architecture", {}).get("width_mult", 1.0)
    if getattr(model, "width_mult", 1.0) != width:
        model = type(model)(width_mult=width)
    config = saved.get("inference", {})
    if "flat_state" in saved:

        def retrieve(spec):
            return (
                saved["payload"][spec["dtype"]]
                .narrow(0, spec["offset"], spec["numel"])
                .reshape(spec["shape"])
            )

        state = {}
        for entry in saved["flat_state"]:
            value = retrieve(entry["value"])
            state[entry["name"]] = (
                value.float() * retrieve(entry["scale"]).float()
                if "scale" in entry
                else value
            )
    else:
        state = dict(saved["state_dict"]) if "state_dict" in saved else saved
        for name, item in saved.get("quantized", {}).items():
            state[name] = item["values"].float() * item["scale"].float()
    model.load_state_dict(state)
    model.input_size = int(config.get("input_size", 160))
    model.inference_config = config
    return model.to(device).eval()
