"""Structured 3/4-width pruning of the project's own randomly trained teacher.

Channel choices use parameter magnitudes only, not test images or labels.
"""

from src import runtime

runtime.configure()
from pathlib import Path
import torch, json
from src.model import SmallFCN

source = torch.load(
    "experiments/baseline/model_fp16.pt", map_location="cpu", weights_only=True
)
target = SmallFCN(width_mult=0.75).state_dict()


def ranked(values, n):
    return values.topk(n).indices.sort().values


outputs = {}


def block(name, input_indices):
    n = target[name + ".0.weight"].shape[0]
    hidden = ranked(
        source[name + ".0.weight"].float().abs().mean((1, 2, 3))
        * source[name + ".1.weight"].float().abs(),
        n,
    )
    out = ranked(
        source[name + ".6.weight"].float().abs().mean((1, 2, 3))
        * source[name + ".7.weight"].float().abs(),
        n,
    )
    target[name + ".0.weight"] = source[name + ".0.weight"][hidden][:, input_indices]
    target[name + ".3.weight"] = source[name + ".3.weight"][hidden]
    target[name + ".6.weight"] = source[name + ".6.weight"][out][:, hidden]
    for bn, idx in [("1", hidden), ("4", hidden), ("7", out)]:
        for suffix in ["weight", "bias", "running_mean", "running_var"]:
            target[f"{name}.{bn}.{suffix}"] = source[f"{name}.{bn}.{suffix}"][idx]
        target[f"{name}.{bn}.num_batches_tracked"] = source[
            f"{name}.{bn}.num_batches_tracked"
        ]
    outputs[name] = out
    return out


idx = torch.arange(3)
for name in ["e1", "e2", "e3", "e4", "e5"]:
    idx = block(name, idx)
target["context.weight"] = source["context.weight"][idx][:, idx]
target["context.bias"] = source["context.bias"][idx]
idx = block("d4", torch.cat([outputs["e5"], outputs["e4"] + 128]))
idx = block("d3", torch.cat([idx, outputs["e3"] + 64]))
idx = block("d2", torch.cat([idx, outputs["e2"] + 48]))
idx = block("d1", torch.cat([idx, outputs["e1"] + 32]))
target["head.weight"] = source["head.weight"][:, idx]
target["head.bias"] = source["head.bias"]
Path("params/pruned").mkdir(exist_ok=True)
torch.save(
    dict(state_dict=target, architecture=dict(width_mult=0.75)),
    "params/pruned/initial.pt",
)
print(
    "Pruned parameters", sum(p.numel() for p in SmallFCN(width_mult=0.75).parameters())
)
