"""Train from scratch; select checkpoints using DUTS-TR validation Q only."""

try:
    from . import runtime
except ImportError:
    import runtime
import argparse, json, random, time
from pathlib import Path
import cv2, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from .dataset import DUTSDataset
    from .model import SmallFCN
    from .utils import Metrics, save_checkpoint, load_checkpoint
except ImportError:
    from dataset import DUTSDataset
    from model import SmallFCN
    from utils import Metrics, save_checkpoint, load_checkpoint


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    metrics = Metrics()
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu().numpy()
        for p, g in zip(pred, y.numpy()):
            metrics.update(p, g)
    return metrics.result()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default="data")
    p.add_argument("--output_dir", default="params")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--width_mult", type=float, choices=[0.5, 0.75, 1.0], default=1.0)
    p.add_argument("--distill_checkpoint", default=None)
    p.add_argument(
        "--initialize",
        default=None,
        help="Own-project checkpoint for a pruning/fine-tuning experiment",
    )
    a = p.parse_args()
    runtime.configure()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = DUTSDataset(a.data_root, "train", augment=True)
    val = DUTSDataset(a.data_root, "val")
    tl = DataLoader(train, batch_size=a.batch_size, shuffle=True, num_workers=0)
    vl = DataLoader(val, batch_size=a.batch_size, num_workers=0)
    model = SmallFCN(width_mult=a.width_mult).to(device)
    if a.initialize:
        model = load_checkpoint(model, a.initialize, device)
    teacher = (
        load_checkpoint(SmallFCN(), a.distill_checkpoint, device)
        if a.distill_checkpoint
        else None
    )
    if teacher is not None:
        teacher.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best = -1.0
    stale = 0
    start_epoch = 1
    if a.resume:
        state = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history = state["history"]
        best = state["best"]
        stale = state["stale"]
        start_epoch = state["epoch"] + 1
        random.setstate(state["random"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda"])
    print(
        f"Device {device}; train {len(train)}; validation {len(val)}; parameters {sum(v.numel() for v in model.parameters())}",
        flush=True,
    )
    for epoch in range(start_epoch, a.epochs + 1):
        begin = time.monotonic()
        model.train()
        total = 0.0
        for batch_index, (x, y) in enumerate(tl):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                ce = nn.functional.cross_entropy(logits, y)
                prob = logits.softmax(1)[:, 1]
                target = y.float()
                dice = (
                    1
                    - (
                        (2 * (prob * target).sum((1, 2)) + 1)
                        / (prob.sum((1, 2)) + target.sum((1, 2)) + 1)
                    ).mean()
                )
                loss = ce + dice
                if teacher is not None:
                    with torch.no_grad():
                        teacher_prob = teacher(x).float().softmax(1)
                    distill = (
                        -(teacher_prob * logits.float().log_softmax(1)).sum(1).mean()
                    )
                    loss = loss + 0.5 * distill
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(x)
            if batch_index % 100 == 0:
                print(
                    f"Epoch {epoch}, batch {batch_index}/{len(tl)}, loss {loss.item():.4f}",
                    flush=True,
                )
        scheduler.step()
        result = validate(model, vl, device)
        row = dict(
            epoch=epoch,
            loss=total / len(train),
            seconds=time.monotonic() - begin,
            **result,
        )
        history.append(row)
        if result["q"] > best:
            best = result["q"]
            stale = 0
            save_checkpoint(model, out / "model_fp16.pt")
        else:
            stale += 1
        torch.save(
            dict(
                model=model.state_dict(),
                optimizer=opt.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),
                history=history,
                best=best,
                stale=stale,
                epoch=epoch,
                random=random.getstate(),
                numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            ),
            out / "resume.pt",
        )
        (out / "training.json").write_text(
            json.dumps(
                dict(
                    config=vars(a),
                    device=str(device),
                    train=len(train),
                    validation=len(val),
                    history=history,
                    best_q=best,
                ),
                indent=2,
            )
        )
        print(json.dumps(row), flush=True)
        if stale >= a.patience:
            break
    print("Training finished; best validation Q", best, flush=True)


if __name__ == "__main__":
    main()
