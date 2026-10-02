"""Stage 3: train the Siamese ResNet-18 U-Net that delineates damaged regions in pre/post image pairs.

Targets come from the real pixel-level labels: xBD damage targets (no-damage /
un-classified -> intact; minor / major / destroyed -> affected) and BRIGHT damage
targets (intact -> intact; damaged / destroyed -> affected). Loss is class-weighted
cross-entropy plus soft Dice on the affected class. The same safeguards as the
classifier apply: train-only augmentation, dropout in the deep decoder blocks, Adam
with weight decay (lower learning rate on the ImageNet-pretrained encoder), early
stopping and overfitting detection on validation loss.

    python -m disaster_detection.train_segmentation [--max-epochs 40]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import config as C
from .area import affected_area_percent
from .dataset import PairDataset, balanced_sampler
from .models import SiameseResNetUNet
from .training import TrainingMonitor, seed_everything

OUT_DIR = C.OUTPUT_ROOT / "segmentation"


def pixel_class_weights(df):
    """Inverse-sqrt pixel frequency of each segmentation class over the training tiles."""
    counts = np.zeros(3)
    for sample_id in df.sample_id:
        seg = np.load(C.CACHE_DIR / f"{sample_id}.npz")["seg"]
        counts += np.bincount(seg.ravel(), minlength=3)
    freq = counts / counts.sum()
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-6))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def seg_loss(logits, target, weights):
    ce = F.cross_entropy(logits, target, weight=weights)
    prob = logits.softmax(1)[:, C.SEG_AFFECTED]
    tgt = (target == C.SEG_AFFECTED).float()
    inter = (prob * tgt).sum((1, 2))
    dice = 1 - (2 * inter + 1) / (prob.sum((1, 2)) + tgt.sum((1, 2)) + 1)
    return ce + dice.mean()


def run_epoch(model, loader, weights, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    conf = np.zeros((3, 3), dtype=np.int64)
    area_pred, area_true = [], []
    with torch.set_grad_enabled(training):
        for batch in loader:
            logits = model(batch["pre"], batch["post"])
            loss = seg_loss(logits, batch["seg"], weights)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            b = len(batch["seg"])
            total += loss.item() * b
            n += b
            pred = logits.argmax(1)
            conf += np.bincount((batch["seg"] * 3 + pred).numpy().ravel(), minlength=9).reshape(3, 3)
            area_pred += [affected_area_percent(p) for p in pred.numpy()]
            area_true += [affected_area_percent(t) for t in batch["seg"].numpy()]
    iou = np.diag(conf) / np.maximum(conf.sum(0) + conf.sum(1) - np.diag(conf), 1)
    tp = conf[C.SEG_AFFECTED, C.SEG_AFFECTED]
    f1_affected = 2 * tp / max(conf[:, C.SEG_AFFECTED].sum() + conf[C.SEG_AFFECTED].sum(), 1)
    return {"loss": total / n, "iou": iou.tolist(), "miou": float(iou.mean()), "f1_affected": float(f1_affected),
            "area_mae_pct": float(np.mean(np.abs(np.array(area_pred) - np.array(area_true)))), "confusion": conf.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", type=Path, default=C.METADATA_CSV)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3, help="decoder learning rate")
    p.add_argument("--lr-encoder", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--limit", type=int, default=0, help="use at most N samples per split (smoke tests only)")
    args = p.parse_args()

    seed_everything(C.SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(args.out_dir / "training_log.txt", "w")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    df = pd.read_csv(args.metadata)
    if args.limit:
        df = pd.concat([g.sample(min(len(g), args.limit), random_state=C.SEED) for _, g in df.groupby("split")])
    train_df, val_df = df[df.split == "train"], df[df.split == "val"]
    weights = pixel_class_weights(train_df)
    log(f"Train {len(train_df)} / val {len(val_df)} tiles; pixel class weights {weights.numpy().round(3).tolist()}")

    train_loader = DataLoader(PairDataset(train_df, train=True), batch_size=args.batch_size,
                              sampler=balanced_sampler(train_df))
    val_loader = DataLoader(PairDataset(val_df), batch_size=args.batch_size)
    model = SiameseResNetUNet(pretrained=True)
    optimizer = torch.optim.Adam([{"params": model.decoder_parameters(), "lr": args.lr},
                                  {"params": model.encoder_parameters(), "lr": args.lr_encoder}],
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    monitor = TrainingMonitor(patience=args.patience)
    best_state, history = None, []
    for epoch in range(1, args.max_epochs + 1):
        start = time.time()
        tr = run_epoch(model, train_loader, weights, optimizer)
        va = run_epoch(model, val_loader, weights)
        scheduler.step(va["loss"])
        improved, stop = monitor.update(epoch, tr["loss"], va["loss"])
        if improved:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch, "train_loss": tr["loss"], "val_loss": va["loss"],
                        "gap": va["loss"] - tr["loss"], "train_miou": tr["miou"], "val_miou": va["miou"],
                        "val_iou_affected": va["iou"][C.SEG_AFFECTED], "val_f1_affected": va["f1_affected"],
                        "val_area_mae_pct": va["area_mae_pct"], "seconds": round(time.time() - start, 1)})
        log(f"[seg] epoch {epoch:2d} | train loss {tr['loss']:.4f} | val loss {va['loss']:.4f} | "
            f"gap {va['loss'] - tr['loss']:+.4f} | val mIoU {va['miou']:.3f} | IoU affected "
            f"{va['iou'][C.SEG_AFFECTED]:.3f} | area MAE {va['area_mae_pct']:.2f} pp | "
            f"{history[-1]['seconds']}s{' *' if improved else ''}")
        if stop:
            log(f"[seg] {monitor.stop_reason}")
            break
    else:
        monitor.stop_reason = f"Reached max epochs ({args.max_epochs}); best epoch {monitor.best_epoch}."
        log(f"[seg] {monitor.stop_reason}")

    model.load_state_dict(best_state)
    clean_tr = run_epoch(model, DataLoader(PairDataset(train_df), batch_size=args.batch_size), weights)
    clean_va = run_epoch(model, val_loader, weights)
    summary = {"best_epoch": monitor.best_epoch, "epochs_run": len(history), "stop_reason": monitor.stop_reason,
               "overfitting_detected": monitor.overfitting_detected, "architecture": "siamese resnet18 u-net",
               "pixel_class_weights": weights.tolist(), "class_names": C.SEG_CLASS_NAMES,
               "clean_train": {k: clean_tr[k] for k in ("loss", "iou", "miou", "f1_affected", "area_mae_pct")},
               "clean_val": {k: clean_va[k] for k in ("loss", "iou", "miou", "f1_affected", "area_mae_pct", "confusion")},
               "gap_loss": clean_va["loss"] - clean_tr["loss"]}
    torch.save({"state_dict": model.state_dict(), "architecture": "siamese resnet18 u-net"},
               args.out_dir / "best_unet.pt")
    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False)
    (args.out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"U-Net train/val loss gap at best epoch {monitor.best_epoch} (clean eval): {summary['gap_loss']:+.4f} "
        f"(train {clean_tr['loss']:.4f} vs val {clean_va['loss']:.4f}). Val IoU affected "
        f"{clean_va['iou'][C.SEG_AFFECTED]:.4f}, mIoU {clean_va['miou']:.4f}, affected-area MAE "
        f"{clean_va['area_mae_pct']:.2f} percentage points.")

    from .plots import plot_simple_loss_curve
    plot_simple_loss_curve(pd.DataFrame(history), summary, "U-Net segmentation training vs validation loss",
                           args.out_dir / "loss_curves.png")


if __name__ == "__main__":
    main()
