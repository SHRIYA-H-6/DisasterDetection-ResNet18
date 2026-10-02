"""Stage 2: train the multitask pseudo-Siamese ResNet-18 classifier.

Overfitting safeguards:
  * dropout (0.5) before and after the shared layer of the classification heads
  * Adam with L2 weight decay
  * flip / rotation / brightness-contrast augmentation on the training split only
  * early stopping on validation loss (patience 7), best weights restored
  * progressive fine-tuning: encoders frozen, then layer4, layer3, layer2 and finally the stem + layer1 unfrozen
  * explicit overfitting detection (val loss rising while train loss falls) that stops training
  * optional stratified k-fold cross-validation over the development pool (train + val)
  * the train/val loss gap is written to training_summary.json and printed

    python -m disaster_detection.train_classifier [--cv-folds 5] [--max-epochs 40]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from . import config as C
from .dataset import PairDataset, balanced_sampler
from .models import MultitaskSiameseResNet18
from .training import TrainingMonitor, seed_everything

OUT_DIR = C.OUTPUT_ROOT / "classifier"


def run_epoch(model, loader, optimizer=None):
    """One pass over `loader`. Trains if an optimizer is given. Returns mean losses, accuracies and probabilities."""
    training = optimizer is not None
    model.train(training)
    totals = {"type": 0.0, "severity": 0.0}
    n = 0
    type_probs, sev_probs, type_true, sev_true, indices = [], [], [], [], []
    with torch.set_grad_enabled(training):
        for batch in loader:
            type_logits, sev_logits = model(batch["pre"], batch["post"])
            loss_type = F.cross_entropy(type_logits, batch["type"])
            loss_sev = F.cross_entropy(sev_logits, batch["severity"])
            loss = loss_type + loss_sev  # equal weighting
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            b = len(batch["type"])
            totals["type"] += loss_type.item() * b
            totals["severity"] += loss_sev.item() * b
            n += b
            type_probs.append(type_logits.softmax(1).detach())
            sev_probs.append(sev_logits.softmax(1).detach())
            type_true.append(batch["type"])
            sev_true.append(batch["severity"])
            indices.append(batch["index"])
    tp, sp = torch.cat(type_probs).numpy(), torch.cat(sev_probs).numpy()
    tt, st = torch.cat(type_true).numpy(), torch.cat(sev_true).numpy()
    return {
        "loss_type": totals["type"] / n, "loss_severity": totals["severity"] / n,
        "loss_total": (totals["type"] + totals["severity"]) / n,
        "acc_type": float((tp.argmax(1) == tt).mean()), "acc_severity": float((sp.argmax(1) == st).mean()),
        "f1_type": f1_score(tt, tp.argmax(1), average="macro"),
        "f1_severity": f1_score(st, sp.argmax(1), average="macro"),
        "type_probs": tp, "severity_probs": sp, "type_true": tt, "severity_true": st,
        "index": torch.cat(indices).numpy(),
    }


def make_optimizer(model, lr_head, lr_backbone, weight_decay):
    return torch.optim.Adam([
        {"params": model.head_parameters(), "lr": lr_head},
        {"params": model.backbone_parameters(), "lr": lr_backbone},
    ], weight_decay=weight_decay)


def train_model(train_df, val_df, args, log=print, tag="final"):
    """Train one model; returns (model with best weights, history DataFrame, summary dict)."""
    seed_everything(C.SEED)
    train_loader = DataLoader(PairDataset(train_df, train=True), batch_size=args.batch_size,
                              sampler=balanced_sampler(train_df))
    val_loader = DataLoader(PairDataset(val_df), batch_size=args.batch_size)
    clean_train_loader = DataLoader(PairDataset(train_df), batch_size=args.batch_size)

    model = MultitaskSiameseResNet18(pretrained=True, dropout=args.dropout)
    optimizer = make_optimizer(model, args.lr_head, args.lr_backbone, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    monitor = TrainingMonitor(patience=args.patience, window=args.overfit_window)
    best_state, history = None, []

    for epoch in range(1, args.max_epochs + 1):
        change = model.unfreeze_for_epoch(epoch)
        if change:
            log(f"[{tag}] epoch {epoch}: unfreezing {change}")
            if epoch > 1:
                monitor.mark_phase_change(epoch)
        start = time.time()
        tr = run_epoch(model, train_loader, optimizer)
        va = run_epoch(model, val_loader)
        scheduler.step(va["loss_total"])
        improved, stop = monitor.update(epoch, tr["loss_total"], va["loss_total"])
        if improved:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        row = {"epoch": epoch, "unfrozen_level": model.unfrozen_level,
               "lr_head": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - start, 1)}
        for split, res in (("train", tr), ("val", va)):
            for k in ("loss_type", "loss_severity", "loss_total", "acc_type", "acc_severity", "f1_type", "f1_severity"):
                row[f"{split}_{k}"] = res[k]
        row["gap_total"] = va["loss_total"] - tr["loss_total"]
        history.append(row)
        log(f"[{tag}] epoch {epoch:2d} | train loss {tr['loss_total']:.4f} (type {tr['loss_type']:.4f}, "
            f"sev {tr['loss_severity']:.4f}) | val loss {va['loss_total']:.4f} (type {va['loss_type']:.4f}, "
            f"sev {va['loss_severity']:.4f}) | gap {row['gap_total']:+.4f} | val acc type {va['acc_type']:.3f} "
            f"sev {va['acc_severity']:.3f} | {row['seconds']}s{' *' if improved else ''}")
        if stop:
            log(f"[{tag}] {monitor.stop_reason}")
            break
    else:
        monitor.stop_reason = f"Reached max epochs ({args.max_epochs}); best epoch {monitor.best_epoch}."
        log(f"[{tag}] {monitor.stop_reason}")

    model.load_state_dict(best_state)
    # Loss gap at the best epoch, measured cleanly: no augmentation, no dropout, same weights.
    clean_tr, clean_va = run_epoch(model, clean_train_loader), run_epoch(model, val_loader)
    best = history[monitor.best_epoch - 1]
    summary = {
        "tag": tag, "n_train": len(train_df), "n_val": len(val_df),
        "epochs_run": len(history), "best_epoch": monitor.best_epoch,
        "stop_reason": monitor.stop_reason, "overfitting_detected": monitor.overfitting_detected,
        "logged_at_best_epoch": {k: best[k] for k in best if k.startswith(("train_loss", "val_loss", "gap"))},
        "clean_eval_at_best_weights": {
            "train_loss_type": clean_tr["loss_type"], "train_loss_severity": clean_tr["loss_severity"],
            "train_loss_total": clean_tr["loss_total"],
            "val_loss_type": clean_va["loss_type"], "val_loss_severity": clean_va["loss_severity"],
            "val_loss_total": clean_va["loss_total"],
            "gap_type": clean_va["loss_type"] - clean_tr["loss_type"],
            "gap_severity": clean_va["loss_severity"] - clean_tr["loss_severity"],
            "gap_total": clean_va["loss_total"] - clean_tr["loss_total"],
            "train_acc_type": clean_tr["acc_type"], "val_acc_type": clean_va["acc_type"],
            "train_acc_severity": clean_tr["acc_severity"], "val_acc_severity": clean_va["acc_severity"],
            "val_f1_type_macro": clean_va["f1_type"], "val_f1_severity_macro": clean_va["f1_severity"],
        },
    }
    return model, pd.DataFrame(history), summary


def describe_gap(summary):
    g = summary["clean_eval_at_best_weights"]
    return (f"Train/val loss gap at best epoch {summary['best_epoch']} (clean eval, no augmentation/dropout): "
            f"total {g['gap_total']:+.4f} (train {g['train_loss_total']:.4f} vs val {g['val_loss_total']:.4f}); "
            f"type {g['gap_type']:+.4f}; severity {g['gap_severity']:+.4f}. "
            f"Overfitting flagged during training: {'YES' if summary['overfitting_detected'] else 'no'}.")


def cross_validate(dev_df, args, out_dir, log):
    folds = sorted(f for f in dev_df.cv_fold.unique() if f >= 0)[:args.cv_folds]
    results = []
    cv_args = argparse.Namespace(**{**vars(args), "max_epochs": args.cv_max_epochs})
    for fold in folds:
        tr, va = dev_df[dev_df.cv_fold != fold], dev_df[dev_df.cv_fold == fold]
        _, hist, summary = train_model(tr, va, cv_args, log, tag=f"cv fold {fold}")
        hist.to_csv(out_dir / f"cv_fold{fold}_history.csv", index=False)
        results.append(summary)
    keys = ["val_loss_total", "gap_total", "val_acc_type", "val_acc_severity", "val_f1_type_macro", "val_f1_severity_macro"]
    agg = {k: {"mean": float(np.mean([r["clean_eval_at_best_weights"][k] for r in results])),
               "std": float(np.std([r["clean_eval_at_best_weights"][k] for r in results]))} for k in keys}
    out = {"n_folds": len(folds), "max_epochs_per_fold": args.cv_max_epochs, "aggregate": agg, "folds": results}
    (out_dir / "cv_results.json").write_text(json.dumps(out, indent=2))
    log("Cross-validation (mean +/- std over folds): " +
        "; ".join(f"{k} {v['mean']:.4f} +/- {v['std']:.4f}" for k, v in agg.items()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", type=Path, default=C.METADATA_CSV)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--overfit-window", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--cv-folds", type=int, default=0, help="k-fold CV over train+val before the final model (0 = skip)")
    p.add_argument("--cv-max-epochs", type=int, default=20)
    p.add_argument("--limit", type=int, default=0, help="use at most N samples per split (smoke tests only)")
    args = p.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
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
    log(f"Train {len(train_df)} / val {len(val_df)} / test {(df.split == 'test').sum()} samples")

    if args.cv_folds:
        cross_validate(df[df.split.isin(["train", "val"])], args, args.out_dir, log)

    model, history, summary = train_model(train_df, val_df, args, log)
    torch.save({"state_dict": model.state_dict(), "dropout": args.dropout,
                "disaster_types": C.DISASTER_TYPES, "severities": C.SEVERITIES, "backbone": "resnet18"},
               args.out_dir / "best_model.pt")
    history.to_csv(args.out_dir / "history.csv", index=False)
    (args.out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    log(describe_gap(summary))

    from .plots import plot_loss_curves
    plot_loss_curves(history, summary, args.out_dir / "loss_curves.png")
    log(f"Saved model, history, summary and loss curves to {args.out_dir}")


if __name__ == "__main__":
    main()
