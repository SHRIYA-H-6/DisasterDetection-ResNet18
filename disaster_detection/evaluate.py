"""Stage 7: full test-set evaluation from one set of test predictions.

Produces, in outputs/evaluation/:
  evaluation_report.txt              console tables (4 d.p.) for disaster type, severity and combined classes,
                                     segmentation metrics and the train/val loss gaps
  disaster_type_report.json          sklearn classification_report, 10 d.p.
  severity_report.json
  combined_report.json
  confusion_disaster_type.png        3x3
  confusion_severity.png             3x3
  confusion_combined.png             9x9
  test_predictions.csv               batch-level summary of every test image
  segmentation_test_metrics.json
  source_breakdown.csv               accuracy per data source / post-image modality (SAR shortcut check)

    python -m disaster_detection.evaluate
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from . import config as C
from .area import area_statistics
from .plots import plot_confusion_matrix
from .predict import CLASSIFIER_CKPT, UNET_CKPT, DisasterPredictor
from .preprocess import cache_path

MODEL_NAME = "PSEUDO-SIAMESE RESNET-18"
RULE = "=" * 59


def round_floats(obj, digits=10):
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: round_floats(v, digits) for k, v in obj.items()}
    return obj


def format_report(report, names, title):
    width = max(len(n) for n in names + ["weighted avg"]) + 2
    head = f"    {'':<{width}}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}"
    lines = [f"--- {title} ---", head, ""]
    for n in names:
        r = report[n]
        lines.append(f"    {n:<{width}}{r['precision']:>10.4f}{r['recall']:>10.4f}{r['f1-score']:>10.4f}"
                     f"{int(r['support']):>10d}")
    total = int(report["macro avg"]["support"])
    lines += ["", f"    {'accuracy':<{width}}{'':>10}{'':>10}{report['accuracy']:>10.4f}{total:>10d}"]
    for avg in ("macro avg", "weighted avg"):
        r = report[avg]
        lines.append(f"    {avg:<{width}}{r['precision']:>10.4f}{r['recall']:>10.4f}{r['f1-score']:>10.4f}"
                     f"{int(r['support']):>10d}")
    return lines


def source_breakdown_lines(df, out_dir):
    """Accuracy per source / post-image modality. BRIGHT post images are SAR, xBD are optical, so a type head that
    keys on modality shows up as near-perfect BRIGHT accuracy next to weak xBD-earthquake accuracy."""
    g = df.groupby(["true_disaster", "source", "post_modality"])
    table = g.agg(n=("sample_id", "size"), acc_disaster=("correct_disaster", "mean"),
                  acc_severity=("correct_severity", "mean"), acc_combined=("correct_combined", "mean")).reset_index()
    table.to_csv(out_dir / "source_breakdown.csv", index=False)
    lines = ["--- ACCURACY BY DATA SOURCE (modality shortcut check) ---",
             f"    {'true type':<12}{'source':<8}{'post img':<10}{'n':>5}{'type acc':>10}{'sev acc':>10}{'comb acc':>10}"]
    for r in table.itertuples():
        lines.append(f"    {r.true_disaster:<12}{r.source:<8}{r.post_modality:<10}{r.n:>5d}{r.acc_disaster:>10.4f}"
                     f"{r.acc_severity:>10.4f}{r.acc_combined:>10.4f}")
    return lines + [""]


def loss_gap_lines(output_root):
    lines = ["--- TRAIN / VALIDATION LOSS GAP (overfitting check) ---"]
    cls_summary = output_root / "classifier" / "training_summary.json"
    if cls_summary.exists():
        s = json.loads(cls_summary.read_text())
        g = s["clean_eval_at_best_weights"]
        lines += [
            f"    Classifier best epoch {s['best_epoch']} of {s['epochs_run']} run; {s['stop_reason']}",
            f"    {'':<16}{'train':>10}{'val':>10}{'gap':>10}",
            f"    {'type loss':<16}{g['train_loss_type']:>10.4f}{g['val_loss_type']:>10.4f}{g['gap_type']:>+10.4f}",
            f"    {'severity loss':<16}{g['train_loss_severity']:>10.4f}{g['val_loss_severity']:>10.4f}"
            f"{g['gap_severity']:>+10.4f}",
            f"    {'combined loss':<16}{g['train_loss_total']:>10.4f}{g['val_loss_total']:>10.4f}{g['gap_total']:>+10.4f}",
            f"    Overfitting flagged during classifier training: {'YES' if s['overfitting_detected'] else 'no'}",
        ]
    cv = output_root / "classifier" / "cv_results.json"
    if cv.exists():
        agg = json.loads(cv.read_text())["aggregate"]
        lines.append(f"    {json.loads(cv.read_text())['n_folds']}-fold CV (dev pool): " + "; ".join(
            f"{k} {v['mean']:.4f} +/- {v['std']:.4f}" for k, v in agg.items()))
    seg_summary = output_root / "segmentation" / "training_summary.json"
    if seg_summary.exists():
        s = json.loads(seg_summary.read_text())
        lines += [f"    U-Net best epoch {s['best_epoch']} of {s['epochs_run']}: train loss "
                  f"{s['clean_train']['loss']:.4f}, val loss {s['clean_val']['loss']:.4f}, gap {s['gap_loss']:+.4f}; "
                  f"overfitting flagged: {'YES' if s['overfitting_detected'] else 'no'}"]
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", type=Path, default=C.METADATA_CSV)
    p.add_argument("--out-dir", type=Path, default=C.OUTPUT_ROOT / "evaluation")
    p.add_argument("--classifier", type=Path, default=CLASSIFIER_CKPT)
    p.add_argument("--unet", type=Path, default=UNET_CKPT)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(args.metadata).query("split == 'test'").reset_index(drop=True)
    predictor = DisasterPredictor(args.classifier, args.unet)
    rows, seg_conf = [], np.zeros((3, 3), dtype=np.int64)
    for row in test.to_dict("records"):
        data = np.load(cache_path(row["sample_id"]))
        pred = predictor.predict(data["pre"], data["post"])
        seg_true = data["seg"]
        seg_conf += np.bincount((seg_true.astype(np.int64) * 3 + pred["segmentation"]).ravel(),
                                minlength=9).reshape(3, 3)
        truth = area_statistics(seg_true)
        rows.append({
            "sample_id": row["sample_id"], "source": row["source"], "event": row["event"],
            "true_disaster": row["disaster_type"], "pred_disaster": pred["disaster_type"],
            "disaster_confidence": pred["disaster_confidence"],
            "true_severity": row["severity"], "pred_severity": pred["severity"],
            "severity_confidence": pred["severity_confidence"],
            "true_combined": row["combined_class"], "pred_combined": pred["combined_class"],
            **{f"p_{k}": v for k, v in pred["disaster_probabilities"].items()},
            **{f"p_{k}": v for k, v in pred["severity_probabilities"].items()},
            "pred_affected_area_pct": pred["affected_area_pct"], "true_affected_area_pct": truth["affected_area_pct"],
            "pred_affected_structure_pct": pred["affected_structure_pct"],
            "severity_measure": row["severity_measure"], "true_severity_value": row["severity_value"],
            "correct_disaster": row["disaster_type"] == pred["disaster_type"],
            "correct_severity": row["severity"] == pred["severity"],
            "correct_combined": row["combined_class"] == pred["combined_class"],
            "post_modality": row["post_modality"], "center_lat": row["center_lat"], "center_lon": row["center_lon"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "test_predictions.csv", index=False)

    specs = [
        ("disaster_type", "true_disaster", "pred_disaster", C.DISASTER_TYPES, "DISASTER TYPE CLASSIFICATION REPORT"),
        ("severity", "true_severity", "pred_severity", C.SEVERITIES, "SEVERITY LEVEL CLASSIFICATION REPORT"),
        ("combined", "true_combined", "pred_combined", C.COMBINED_CLASSES,
         "COMBINED DISASTER + SEVERITY CLASSIFICATION REPORT"),
    ]
    lines = [RULE, f"{'MULTITASK ' + MODEL_NAME + ' TEST EVALUATION':^59}".rstrip(), RULE]
    for key, true_col, pred_col, names, title in specs:
        report = classification_report(df[true_col], df[pred_col], labels=names, target_names=names,
                                       output_dict=True, zero_division=0)
        (args.out_dir / f"{key}_report.json").write_text(json.dumps(round_floats(report), indent=2))
        lines += format_report(report, names, title) + [""]
        cm = confusion_matrix(df[true_col], df[pred_col], labels=names)
        plot_confusion_matrix(cm, names, f"Confusion matrix - {key.replace('_', ' ')} (test, n={len(df)})",
                              args.out_dir / f"confusion_{key}.png")

    iou = np.diag(seg_conf) / np.maximum(seg_conf.sum(0) + seg_conf.sum(1) - np.diag(seg_conf), 1)
    tp = seg_conf[C.SEG_AFFECTED, C.SEG_AFFECTED]
    seg_metrics = {
        "class_names": C.SEG_CLASS_NAMES, "iou": iou.tolist(), "mean_iou": float(iou.mean()),
        "affected_precision": float(tp / max(seg_conf[:, C.SEG_AFFECTED].sum(), 1)),
        "affected_recall": float(tp / max(seg_conf[C.SEG_AFFECTED].sum(), 1)),
        "affected_area_mae_pct_points": float((df.pred_affected_area_pct - df.true_affected_area_pct).abs().mean()),
        "affected_area_correlation": float(df.pred_affected_area_pct.corr(df.true_affected_area_pct)),
        "pixel_confusion": seg_conf.tolist(),
    }
    seg_metrics["affected_f1"] = (2 * seg_metrics["affected_precision"] * seg_metrics["affected_recall"]
                                  / max(seg_metrics["affected_precision"] + seg_metrics["affected_recall"], 1e-12))
    (args.out_dir / "segmentation_test_metrics.json").write_text(json.dumps(round_floats(seg_metrics), indent=2))
    lines += ["--- U-NET SEGMENTATION / AFFECTED-AREA (test) ---",
              *[f"    IoU {n:<32}{v:.4f}" for n, v in zip(C.SEG_CLASS_NAMES, iou)],
              f"    mean IoU{'':<29}{seg_metrics['mean_iou']:.4f}",
              f"    affected precision / recall / F1   {seg_metrics['affected_precision']:.4f} / "
              f"{seg_metrics['affected_recall']:.4f} / {seg_metrics['affected_f1']:.4f}",
              f"    affected-area MAE (pct points)     {seg_metrics['affected_area_mae_pct_points']:.4f}",
              ""]
    lines += source_breakdown_lines(df, args.out_dir)
    lines += loss_gap_lines(C.OUTPUT_ROOT)
    missing = [c for c in C.COMBINED_CLASSES if c not in set(df.true_combined)]
    if missing:
        lines += ["", f"WARNING: zero test support for {len(missing)} class(es): {', '.join(missing)}. "
                      f"This evaluation is NOT a complete {len(C.COMBINED_CLASSES)}-class result."]
    lines.append(RULE)
    text = "\n".join(lines)
    print(text)
    (args.out_dir / "evaluation_report.txt").write_text(text + "\n")
    print(f"\nSaved reports, confusion matrices and test_predictions.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
