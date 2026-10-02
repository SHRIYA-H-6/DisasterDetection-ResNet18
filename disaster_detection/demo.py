"""End-to-end demo on a handful of real held-out test images.

For each sample, reads the ORIGINAL image files (not the training cache), then runs
prediction -> summary report (console + .txt + .json) -> Matplotlib overlay -> Folium map.

    python -m disaster_detection.demo [--per-type 2]
"""
import argparse
from pathlib import Path

import pandas as pd

from . import config as C
from .predict import DisasterPredictor
from .preprocess import load_sample
from .report import build_report, save_report
from .visualize import folium_map, plot_overlay


def pick_samples(test, per_type, seed=C.SEED):
    """Up to `per_type` test samples per disaster type, preferring a spread of severities."""
    picks = []
    for disaster in C.DISASTER_TYPES:
        sub = test[test.disaster_type == disaster].sample(frac=1, random_state=seed)
        sub = sub.sort_values("severity", key=lambda s: s.map({"Severe": 0, "Moderate": 1, "Low": 2}), kind="stable")
        picks.append(sub.drop_duplicates("severity").head(per_type))
    return pd.concat(picks)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", type=Path, default=C.METADATA_CSV)
    p.add_argument("--out-dir", type=Path, default=C.OUTPUT_ROOT / "demo")
    p.add_argument("--per-type", type=int, default=2)
    p.add_argument("--sample-ids", nargs="*", help="explicit sample_ids instead of automatic picks")
    args = p.parse_args()

    df = pd.read_csv(args.metadata)
    test = df[df.split == "test"]
    samples = test[test.sample_id.isin(args.sample_ids)] if args.sample_ids else pick_samples(test, args.per_type)
    predictor = DisasterPredictor()
    for row in samples.to_dict("records"):
        out = args.out_dir / row["sample_id"]
        out.mkdir(parents=True, exist_ok=True)
        pre, post, seg_true = load_sample(row)
        pred = predictor.predict(pre, post)
        text = build_report(pred)
        print("=" * 59)
        print(f"Sample: {row['sample_id']}  ({row['source']}, ground truth {row['combined_class']}, "
              f"labelled affected area {100 * (seg_true == C.SEG_AFFECTED).mean():.2f}%)")
        print("-" * 59)
        print(text)
        save_report(pred, text, out / "summary_report",
                    extra={"sample_id": row["sample_id"], "source": row["source"],
                           "ground_truth_combined_class": row["combined_class"]})
        plot_overlay(pre, post, pred["segmentation"], out / "affected_regions.png", seg_true=seg_true,
                     title=f"{row['sample_id']}: predicted {pred['combined_class']} "
                           f"(truth {row['combined_class']}), affected area {pred['affected_area_pct']:.2f}%")
        if folium_map(row, post, pred["segmentation"], out / "affected_regions_map.html", text, seg_true=seg_true):
            print(f"Map: {out / 'affected_regions_map.html'}")
        else:
            print("Map: no georeferencing for this source, Folium map skipped")
    print("=" * 59)
    print(f"Demo outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
