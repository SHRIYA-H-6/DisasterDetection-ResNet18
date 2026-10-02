"""Stage 1: scan the real datasets, derive labels, split, and enforce minimum samples per class.

Writes data/metadata.csv with one row per sample. Exits with an error (and a
report of what was found and which sources were checked) if any of the 9
combined classes has fewer than MIN_SAMPLES_FLOOR real samples.

    python -m disaster_detection.build_metadata [--data-roots PATH [PATH ...]]
"""
import argparse
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import config as C
from .geo import apply_gdal_transform, geotiff_transform, to_lonlat

DAMAGED = ("minor-damage", "major-damage", "destroyed")
CLASSIFIED = ("no-damage",) + DAMAGED


def load_geotransforms(roots):
    """xBD tile geotransforms (from xview_geotransforms.json.tgz in any data root)."""
    transforms = {}
    for root in roots:
        archive = root / C.GEOTRANSFORMS_NAME
        if archive.exists():
            with tarfile.open(archive) as tar:
                member = next(m for m in tar.getmembers() if m.name.endswith(".json"))
                transforms.update(json.load(tar.extractfile(member)))
    return transforms


def scan_xbd(root, geotransforms):
    """xBD tiles of the configured events in any <root>/<split>/labels folder (train, tier3, hold, test)."""
    rows = []
    for label_dir in sorted(root.glob("*/labels")):
        split_dir = label_dir.parent.name
        for label_path in sorted(label_dir.glob("*_post_disaster.json")):
            stem = label_path.name.replace("_post_disaster.json", "")
            event = stem.rsplit("_", 1)[0]
            if event not in C.XBD_EVENTS:
                continue
            pre = root / split_dir / "images" / f"{stem}_pre_disaster.png"
            post = root / split_dir / "images" / f"{stem}_post_disaster.png"
            target = root / split_dir / "targets" / f"{stem}_post_disaster_target.png"
            if not (pre.exists() and post.exists() and target.exists()):
                continue
            label = json.loads(label_path.read_text())
            counts = Counter(f["properties"].get("subtype", "un-classified") for f in label["features"]["xy"])
            n_classified = sum(counts[k] for k in CLASSIFIED)
            n_damaged = sum(counts[k] for k in DAMAGED)
            if n_classified == 0:  # no buildings with a damage label -> severity undefined
                continue
            fraction = n_damaged / n_classified
            disaster = C.XBD_EVENTS[event]
            severity = C.severity_from_fraction(fraction)
            lon = lat = np.nan
            geo = geotransforms.get(post.name)
            geo_transform, geo_epsg = "", ""
            if geo:
                lon, lat = apply_gdal_transform(geo[0], 512, 512)
                geo_transform, geo_epsg = json.dumps(list(geo[0])), 4326
            rows.append(dict(
                sample_id=stem, source="xBD", data_root=root.name, event=event,
                disaster_type=disaster, severity=severity, combined_class=f"{disaster}_{severity}",
                severity_measure="damaged_building_fraction", severity_value=fraction,
                n_buildings=n_classified, n_damaged=n_damaged,
                n_minor=counts["minor-damage"], n_major=counts["major-damage"], n_destroyed=counts["destroyed"],
                pre_path=str(pre.relative_to(root)), post_path=str(post.relative_to(root)),
                target_path=str(target.relative_to(root)), post_modality="optical",
                center_lon=lon, center_lat=lat, geo_transform=geo_transform, geo_epsg=geo_epsg,
                orig_width=label["metadata"]["width"], orig_height=label["metadata"]["height"],
                capture_date=label["metadata"].get("capture_date", ""),
            ))
    return rows


def scan_bright(root):
    """BRIGHT earthquake tiles: pre-event optical, post-event SAR, target 0 bg / 1 intact / 2 damaged / 3 destroyed."""
    import tifffile
    rows = []
    target_dir = root / "bright" / "target"
    if not target_dir.exists():
        return rows
    for target_path in sorted(target_dir.glob("*_building_damage.tif")):
        stem = target_path.name.replace("_building_damage.tif", "")
        event = stem.rsplit("_", 1)[0]
        if C.BRIGHT_EVENT_FILTER not in event:
            continue
        pre = root / "bright" / "pre-event" / f"{stem}_pre_disaster.tif"
        post = root / "bright" / "post-event" / f"{stem}_post_disaster.tif"
        if not (pre.exists() and post.exists()):
            continue
        target = tifffile.imread(target_path)
        building_px = int((target >= 1).sum())
        if building_px == 0:
            continue
        damaged_px = int((target >= 2).sum())
        fraction = damaged_px / building_px
        severity = C.severity_from_fraction(fraction)
        lon = lat = np.nan
        h, w = target.shape[:2]
        transform, epsg = geotiff_transform(target_path)
        if transform is not None:
            lon, lat = to_lonlat(*apply_gdal_transform(transform, w / 2, h / 2), epsg)
        rows.append(dict(
            sample_id=stem, source="BRIGHT", data_root=root.name, event=event,
            disaster_type="Earthquake", severity=severity, combined_class=f"Earthquake_{severity}",
            severity_measure="damaged_building_pixel_fraction", severity_value=fraction,
            n_buildings=building_px, n_damaged=damaged_px, n_minor=0,
            n_major=int((target == 2).sum()), n_destroyed=int((target == 3).sum()),
            pre_path=str(pre.relative_to(root)), post_path=str(post.relative_to(root)),
            target_path=str(target_path.relative_to(root)), post_modality="SAR",
            center_lon=lon, center_lat=lat,
            geo_transform=json.dumps(list(transform)) if transform is not None else "", geo_epsg=epsg or "",
            orig_width=w, orig_height=h, capture_date="",
        ))
    return rows


def assign_splits(df, test_size=0.2, val_size=0.2, n_folds=5):
    idx = np.arange(len(df))
    dev, test = train_test_split(idx, test_size=test_size, stratify=df.combined_class, random_state=C.SEED)
    train, val = train_test_split(dev, test_size=val_size, stratify=df.combined_class.iloc[dev], random_state=C.SEED)
    split = np.empty(len(df), dtype=object)
    split[train], split[val], split[test] = "train", "val", "test"
    df["split"] = split
    # K-fold assignment over the development pool (train + val); the test split is never used for CV.
    df["cv_fold"] = -1
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=C.SEED)
    for fold, (_, fold_idx) in enumerate(skf.split(dev, df.combined_class.iloc[dev])):
        df.loc[df.index[dev[fold_idx]], "cv_fold"] = fold
    return df


def enforce_minimums(df, all_counts, roots):
    counts = all_counts.reindex(C.COMBINED_CLASSES, fill_value=0)
    split_counts = pd.crosstab(df.combined_class, df.split).reindex(C.COMBINED_CLASSES, fill_value=0)
    print("\nReal samples per combined class")
    print(f"  {'class':22s} {'total':>6s} {'train':>6s} {'val':>6s} {'test':>6s}  status")
    failures = []
    for cls in C.COMBINED_CLASSES:
        n = int(counts[cls])
        if n < C.MIN_SAMPLES_FLOOR:
            status = f"FAIL (< floor {C.MIN_SAMPLES_FLOOR})"
            failures.append((cls, n))
        elif n < C.MIN_SAMPLES_TARGET:
            status = f"below target {C.MIN_SAMPLES_TARGET}, above floor"
        else:
            status = "ok"
        sc = split_counts.loc[cls]
        print(f"  {cls:22s} {n:6d} {sc.get('train', 0):6d} {sc.get('val', 0):6d} {sc.get('test', 0):6d}  {status}")
    if failures:
        print("\nMINIMUM-SAMPLE CHECK FAILED")
        for cls, n in failures:
            print(f"  {cls}: {n} real samples found")
        print("Sources checked:")
        for root in roots:
            xbd_dirs = [str(p.parent.relative_to(root)) for p in sorted(root.glob("*/labels"))]
            print(f"  {root}: xBD folders {xbd_dirs or 'none'} (events {sorted(C.XBD_EVENTS)}); "
                  f"BRIGHT {'present' if (root / 'bright').exists() else 'not present'}")
        return False
    print(f"\nAll {len(C.COMBINED_CLASSES)} combined classes have >= {C.MIN_SAMPLES_FLOOR} real samples.")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-roots", type=Path, nargs="+", default=C.DATA_ROOTS)
    p.add_argument("--out", type=Path, default=C.METADATA_CSV)
    p.add_argument("--allow-missing", action="store_true",
                   help="write metadata.csv even if a class is below the floor (pipeline testing only)")
    args = p.parse_args()
    if not args.data_roots:
        sys.exit("No data roots found. Clone the data repos next to this one or set DISASTER_DATA_ROOTS.")
    print("Data roots: " + ", ".join(str(r) for r in args.data_roots))

    geotransforms = load_geotransforms(args.data_roots)
    rows = []
    for root in args.data_roots:
        rows += scan_xbd(root, geotransforms) + scan_bright(root)
    if not rows:
        sys.exit("No samples found under the data roots.")
    df = pd.DataFrame(rows).drop_duplicates("sample_id").reset_index(drop=True)  # first root wins
    print(f"Scanned {len(df)} samples: " + ", ".join(f"{k}={v}" for k, v in df.source.value_counts().items()))
    print("By event: " + ", ".join(f"{k}={v}" for k, v in df.event.value_counts().items()))

    all_counts = df.combined_class.value_counts()
    stratifiable = df[df.combined_class.map(all_counts) >= 5].copy()  # stratified splits need a few per class
    df = assign_splits(stratifiable.reset_index(drop=True))
    ok = enforce_minimums(df, all_counts, args.data_roots)
    if ok or args.allow_missing:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Wrote {args.out} ({len(df)} rows)" + ("" if ok else "  [INCOMPLETE: pipeline testing only]"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
