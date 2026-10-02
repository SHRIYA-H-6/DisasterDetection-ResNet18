"""Pick a damage-stratified subset of real BRIGHT earthquake tiles.

xBD's only earthquake event (mexico-earthquake) has no moderately or severely
damaged tiles, so Earthquake Moderate/Severe samples come from BRIGHT
(Chen et al., 2025): pre-event optical + post-event SAR tiles with expert
building-damage labels (0 background, 1 intact, 2 damaged, 3 destroyed).

Run locally after downloading BRIGHT from Zenodo (record 14619797):

    python prepare_bright_subset.py path/to/xbd-data path/to/download1.zip [path/to/download2.zip ...]

Sources can be .zip files or extracted folders containing pre-event/, post-event/
and target/. Only tiles whose event name contains --event-filter are used.
"""
import argparse
import io
import os
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict

import numpy as np
import tifffile

SEVERITY_EDGES = (0.10, 0.30)  # damaged share of building pixels: Low <= 10% < Moderate <= 30% < Severe
NAME = re.compile(r"(?:^|/)(pre-event|post-event|target)/([^/]+?)_(\d+)_(pre_disaster|post_disaster|building_damage)\.tif$")


def list_sources(paths):
    """Map (event, tile_id) -> {folder: opener} across zip files and folders."""
    tiles = defaultdict(dict)
    for path in paths:
        if zipfile.is_zipfile(path):
            zf = zipfile.ZipFile(path)
            for name in zf.namelist():
                m = NAME.search(name.replace("\\", "/"))
                if m:
                    tiles[(m.group(2), m.group(3))][m.group(1)] = (os.path.basename(name), lambda z=zf, n=name: io.BytesIO(z.read(n)))
        else:
            for root, _, files in os.walk(path):
                for fname in files:
                    full = os.path.join(root, fname)
                    m = NAME.search(full.replace("\\", "/"))
                    if m:
                        tiles[(m.group(2), m.group(3))][m.group(1)] = (fname, lambda f=full: open(f, "rb"))
    return tiles


def severity(fraction):
    low, high = SEVERITY_EDGES
    return "Low" if fraction <= low else "Moderate" if fraction <= high else "Severe"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir", help="data repo folder; files go to <out_dir>/bright/{pre-event,post-event,target}")
    p.add_argument("sources", nargs="+", help="BRIGHT .zip files and/or extracted folders")
    p.add_argument("--event-filter", default="earthquake", help="keep events whose name contains this")
    p.add_argument("--per-severity", type=int, default=60, help="max tiles per severity level per event")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tiles = list_sources(args.sources)
    all_events = Counter(event for event, _ in tiles)
    print("Events found:", ", ".join(f"{e}={n}" for e, n in sorted(all_events.items())) or "none")
    complete = {k: v for k, v in tiles.items()
                if args.event_filter in k[0] and {"pre-event", "post-event", "target"} <= v.keys()}
    if not complete:
        sys.exit(f"No complete pre-event/post-event/target triplets for events containing '{args.event_filter}'.")
    print(f"{len(complete)} complete '{args.event_filter}' tiles. Measuring damage...")

    groups = defaultdict(list)
    values = Counter()
    for i, (key, files) in enumerate(sorted(complete.items()), 1):
        with files["target"][1]() as fh:
            target = tifffile.imread(fh)
        uniq, cnt = np.unique(target, return_counts=True)
        values.update(dict(zip(uniq.tolist(), cnt.tolist())))
        buildings = int((target >= 1).sum())
        if buildings:
            fraction = float((target >= 2).sum()) / buildings
            groups[(key[0], severity(fraction))].append(key)
        print(f"\r  {i}/{len(complete)}", end="", flush=True)
    print(f"\nLabel pixel values found (expect 0-3): {dict(sorted(values.items()))}")

    rng = random.Random(args.seed)
    print(f"\n{'event':28s} {'severity':10s} {'available':>9s} {'copied':>7s}")
    for (event, sev) in sorted(groups):
        chosen = rng.sample(groups[(event, sev)], min(args.per_severity, len(groups[(event, sev)])))
        for key in chosen:
            for folder in ("pre-event", "post-event", "target"):
                fname, opener = complete[key][folder]
                dest = os.path.join(args.out_dir, "bright", folder, fname)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with opener() as src, open(dest, "wb") as dst:
                    dst.write(src.read())
        print(f"{event:28s} {sev:10s} {len(groups[(event, sev)]):9d} {len(chosen):7d}")
    print(f"\nDone. Output in: {os.path.abspath(os.path.join(args.out_dir, 'bright'))}")


if __name__ == "__main__":
    main()
