"""Load real xBD / BRIGHT samples and cache them at IMAGE_SIZE as .npz.

Each cache entry holds:
  pre, post : uint8 RGB (H, W, 3). BRIGHT post-event images are single-band SAR,
              stretched and repeated to 3 channels.
  seg       : uint8 (H, W) with SEG_BACKGROUND / SEG_INTACT / SEG_AFFECTED.

    python -m disaster_detection.preprocess
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import config as C

# xBD target code -> segmentation class
XBD_TO_SEG = np.array([
    C.SEG_BACKGROUND,  # background
    C.SEG_INTACT,      # no-damage
    C.SEG_AFFECTED,    # minor-damage
    C.SEG_AFFECTED,    # major-damage
    C.SEG_AFFECTED,    # destroyed
    C.SEG_INTACT,      # un-classified building: present, damage unknown
], dtype=np.uint8)


def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def stretch(arr):
    """Percentile (2-98%) stretch to uint8."""
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    return (np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def read_tif_rgb(path):
    """Read a BRIGHT GeoTIFF as uint8 RGB (optical: first 3 bands; SAR: band repeated)."""
    import tifffile
    arr = tifffile.imread(path)
    if arr.ndim == 3 and arr.shape[0] < arr.shape[-1] and arr.shape[0] <= 4:
        arr = arr.transpose(1, 2, 0)  # bands-first -> bands-last
    if arr.ndim == 2:
        arr = arr[..., None]
    arr = arr[..., :3] if arr.shape[-1] >= 3 else np.repeat(arr[..., :1], 3, axis=-1)
    return arr if arr.dtype == np.uint8 else stretch(arr)


def resize_labels(labels, n_classes, size):
    """Downsample a label map by area-averaging one-hot planes and taking the argmax."""
    one_hot = np.stack([(labels == k).astype(np.float32) for k in range(n_classes)], axis=-1)
    small = cv2.resize(one_hot, (size, size), interpolation=cv2.INTER_AREA)
    return small.argmax(-1).astype(np.uint8)


def load_sample(row, size=C.IMAGE_SIZE):
    """Return (pre, post, seg) arrays at `size`, read from the original files of a metadata row."""
    if row["source"] == "xBD":
        pre = read_rgb(C.data_path(row, "pre_path"))
        post = read_rgb(C.data_path(row, "post_path"))
        target = cv2.imread(str(C.data_path(row, "target_path")), cv2.IMREAD_UNCHANGED)
        seg = XBD_TO_SEG[np.clip(target, 0, 5)]
    elif row["source"] == "BRIGHT":
        import tifffile
        pre = read_tif_rgb(C.data_path(row, "pre_path"))
        post = read_tif_rgb(C.data_path(row, "post_path"))
        target = tifffile.imread(C.data_path(row, "target_path"))
        seg = np.select([target >= 2, target == 1], [C.SEG_AFFECTED, C.SEG_INTACT], C.SEG_BACKGROUND).astype(np.uint8)
        if post.shape[:2] != pre.shape[:2]:
            post = cv2.resize(post, pre.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError(f"unknown source {row['source']}")
    pre = cv2.resize(pre, (size, size), interpolation=cv2.INTER_AREA)
    post = cv2.resize(post, (size, size), interpolation=cv2.INTER_AREA)
    seg = resize_labels(seg, 3, size)
    return pre, post, seg


def cache_path(sample_id, cache_dir=C.CACHE_DIR):
    return Path(cache_dir) / f"{sample_id}.npz"


def build_cache(metadata, cache_dir=C.CACHE_DIR, overwrite=False):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    affected = []
    for i, row in enumerate(metadata.to_dict("records"), 1):
        path = cache_path(row["sample_id"], cache_dir)
        if overwrite or not path.exists():
            pre, post, seg = load_sample(row)
            np.savez_compressed(path, pre=pre, post=post, seg=seg)
        else:
            seg = np.load(path)["seg"]
        affected.append(float((seg == C.SEG_AFFECTED).mean()))
        print(f"\r  cached {i}/{len(metadata)}", end="", flush=True)
    print()
    return affected


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", type=Path, default=C.METADATA_CSV)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    df = pd.read_csv(args.metadata)
    df["gt_affected_fraction"] = build_cache(df, overwrite=args.overwrite)
    df.to_csv(args.metadata, index=False)
    print("Ground-truth affected-area fraction (from labels) by class:")
    print(df.groupby("combined_class").gt_affected_fraction.describe()[["count", "mean", "max"]].round(4).to_string())


if __name__ == "__main__":
    main()
