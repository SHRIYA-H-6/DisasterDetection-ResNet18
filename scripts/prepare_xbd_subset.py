"""Extract only the xBD events this project needs from the official challenge archive.

Run locally after downloading the "Challenge training set" from xview2.org:

    python prepare_xbd_subset.py path/to/train_images_labels_targets.tar.gz path/to/output_folder

It verifies the SHA1 published on xview2.org, then streams the archive and copies
only the images, labels and targets of the selected events (no full 7.8 GB unpack).
"""
import argparse
import hashlib
import os
import sys
import tarfile
from collections import Counter

TRAIN_SHA1 = "b37a4ef4ee9c909e2b19d046e49d42ee3965714b"
DEFAULT_EVENTS = ["hurricane-michael"]  # mexico-earthquake and palu-tsunami are already in xbd-data


def sha1_of(path, chunk=8 * 1024 * 1024):
    h = hashlib.sha1()
    done, total = 0, os.path.getsize(path)
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
            done += len(block)
            print(f"\r  hashing {done / total:6.1%}", end="", flush=True)
    print()
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("archive", help="downloaded xBD challenge training archive (.tar.gz or .tar)")
    p.add_argument("out_dir", help="folder to write train/images, train/labels, train/targets into")
    p.add_argument("--events", nargs="+", default=DEFAULT_EVENTS, help="xBD event names to keep")
    p.add_argument("--skip-hash", action="store_true", help="skip SHA1 verification")
    args = p.parse_args()

    if not args.skip_hash:
        print("Verifying SHA1 (takes a few minutes for 7.8 GB)...")
        digest = sha1_of(args.archive)
        if digest != TRAIN_SHA1:
            sys.exit(f"SHA1 mismatch: got {digest}, expected {TRAIN_SHA1}. Re-download the archive.")
        print("  SHA1 OK")

    prefixes = tuple(f"{e}_" for e in args.events)
    counts = Counter()
    print(f"Extracting events: {', '.join(args.events)}")
    with tarfile.open(args.archive, "r|*")  # .tar or .tar.gz as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = os.path.basename(member.name)
            folder = os.path.basename(os.path.dirname(member.name))  # images / labels / targets
            if not name.startswith(prefixes) or folder not in ("images", "labels", "targets"):
                continue
            dest = os.path.join(args.out_dir, "train", folder, name)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with tar.extractfile(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            counts[(name.split("_")[0], folder)] += 1
            print(f"\r  extracted {sum(counts.values())} files", end="", flush=True)
    print()

    if not counts:
        sys.exit("No matching files found - check the archive and event names.")
    print("\nFiles per event:")
    for event in args.events:
        row = {f: counts[(event, f)] for f in ("images", "labels", "targets")}
        print(f"  {event:20s} images={row['images']:5d} labels={row['labels']:5d} targets={row['targets']:5d}")
    print(f"\nDone. Output in: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
