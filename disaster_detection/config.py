"""Shared constants: class names, severity thresholds, paths and image sizes."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _data_roots():
    """Folders holding real data, searched in order. Override with DISASTER_DATA_ROOTS (os.pathsep-separated).

    Default: every existing sibling folder named xbd-data* (e.g. ../xbd-data with the original xBD subset and
    ../xbd-data-2 with the hurricane + BRIGHT additions).
    """
    env = os.environ.get("DISASTER_DATA_ROOTS")
    if env:
        return [Path(p) for p in env.split(os.pathsep) if p]
    return sorted(p for p in PROJECT_ROOT.parent.glob("xbd-data*") if p.is_dir())


DATA_ROOTS = _data_roots()
OUTPUT_ROOT = Path(os.environ.get("DISASTER_OUTPUT_ROOT", PROJECT_ROOT / "outputs"))
CACHE_DIR = OUTPUT_ROOT / "cache"
METADATA_CSV = PROJECT_ROOT / "data" / "metadata.csv"
GEOTRANSFORMS_NAME = "xview_geotransforms.json.tgz"

DISASTER_TYPES = ["Earthquake", "Tsunami", "Hurricane"]
SEVERITIES = ["Low", "Moderate", "Severe"]
COMBINED_CLASSES = [f"{d}_{s}" for d in DISASTER_TYPES for s in SEVERITIES]

# xBD event -> project disaster type. Events missing from the data roots are simply not found.
XBD_EVENTS = {
    "mexico-earthquake": "Earthquake",
    "palu-tsunami": "Tsunami",
    "sunda-tsunami": "Tsunami",
    "hurricane-harvey": "Hurricane",
    "hurricane-florence": "Hurricane",
    "hurricane-michael": "Hurricane",
    "hurricane-matthew": "Hurricane",
}
# BRIGHT (Chen et al., 2025) events containing this string are used as Earthquake samples:
# xBD's only earthquake (mexico-earthquake) has no moderately/severely damaged tiles.
BRIGHT_EVENT_FILTER = "earthquake"

# Severity from the fraction of damaged buildings among classified buildings in an
# xBD tile (minor/major/destroyed) or of damaged building pixels in a BRIGHT tile
# (damaged/destroyed): Low <= 10% < Moderate <= 30% < Severe.
SEVERITY_EDGES = (0.10, 0.30)

MIN_SAMPLES_FLOOR = 10
MIN_SAMPLES_TARGET = 15

# xBD target PNG pixel codes.
XBD_BACKGROUND, XBD_NO_DAMAGE, XBD_MINOR, XBD_MAJOR, XBD_DESTROYED, XBD_UNCLASSIFIED = range(6)

# Segmentation classes predicted by the U-Net.
SEG_BACKGROUND, SEG_INTACT, SEG_AFFECTED = 0, 1, 2
SEG_CLASS_NAMES = ["background", "intact building", "damaged building"]

IMAGE_SIZE = 256  # tiles are resized to this for both models
SEED = 42

# ImageNet-pretrained ResNet-18 (timm "resnet18.a1_in1k", torchvision-compatible keys), from the timm GitHub
# release. The 8 hex digits in the file name are the start of its SHA-256 and are verified on download.
RESNET18_WEIGHTS_URL = ("https://github.com/huggingface/pytorch-image-models/releases/download/"
                        "v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth")


def severity_from_fraction(fraction, edges=SEVERITY_EDGES):
    low, high = edges
    if fraction <= low:
        return "Low"
    if fraction <= high:
        return "Moderate"
    return "Severe"


def find_root(name):
    """Data root folder by its directory name (as stored in metadata.csv's data_root column)."""
    for root in DATA_ROOTS:
        if root.name == name:
            return root
    raise FileNotFoundError(f"Data root '{name}' not found among {[str(r) for r in DATA_ROOTS]}. "
                            "Clone it next to this repo or set DISASTER_DATA_ROOTS.")


def data_path(row, key):
    """Absolute path of a file referenced by a metadata row (key: pre_path / post_path / target_path)."""
    return find_root(row["data_root"]) / row[key]
