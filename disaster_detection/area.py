"""Stage 4: affected-area measurement from U-Net segmentation output (not a heuristic)."""
import numpy as np

from . import config as C


def affected_area_percent(seg):
    """Percentage of the image's pixels the segmentation labels as affected."""
    seg = np.asarray(seg)
    return 100.0 * float((seg == C.SEG_AFFECTED).mean())


def area_statistics(seg, pixel_area_m2=None):
    """Affected-area figures for one predicted mask.

    affected_area_pct      : affected pixels / all pixels (reported as "Affected Area")
    affected_structure_pct : affected pixels / (intact + affected) pixels, i.e. the damaged share
                             of the structures/terrain the model detected
    affected_area_m2       : affected ground area when the pixel footprint is known
    """
    seg = np.asarray(seg)
    affected = int((seg == C.SEG_AFFECTED).sum())
    intact = int((seg == C.SEG_INTACT).sum())
    stats = {
        "affected_area_pct": 100.0 * affected / seg.size,
        "affected_structure_pct": 100.0 * affected / (affected + intact) if affected + intact else 0.0,
        "affected_pixels": affected, "total_pixels": int(seg.size),
    }
    if pixel_area_m2:
        stats["affected_area_m2"] = affected * pixel_area_m2
    return stats
