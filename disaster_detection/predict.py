"""End-to-end inference: classifier + U-Net -> probabilities, affected-area mask and statistics.

Command line, for any pre/post image pair (PNG/JPG/GeoTIFF):

    python -m disaster_detection.predict --pre pre.png --post post.png --out-dir outputs/predict/my_tile
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .area import area_statistics
from .dataset import to_tensor
from .models import MultitaskSiameseResNet18, SiameseResNetUNet

CLASSIFIER_CKPT = C.OUTPUT_ROOT / "classifier" / "best_model.pt"
UNET_CKPT = C.OUTPUT_ROOT / "segmentation" / "best_unet.pt"


class DisasterPredictor:
    def __init__(self, classifier_ckpt=CLASSIFIER_CKPT, unet_ckpt=UNET_CKPT):
        ckpt = torch.load(classifier_ckpt, map_location="cpu")
        self.classifier = MultitaskSiameseResNet18(pretrained=False, dropout=ckpt.get("dropout", 0.5))
        self.classifier.load_state_dict(ckpt["state_dict"])
        self.classifier.eval()
        seg_ckpt = torch.load(unet_ckpt, map_location="cpu")
        self.unet = SiameseResNetUNet(pretrained=False)
        self.unet.load_state_dict(seg_ckpt["state_dict"])
        self.unet.eval()

    @torch.no_grad()
    def predict(self, pre, post, pixel_area_m2=None):
        """pre/post: uint8 RGB arrays at IMAGE_SIZE. Returns a prediction dict."""
        pre_t, post_t = to_tensor(pre)[None], to_tensor(post)[None]
        type_logits, sev_logits = self.classifier(pre_t, post_t)
        type_probs = type_logits.softmax(1)[0].numpy()
        sev_probs = sev_logits.softmax(1)[0].numpy()
        seg = self.unet(pre_t, post_t).argmax(1)[0].numpy().astype(np.uint8)
        t, s = int(type_probs.argmax()), int(sev_probs.argmax())
        return {
            "disaster_type": C.DISASTER_TYPES[t], "disaster_confidence": float(type_probs[t]),
            "severity": C.SEVERITIES[s], "severity_confidence": float(sev_probs[s]),
            "combined_class": f"{C.DISASTER_TYPES[t]}_{C.SEVERITIES[s]}",
            "disaster_probabilities": {n: float(p) for n, p in zip(C.DISASTER_TYPES, type_probs)},
            "severity_probabilities": {n: float(p) for n, p in zip(C.SEVERITIES, sev_probs)},
            "segmentation": seg,
            **area_statistics(seg, pixel_area_m2),
        }


def load_pair(pre_path, post_path, size=C.IMAGE_SIZE):
    """Load an arbitrary pre/post pair as uint8 RGB at `size` (GeoTIFF or ordinary image files)."""
    import cv2
    from .preprocess import read_rgb, read_tif_rgb
    read = lambda p: read_tif_rgb(p) if str(p).lower().endswith((".tif", ".tiff")) else read_rgb(p)
    pre, post = read(pre_path), read(post_path)
    return (cv2.resize(pre, (size, size), interpolation=cv2.INTER_AREA),
            cv2.resize(post, (size, size), interpolation=cv2.INTER_AREA))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pre", required=True, type=Path)
    p.add_argument("--post", required=True, type=Path)
    p.add_argument("--out-dir", type=Path, default=C.OUTPUT_ROOT / "predict")
    args = p.parse_args()
    from .report import build_report, save_report
    from .visualize import plot_overlay

    pre, post = load_pair(args.pre, args.post)
    pred = DisasterPredictor().predict(pre, post)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    text = build_report(pred)
    print(text)
    save_report(pred, text, args.out_dir / "summary_report", extra={"pre": str(args.pre), "post": str(args.post)})
    plot_overlay(pre, post, pred["segmentation"], args.out_dir / "affected_regions.png",
                 title=f"{pred['combined_class']} - affected area {pred['affected_area_pct']:.2f}%")
    print(f"Saved report and overlay to {args.out_dir} (no coordinates given, so no Folium map)")


if __name__ == "__main__":
    main()
