"""Stage 5: affected-region maps - Matplotlib overlay and Folium interactive map."""
import base64
import html
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from . import config as C  # noqa: E402
from .geo import pixel_lonlat  # noqa: E402
from .plots import SURFACE, TEXT_PRIMARY, TEXT_SECONDARY  # noqa: E402

AFFECTED_COLOR = "#e34948"
INTACT_COLOR = "#2a78d6"
TRUTH_COLOR = "#eda100"


def _overlay(rgb, seg, alpha=0.45):
    out = rgb.astype(np.float32).copy()
    for cls, hex_color, a in ((C.SEG_INTACT, INTACT_COLOR, alpha * 0.6), (C.SEG_AFFECTED, AFFECTED_COLOR, alpha)):
        color = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)
        m = seg == cls
        out[m] = (1 - a) * out[m] + a * color
    return out.astype(np.uint8)


def _contours(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c[:, 0, :] for c in contours if len(c) >= 3]


def plot_overlay(pre, post, seg_pred, path, seg_true=None, title=""):
    """Pre image | post image | post with predicted mask and damage boundary [| ground truth]."""
    panels = [("Pre-event", pre), ("Post-event", post), ("Predicted affected regions", _overlay(post, seg_pred))]
    if seg_true is not None:
        panels.append(("Ground-truth labels", _overlay(post, seg_true)))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.8), facecolor=SURFACE)
    for ax, (name, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(name, color=TEXT_PRIMARY, fontsize=10, loc="left")
        ax.axis("off")
    for c in _contours(seg_pred == C.SEG_AFFECTED):
        axes[2].plot(*np.vstack([c, c[:1]]).T, color=AFFECTED_COLOR, linewidth=1.2)
    if seg_true is not None:
        for c in _contours(seg_true == C.SEG_AFFECTED):
            axes[3].plot(*np.vstack([c, c[:1]]).T, color=TRUTH_COLOR, linewidth=1.2)
    handles = [Patch(color=AFFECTED_COLOR, label="damaged building (affected)"),
               Patch(color=INTACT_COLOR, alpha=0.6, label="intact building")]
    if seg_true is not None:
        handles.append(Patch(color=TRUTH_COLOR, label="ground-truth damage boundary"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9,
               labelcolor=TEXT_SECONDARY)
    if title:
        fig.suptitle(title, color=TEXT_PRIMARY, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def _png_data_url(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def folium_map(row, post, seg_pred, path, report_text="", seg_true=None):
    """Interactive HTML map of the predicted affected regions at real coordinates.

    Returns False (and writes nothing) when the sample has no georeferencing.
    """
    import folium

    if not isinstance(row.get("geo_transform"), str) or not row["geo_transform"]:
        return False
    transform = json.loads(row["geo_transform"])
    epsg = int(row["geo_epsg"]) if str(row.get("geo_epsg", "")).strip() not in ("", "nan") else None
    scale_x = float(row["orig_width"]) / seg_pred.shape[1]
    scale_y = float(row["orig_height"]) / seg_pred.shape[0]

    def to_latlon(points):
        lon, lat = pixel_lonlat(transform, epsg, (points[:, 0] + 0.5) * scale_x, (points[:, 1] + 0.5) * scale_y)
        return np.column_stack([lat, lon]).tolist()

    h, w = seg_pred.shape
    corners = to_latlon(np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]]))
    lats, lons = [c[0] for c in corners], [c[1] for c in corners]
    fmap = folium.Map(location=[np.mean(lats), np.mean(lons)], zoom_start=16, control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite basemap (Esri)").add_to(fmap)
    folium.raster_layers.ImageOverlay(
        image=_png_data_url(post), bounds=[[min(lats), min(lons)], [max(lats), max(lons)]], opacity=0.85,
        name="Post-event image").add_to(fmap)

    footprint = folium.FeatureGroup(name="Tile footprint")
    folium.Polygon(corners, color="#52514e", weight=2, fill=False, tooltip=row["sample_id"]).add_to(footprint)
    footprint.add_to(fmap)

    predicted = folium.FeatureGroup(name="Predicted affected regions")
    for c in _contours(seg_pred == C.SEG_AFFECTED):
        folium.Polygon(to_latlon(c), color=AFFECTED_COLOR, weight=2, fill=True, fill_opacity=0.45,
                       tooltip="Predicted affected region").add_to(predicted)
    predicted.add_to(fmap)
    if seg_true is not None:
        truth = folium.FeatureGroup(name="Ground-truth damage", show=False)
        for c in _contours(seg_true == C.SEG_AFFECTED):
            folium.Polygon(to_latlon(c), color=TRUTH_COLOR, weight=2, fill=False, dash_array="5",
                           tooltip="Ground-truth damage").add_to(truth)
        truth.add_to(fmap)
    folium.Marker([np.mean(lats), np.mean(lons)],
                  popup=folium.Popup(f"<b>{html.escape(row['sample_id'])}</b><div style='font-family:monospace;"
                                     f"font-size:11px;white-space:pre-wrap'>{html.escape(report_text).replace(chr(10), '<br>')}</div>",
                                     max_width=640),
                  tooltip="Summary report").add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    fmap.save(str(path))
    return True
