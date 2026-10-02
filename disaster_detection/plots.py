"""Matplotlib figures: loss curves (overfitting diagnosis) and confusion matrices."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
TRAIN_COLOR = "#2a78d6"  # categorical slot 1
VAL_COLOR = "#eb6834"    # categorical slot 2
GAP_FILL = "#d9d8d3"
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_loss_curves(history, summary, path):
    """Train vs validation loss for each head and combined, with the gap shaded and phases marked."""
    epochs = history.epoch.to_numpy()
    phase_changes = history.epoch[history.unfrozen_level.diff().fillna(0) != 0].tolist()
    best = summary["best_epoch"]
    panels = [("loss_type", "Disaster-type head"), ("loss_severity", "Severity head"),
              ("loss_total", "Combined (type + severity)")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), facecolor=SURFACE)
    for ax, (key, title) in zip(axes.flat, panels):
        _style(ax)
        tr, va = history[f"train_{key}"].to_numpy(), history[f"val_{key}"].to_numpy()
        ax.fill_between(epochs, tr, va, color=GAP_FILL, alpha=0.7, linewidth=0, label="train/val gap")
        ax.plot(epochs, tr, color=TRAIN_COLOR, linewidth=2, marker="o", markersize=4, label="train")
        ax.plot(epochs, va, color=VAL_COLOR, linewidth=2, marker="o", markersize=4, label="validation")
        for e in phase_changes:
            ax.axvline(e, color=TEXT_SECONDARY, linestyle=":", linewidth=1)
        ax.axvline(best, color=TEXT_PRIMARY, linestyle="--", linewidth=1)
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left")
        ax.set_xlabel("epoch", color=TEXT_SECONDARY, fontsize=9)
        ax.set_ylabel("cross-entropy loss", color=TEXT_SECONDARY, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY)

    ax = axes.flat[3]
    _style(ax)
    gap = history.val_loss_total - history.train_loss_total
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=1)
    ax.plot(epochs, gap, color=VAL_COLOR, linewidth=2, marker="o", markersize=4)
    for e in phase_changes:
        ax.axvline(e, color=TEXT_SECONDARY, linestyle=":", linewidth=1)
    ax.axvline(best, color=TEXT_PRIMARY, linestyle="--", linewidth=1)
    ax.set_title("Gap: validation minus train combined loss", color=TEXT_PRIMARY, fontsize=11, loc="left")
    ax.set_xlabel("epoch", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("loss gap", color=TEXT_SECONDARY, fontsize=9)

    g = summary["clean_eval_at_best_weights"]
    note = (f"Best epoch {best} (dashed). Dotted lines: encoder layers unfrozen. "
            f"Clean-eval gap at best weights: total {g['gap_total']:+.4f}, type {g['gap_type']:+.4f}, "
            f"severity {g['gap_severity']:+.4f}.\n{summary['stop_reason']}")
    fig.suptitle("Classifier training vs validation loss", color=TEXT_PRIMARY, fontsize=13, x=0.01, ha="left")
    fig.text(0.01, 0.005, note, color=TEXT_SECONDARY, fontsize=8.5, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion_matrix(cm, labels, title, path):
    """Confusion-matrix heatmap with counts and row-normalised recall in each cell."""
    cm = np.asarray(cm)
    row_sums = cm.sum(1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    size = max(5.0, 0.85 * len(labels) + 2.5)
    fig, ax = plt.subplots(figsize=(size + 1.2, size), facecolor=SURFACE)
    im = ax.imshow(norm, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "#ffffff" if norm[i, j] > 0.55 else TEXT_PRIMARY
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}" if cm[i, j] else "0", ha="center", va="center",
                    color=color, fontsize=8 if len(labels) > 6 else 10)
    ax.set_xticks(range(len(labels)), labels, rotation=45 if len(labels) > 4 else 0,
                  ha="right" if len(labels) > 4 else "center", fontsize=9, color=TEXT_SECONDARY)
    ax.set_yticks(range(len(labels)), labels, fontsize=9, color=TEXT_SECONDARY)
    ax.set_xlabel("predicted", color=TEXT_SECONDARY)
    ax.set_ylabel("true", color=TEXT_SECONDARY)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=12, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("share of true class (recall)", color=TEXT_SECONDARY)
    cbar.ax.tick_params(colors=TEXT_SECONDARY)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_simple_loss_curve(history, summary, title, path):
    """Single-loss train vs validation curve with the gap shaded (used for the U-Net)."""
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    _style(ax)
    e = history.epoch.to_numpy()
    tr, va = history.train_loss.to_numpy(), history.val_loss.to_numpy()
    ax.fill_between(e, tr, va, color=GAP_FILL, alpha=0.7, linewidth=0, label="train/val gap")
    ax.plot(e, tr, color=TRAIN_COLOR, linewidth=2, marker="o", markersize=4, label="train")
    ax.plot(e, va, color=VAL_COLOR, linewidth=2, marker="o", markersize=4, label="validation")
    ax.axvline(summary["best_epoch"], color=TEXT_PRIMARY, linestyle="--", linewidth=1)
    ax.set_xlabel("epoch", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("weighted CE + Dice loss", color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY)
    fig.text(0.01, 0.01, f"Best epoch {summary['best_epoch']} (dashed). Clean-eval gap {summary['gap_loss']:+.4f}. "
             f"{summary['stop_reason']}", color=TEXT_SECONDARY, fontsize=8, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
