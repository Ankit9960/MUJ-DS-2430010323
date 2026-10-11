"""Figure generation for the report.

Design rules applied throughout: one measure per axis, recessive grid and axes,
a legend whenever two or more series are present, direct value labels on bars
(which also supplies the relief required where a fill sits below 3:1 against the
surface), and a single-hue sequential ramp for the confusion matrices.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src import config as C

# --- palette -------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]      # blue, orange, aqua
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
CRITICAL = "#d03b3b"

# Single-hue sequential ramp (blue 100 -> 700) for magnitude encoding.
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": BASELINE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "figure.dpi": 140,
})


def _clean(ax, ygrid_only: bool = True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    if ygrid_only:
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)


def _save(fig, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
def plot_confusion_matrix(cm, title: str, path: Path, normalise: bool = True):
    """Row-normalised confusion matrix. Rows are true classes, so the diagonal
    reads as per-class sensitivity -- the number the AAMI standard cares about."""
    cm = np.asarray(cm, dtype=float)
    counts = cm.copy()
    if normalise:
        row = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0) * 100

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cm, cmap=SEQ_BLUE, vmin=0, vmax=100 if normalise else cm.max())
    ax.grid(False)

    labels = C.AAMI_CLASSES
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, color=INK, fontsize=11, pad=12, loc="left")

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            # Ink flips to white on the dark end of the ramp so labels stay legible.
            colour = "#ffffff" if val > (55 if normalise else cm.max() * 0.55) else INK
            txt = f"{val:.1f}%" if normalise else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center", color=colour, fontsize=9)
            if normalise:
                ax.text(j, i + 0.30, f"n={int(counts[i, j])}", ha="center", va="center",
                        color=colour, fontsize=6.5, alpha=0.75)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% of true class" if normalise else "beats", color=INK_SECONDARY)
    cbar.outline.set_visible(False)
    return _save(fig, path)


def plot_leakage_comparison(intra: dict, inter: dict, path: Path,
                            title: str = "Effect of the evaluation protocol"):
    """The headline CO5 figure: identical model, two split protocols."""
    metrics = [("accuracy_mean", "Overall\naccuracy"),
               ("macro_f1_mean", "Macro F1"),
               ("S_sensitivity_mean", "S (SVEB)\nsensitivity"),
               ("V_sensitivity_mean", "V (VEB)\nsensitivity")]
    keys = [k for k, _ in metrics]
    labels = [l for _, l in metrics]

    intra_v = [intra.get(k, np.nan) * 100 for k in keys]
    inter_v = [inter.get(k, np.nan) * 100 for k in keys]
    intra_e = [intra.get(k.replace("_mean", "_std"), 0) * 100 for k in keys]
    inter_e = [inter.get(k.replace("_mean", "_std"), 0) * 100 for k in keys]

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    b1 = ax.bar(x - w / 2 - 0.01, intra_v, w, yerr=intra_e, capsize=3,
                label="Intra-patient (random beat split)", color=SERIES[1],
                edgecolor=SURFACE, linewidth=2, error_kw=dict(ecolor=MUTED, lw=1))
    b2 = ax.bar(x + w / 2 + 0.01, inter_v, w, yerr=inter_e, capsize=3,
                label="Inter-patient (DS1/DS2)", color=SERIES[0],
                edgecolor=SURFACE, linewidth=2, error_kw=dict(ecolor=MUTED, lw=1))

    for bars, vals in ((b1, intra_v), (b2, inter_v)):
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 1.8, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=8.5, color=INK_SECONDARY)

    # Drop label sits above the group, clear of both bars and the tick labels.
    for xi, (a, b) in enumerate(zip(intra_v, inter_v)):
        if not (np.isnan(a) or np.isnan(b)):
            ax.text(xi, max(a, b) + 9.5, f"−{a - b:.1f} pts", ha="center",
                    fontsize=9, color=CRITICAL, fontweight="bold")

    ax.set_xticks(x, labels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 125)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=34)
    # Legend above the plot area so it can never collide with a tall bar.
    ax.legend(frameon=False, fontsize=9, ncol=2,
              loc="upper left", bbox_to_anchor=(0.0, 1.11), handlelength=1.4)
    _clean(ax)
    fig.text(0.01, -0.02,
             "Same architecture, same hyper-parameters, same beats — only the "
             "train/test partition differs.",
             fontsize=8.5, color=MUTED)
    return _save(fig, path)


def plot_ablation(rows: list[dict], path: Path, metric: str = "macro_f1_mean",
                  metric_label: str = "Macro F1", title: str = "Ablation study"):
    """Horizontal bars, one per ablation variant, sorted by score."""
    rows = sorted(rows, key=lambda r: r.get(metric, 0))
    names = [r["label"] for r in rows]
    vals = [r.get(metric, np.nan) * 100 for r in rows]
    errs = [r.get(metric.replace("_mean", "_std"), 0) * 100 for r in rows]
    is_base = [bool(r.get("is_base")) for r in rows]

    colours = [SERIES[0] if b else SERIES[2] for b in is_base]

    fig, ax = plt.subplots(figsize=(9.4, 0.56 * len(rows) + 2.2))
    bars = ax.barh(np.arange(len(rows)), vals, xerr=errs, capsize=3,
                   color=colours, edgecolor=SURFACE, linewidth=2,
                   error_kw=dict(ecolor=MUTED, lw=1))
    # Label clears the error-bar cap, not just the bar end.
    for bar, v, e in zip(bars, vals, errs):
        ax.text(bar.get_width() + e + 1.8, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}", va="center", fontsize=9.5, color=INK_SECONDARY)

    ax.set_yticks(np.arange(len(rows)), names, fontsize=9)
    ax.set_xlabel(f"{metric_label} (%)")
    hi = max([v + e for v, e in zip(vals, errs) if not np.isnan(v)] + [1])
    ax.set_xlim(0, hi * 1.22)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=32)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[0]),
               plt.Rectangle((0, 0), 1, 1, color=SERIES[2])]
    # Above the plot area: horizontal bars leave no safe interior corner.
    ax.legend(handles, ["Proposed configuration", "Ablated variant"],
              frameon=False, fontsize=9, ncol=2, handlelength=1.4,
              loc="upper left", bbox_to_anchor=(0.0, 1.0 + 1.6 / len(rows)))
    return _save(fig, path)


def plot_per_class_f1(results: dict[str, dict], path: Path,
                      title: str = "Per-class F1 by configuration"):
    """Grouped bars over the five AAMI classes."""
    names = list(results.keys())
    x = np.arange(len(C.AAMI_CLASSES))
    w = 0.8 / max(len(names), 1)

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for i, name in enumerate(names):
        per = results[name]["per_class"]
        vals = [per[c]["f1_mean"] * 100 for c in C.AAMI_CLASSES]
        off = (i - (len(names) - 1) / 2) * w
        bars = ax.bar(x + off, vals, w * 0.92, label=name,
                      color=SERIES[i % len(SERIES)], edgecolor=SURFACE, linewidth=2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.4, f"{v:.0f}",
                    ha="center", fontsize=7.5, color=INK_SECONDARY)

    ax.set_xticks(x, [f"{c}\n({_class_note(c)})" for c in C.AAMI_CLASSES])
    ax.set_ylabel("F1 (%)")
    ax.set_ylim(0, 108)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9)
    _clean(ax)
    return _save(fig, path)


def _class_note(c: str) -> str:
    return {"N": "normal", "S": "supravent.", "V": "ventric.",
            "F": "fusion", "Q": "unknown"}[c]


def plot_training_curves(history: list[dict], path: Path, title: str = "Training"):
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ep = [h["epoch"] for h in history]
    ax.plot(ep, [h["val_macro_f1"] * 100 for h in history], color=SERIES[0],
            linewidth=2, label="Validation macro F1")
    ax.plot(ep, [h["val_accuracy"] * 100 for h in history], color=SERIES[2],
            linewidth=2, linestyle="--", label="Validation accuracy")
    best = int(np.argmax([h["val_macro_f1"] for h in history]))
    ax.scatter([ep[best]], [history[best]["val_macro_f1"] * 100], s=48,
               color=SERIES[0], zorder=5, edgecolor=SURFACE, linewidth=2)
    ax.annotate(f"best epoch {ep[best]}",
                (ep[best], history[best]["val_macro_f1"] * 100),
                textcoords="offset points", xytext=(8, -14),
                fontsize=8.5, color=INK_SECONDARY)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score (%)")
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9)
    _clean(ax)
    return _save(fig, path)


def plot_class_distribution(counts: dict, path: Path,
                            title: str = "Class distribution (MIT-BIH, AAMI classes)"):
    """Log-scale bar chart -- the imbalance is the point, and it spans ~3 decades."""
    classes = list(counts.keys())
    vals = [counts[c] for c in classes]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bars = ax.bar(classes, vals, color=SERIES[0], edgecolor=SURFACE, linewidth=2, width=0.62)
    total = sum(vals) or 1
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.12,
                f"{v:,}\n({100 * v / total:.1f}%)", ha="center",
                fontsize=8.5, color=INK_SECONDARY)
    ax.set_yscale("log")
    ax.set_ylabel("Beats (log scale)")
    ax.set_ylim(1, max(vals) * 6 if vals else 10)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    _clean(ax)
    return _save(fig, path)
