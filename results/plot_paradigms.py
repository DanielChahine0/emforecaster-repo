#!/usr/bin/env python3
"""Schematic of the four forecasting paradigms, one panel per grid.

Panel <-> grid mapping (forecasting/grids.py, main.py --setup):

  (a) univariate           the 'univariate_series' grid: DET/PTV single-series
                           runs - one series, one model, one forecast.
  (b) channel-independent  the 'univariate' per-band grid: one model trained
                           per frequency band, no cross-channel information.
  (c) channel-dependent    the 'multivariate' grid: one model per cell, all
                           bands jointly. Drawn with the standard cross-channel
                           mixing motif; note DLinear/PatchTST keep per-channel
                           weights inside the shared model, so the mixing
                           cartoon is the paradigm, not every architecture.
  (d) channel-clustering   the 'cluster' grid (CCM, NeurIPS 2024). Drawn SOFT
                           on purpose, matching models/ccm.py: every channel
                           gets a sinkhorn-normalized membership over all k
                           learned cluster embeddings, all channels share one
                           model and one forward pass, and k cluster-wise heads
                           are blended per channel by membership. There is no
                           hard partition, no unclustered leftover channel, and
                           no separate per-cluster pass.

Visual vocabulary: solid squiggle = input window, dashed = forecast; one
accent (Verdigris, the lead plot_metrics colour) marks cluster membership,
filled vs hollow carrying the two clusters - the same second-cycle encoding
plot_metrics.series_style uses. In (b)/(c)/(d) lanes 1&3 share slow dynamics
and lanes 2&4 fast ones, so (d)'s similarity-based grouping is visibly not
adjacency in band order.

Writes results/figures/paradigms/paradigms.{pdf,svg,png} at IEEEtran
textwidth (7.16in); embed the PDF with \\figure*.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

import plot_metrics as pm

OUTDIR = Path(__file__).resolve().parent / "figures" / "paradigms"

ACCENT = pm.SERIES_COLORS[0]  # Verdigris
BOX_FC = "#f2f5f7"

# Per-lane squiggle recipes: (f1, f2, phase, a2). Lanes 0/2 slow, 1/3 fast.
LANE_WAVES = [
    (1.05, 2.10, 0.15, 0.45),
    (2.60, 5.20, 0.60, 0.55),
    (1.25, 2.40, 0.80, 0.40),
    (2.90, 4.60, 0.30, 0.50),
]
SINGLE_WAVE = (1.60, 3.10, 0.45, 0.50)

LANE_Y = [12.2, 10.0, 7.8, 5.6]
AMP = 0.75
CLUSTER_OF = [0, 1, 0, 1]  # lane -> cluster (0 = filled, 1 = hollow)

TITLE_Y, CAPTION_Y = 2.75, 2.0
TITLES = [
    "(a) Univariate",
    "(b) Channel-independent",
    "(c) Channel-dependent",
    "(d) Channel-clustering (CCM)",
]
CAPTIONS = [
    "One series, one model,\none forecast.",
    "One model per channel;\nno cross-channel information.",
    "One shared model for all\nchannels; they mix inside it.",
    "Soft assignment to k learned\nclusters; k cluster-wise heads.",
]


def wave(recipe, t):
    f1, f2, phase, a2 = recipe
    y = np.sin(2 * np.pi * (f1 * t + phase)) + a2 * np.sin(2 * np.pi * (f2 * t + 0.7 * phase))
    return y / (1 + a2)


def draw_series(ax, recipe, x0, x1, yc, dashed=False, color=pm.INK_PRIMARY):
    """History on t in [0,1), forecast continues the same waveform past 1."""
    t = np.linspace(1.0, 1.42, 90) if dashed else np.linspace(0.0, 1.0, 160)
    x = x0 + (x1 - x0) * (t - t[0]) / (t[-1] - t[0])
    style = dict(ls=(0, (2.4, 1.5)), lw=0.95) if dashed else dict(ls="-", lw=0.95)
    ax.plot(x, yc + AMP * wave(recipe, t), color=color, solid_capstyle="round", **style)


def draw_arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=pm.INK_SECONDARY,
                                lw=0.75, mutation_scale=6, shrinkA=0, shrinkB=0))


def draw_box(ax, x0, y0, w, h):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, zorder=1,
                                boxstyle="round,pad=0,rounding_size=0.3",
                                fc=BOX_FC, ec=pm.INK_PRIMARY, lw=0.7))


def cluster_marker(ax, x, y, cluster, ms):
    """Filled vs hollow Verdigris dot - plot_metrics' own second-cycle idiom."""
    filled = cluster == 0
    ax.plot([x], [y], "o", ms=ms, zorder=4,
            mfc=ACCENT if filled else pm.SURFACE,
            mec=pm.SURFACE if filled else ACCENT,
            mew=0.9 if filled else 1.1)


def caption(ax, i):
    ax.text(5, TITLE_Y, TITLES[i], ha="center", va="top", fontsize=8,
            fontweight="bold", color=pm.INK_PRIMARY)
    ax.text(5, CAPTION_Y, CAPTIONS[i], ha="center", va="top", fontsize=6.5,
            color=pm.INK_SECONDARY, linespacing=1.3)


def panel_a(ax):
    yc = 8.9
    draw_series(ax, SINGLE_WAVE, 0.3, 2.7, yc)
    draw_arrow(ax, 2.85, 3.7, yc)
    draw_box(ax, 3.8, 7.6, 2.5, 2.6)
    ax.text(5.05, 8.9, "$f$", ha="center", va="center", fontsize=8.5,
            color=pm.INK_PRIMARY)
    draw_arrow(ax, 6.4, 7.25, yc)
    draw_series(ax, SINGLE_WAVE, 7.4, 9.7, yc, dashed=True)
    caption(ax, 0)


def panel_b(ax):
    for i, yc in enumerate(LANE_Y):
        draw_series(ax, LANE_WAVES[i], 0.15, 2.35, yc)
        draw_arrow(ax, 2.5, 3.5, yc)
        draw_box(ax, 3.6, yc - 0.85, 2.8, 1.7)
        ax.text(5.0, yc, f"$f_{i + 1}$", ha="center", va="center", fontsize=7.5,
                color=pm.INK_PRIMARY)
        draw_arrow(ax, 6.5, 7.4, yc)
        draw_series(ax, LANE_WAVES[i], 7.55, 9.85, yc, dashed=True)
    caption(ax, 1)


def panel_c(ax):
    draw_box(ax, 3.6, 4.7, 2.8, 8.6)
    for i, yc in enumerate(LANE_Y):
        draw_series(ax, LANE_WAVES[i], 0.15, 2.35, yc)
        draw_arrow(ax, 2.5, 3.5, yc)
        draw_arrow(ax, 6.5, 7.4, yc)
        draw_series(ax, LANE_WAVES[i], 7.55, 9.85, yc, dashed=True)
    # All-to-all mixing motif: the cross-channel interaction the joint model adds.
    for yi in LANE_Y:
        for yo in LANE_Y:
            ax.plot([4.1, 5.9], [yi, yo], color=pm.INK_MUTED, lw=0.6,
                    alpha=0.85, zorder=2)
    for x in (4.1, 5.9):
        for y in LANE_Y:
            ax.plot([x], [y], "o", ms=2.8, mfc=pm.INK_SECONDARY,
                    mec=pm.SURFACE, mew=0.5, zorder=3)
    ax.text(5.0, 4.05, "shared model", ha="center", va="center", fontsize=6,
            color=pm.INK_SECONDARY, style="italic")
    caption(ax, 2)


def panel_d(ax):
    draw_box(ax, 3.65, 4.7, 2.8, 8.6)
    proto_y = {0: 10.35, 1: 7.45}
    for i, yc in enumerate(LANE_Y):
        cluster_marker(ax, 0.38, yc, CLUSTER_OF[i], ms=4.5)
        draw_series(ax, LANE_WAVES[i], 0.75, 2.55, yc)
        draw_arrow(ax, 2.7, 3.55, yc)
        draw_arrow(ax, 6.55, 7.35, yc)
        draw_series(ax, LANE_WAVES[i], 7.5, 9.8, yc, dashed=True)
        # Soft membership: a strong tie to the dominant cluster embedding, a
        # faint one to the other - never a hard route to just one.
        for cl, py in proto_y.items():
            dominant = cl == CLUSTER_OF[i]
            for xa, xb in ((4.05, 5.05), (5.05, 6.05)):
                ya, yb = (yc, py) if xa == 4.05 else (py, yc)
                ax.plot([xa, xb], [ya, yb], zorder=2,
                        color=ACCENT if dominant else pm.INK_MUTED,
                        lw=1.05 if dominant else 0.5,
                        ls="-" if dominant else (0, (1.2, 1.4)),
                        alpha=0.9 if dominant else 0.8)
    for x in (4.05, 6.05):
        for y in LANE_Y:
            ax.plot([x], [y], "o", ms=2.8, mfc=pm.INK_SECONDARY,
                    mec=pm.SURFACE, mew=0.5, zorder=3)
    for cl, py in proto_y.items():
        cluster_marker(ax, 5.05, py, cl, ms=10)
        ax.text(5.05, py, f"$c_{cl + 1}$", ha="center", va="center", fontsize=5.5,
                color=pm.SURFACE if cl == 0 else pm.label_color(ACCENT), zorder=5)
    ax.text(5.05, 4.05, "shared model, $k$ clusters", ha="center", va="center",
            fontsize=6, color=pm.INK_SECONDARY, style="italic")
    caption(ax, 3)


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": pm.FONT_STACKS["sans"],
        "mathtext.fontset": "dejavusans",
        "savefig.facecolor": pm.SURFACE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.35))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, wspace=0.07)
    for ax, panel in zip(axes, (panel_a, panel_b, panel_c, panel_d)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 13.6)
        ax.axis("off")
        panel(ax)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        path = OUTDIR / f"paradigms.{ext}"
        fig.savefig(path, dpi=400, facecolor=pm.SURFACE,
                    bbox_inches="tight", pad_inches=0.02)
        print(f"wrote {path.relative_to(Path.cwd())}"
              if path.is_relative_to(Path.cwd()) else f"wrote {path}")


if __name__ == "__main__":
    main()
