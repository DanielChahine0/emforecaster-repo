"""P5: what clustering gains as k grows, per model.

One facet per clustered model, one line per dataset (the dataset palette,
never a paradigm's hue): each k's change against
the same network at k = 1 (the controlled baseline), as the mean of the
per-seed relative changes with a +-1 seed standard deviation ribbon. The k
chosen on validation is ringed; the k the test split would pick is marked
where it differs, so the distance between the two is visible. The fixed
train-mean membership protocol is primary; the logged batch protocol is the
faint dashed twin.

Replaces the ED2022-only absolute cluster-sweep figure of plot_metrics: every
dataset of the family, every model with a sweep, and the protocols side by
side. A model whose sweep is still training is drawn from what it has, its
points hollow, and gets no ring until its sweep is complete.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from . import figdata as FD
from .figkit import (
    INK_MUTED, INK_PRIMARY, INK_SECONDARY, SURFACE, dataset_colors, dataset_name,
    layout, legend, model_name, panel_title, style_axes, title, window_label, write,
    xlabel,
)
from .fig_paradigms import METRIC_NAME, WIDTH

NCOL = 3

# Marker size step (in marker units) between datasets' rings and test-best marks.
RING_STEP = 0.45


def fig_k_gain(corpus: FD.Corpus, metric: str, style, root: Path, formats) -> list:
    models = [m for m in corpus.models() if any(c[1] == m for c in corpus.board.sweeps)]
    if not models:
        return []
    datasets = corpus.datasets
    grid = list(corpus.board.grid)
    xpos = {k: i for i, k in enumerate(grid)}
    ncol = min(NCOL, len(models))
    nrow = -(-len(models) // ncol)
    height = 1.25 + 1.55 * nrow
    fig, axes = plt.subplots(nrow, ncol, figsize=(WIDTH, height), squeeze=False,
                             sharex=True, sharey=True)
    hue = dataset_colors(datasets)
    oracle_any = False
    partial_any = False
    for idx, model in enumerate(models):
        ax = axes[idx // ncol][idx % ncol]
        style_axes(ax, style, grid="y")
        ax.axhline(0, color=INK_SECONDARY, lw=0.6, zorder=2)
        partial = False
        for step, ds in enumerate(datasets):
            cell = (ds, model)
            prof = FD.k_profile(corpus, cell, metric, "fixed")
            if prof is None:
                continue
            color = hue[ds]
            x = np.array([xpos[p.k] for p in prof.points])
            y = np.array([100 * p.rel for p in prof.points])
            sd = np.array([100 * p.sd for p in prof.points])
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.13, lw=0, zorder=2)
            ax.plot(x, y, color=color, lw=style.line_width * 0.7, zorder=4,
                    solid_capstyle="round")
            for p, xi, yi in zip(prof.points, x, y):
                ax.plot([xi], [yi], marker="o", linestyle="None", zorder=5,
                        markersize=style.marker * 0.8,
                        markerfacecolor=color if p.complete else SURFACE,
                        markeredgecolor=color, markeredgewidth=style.line_width * 0.45)
                partial |= not p.complete
            # Each dataset's ring (and test-best mark) one size step up, so two
            # datasets choosing the same k at the same change -- every k = 1
            # sits at 0 -- stay two concentric marks rather than one.
            if prof.chosen is not None and prof.chosen in xpos:
                yc = y[[p.k for p in prof.points].index(prof.chosen)]
                ax.plot([xpos[prof.chosen]], [yc], marker="o", linestyle="None", zorder=6,
                        markersize=style.marker * (1.9 + RING_STEP * step),
                        markerfacecolor="none", markeredgecolor=color,
                        markeredgewidth=style.line_width * 0.5)
            if prof.oracle is not None and prof.oracle != prof.chosen and prof.oracle in xpos:
                yo = y[[p.k for p in prof.points].index(prof.oracle)]
                ax.plot([xpos[prof.oracle]], [yo], marker="v", linestyle="None",
                        zorder=6 - 0.1 * step,
                        markersize=style.marker * (0.9 + RING_STEP * step),
                        markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=0.6,
                        transform=ax.transData)
                oracle_any = True
            batch = FD.k_profile(corpus, cell, metric, "batch")
            if batch is not None:
                ax.plot([xpos[p.k] for p in batch.points], [100 * p.rel for p in batch.points],
                        color=color, lw=style.line_width * 0.45, ls=(0, (3, 2)), alpha=0.55,
                        zorder=3)
        partial_any |= partial
        panel_title(ax, model_name(model) + (" · partial" if partial else ""), style)
        if partial:
            ax.title.set_color(INK_MUTED)
    for idx in range(len(models), nrow * ncol):
        axes[idx // ncol][idx % ncol].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Δ {METRIC_NAME[metric]} vs k=1 (%)", fontsize=style.axis_label,
                      color=INK_SECONDARY)
    for ax in axes.flat:
        ax.set_xticks(range(len(grid)))
        ax.set_xticklabels([str(k) for k in grid])
        ax.set_xlim(-0.3, len(grid) - 0.7)
        ax.tick_params(axis="x", labelbottom=True)

    fam = "EMF" if corpus.family == "emf" else "VoIP"
    title(fig, f"Clustering gain against k · {METRIC_NAME[metric]}", style,
          f"{fam} · {window_label(corpus.window)} · fixed membership, "
          f"±1 seed sd; batch protocol dashed")
    handles = [Line2D([], [], color=hue[ds], lw=style.line_width * 0.7, marker="o",
                      markersize=style.marker * 0.8, label=dataset_name(ds)) for ds in datasets]
    handles.append(Line2D([], [], color=INK_SECONDARY, lw=style.line_width * 0.45,
                          ls=(0, (3, 2)), label="Batch protocol"))
    handles.append(Line2D([], [], marker="o", linestyle="None", markersize=style.marker * 1.6,
                          markerfacecolor="none", markeredgecolor=INK_PRIMARY,
                          markeredgewidth=style.line_width * 0.5, label="Val-chosen k"))
    if oracle_any:
        handles.append(Line2D([], [], marker="v", linestyle="None",
                              markersize=style.marker * 0.9, markerfacecolor=INK_PRIMARY,
                              markeredgecolor=SURFACE, label="Test-best k"))
    if partial_any:
        handles.append(Line2D([], [], marker="o", linestyle="None", markersize=style.marker * 0.8,
                              markerfacecolor=SURFACE, markeredgecolor=INK_PRIMARY,
                              label="Seeds missing"))
    legend(fig, handles, style, y=1 - 0.24 / height)
    layout(fig, style, top=0.82, bottom=0.42, left=0.09, wspace=0.08, hspace=0.42)
    xlabel(fig, list(axes.flat), "k (clusters)", style)
    return write(fig, root, group=corpus.family, subject="k-gain",
                 configs=[corpus.window], metric=metric, formats=formats)
