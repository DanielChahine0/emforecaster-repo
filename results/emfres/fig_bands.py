"""P3: is a paradigm's advantage broad, or carried by a few bands?

One figure per dataset and comparison. The heatmap is log2(challenger /
baseline) of each band's test MSE, seed-mean over the seeds both pipelines
have: Verdigris where the challenger is better on that band, Burnt Peach
where the baseline is. The strip under it gives each band's signed share of
the dataset's band gaps (figdata.BandGrid.share: the band's gap summed over
models, over the sum of every band's absolute gap), so the bars' lengths add
up to 100% and a gap concentrated in a few bands shows as a few long bars. A
model that does not train on the dataset (corpus.flagged) keeps its row,
marked with a dagger and muted, but is left out of the strip. A model the
comparison applies to but that has no row -- a side still training, or no
band vectors -- is named in a note.

Two comparisons: per-channel against multivariate, and the validation-chosen
k against k = 1 (the fixed-membership band vectors, once a model's sweep is
complete). EMF bands are labelled by frequency, VoIP channels by signal.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from . import figdata as FD
from . import paradigms as PG
from .figkit import (
    BURNT_PEACH, INK_MUTED, INK_SECONDARY, VERDIGRIS, dataset_name, diverging_cmap,
    footnotes, model_name, style_axes, title, window_label, write,
)
from .fig_paradigms import DAGGER, WIDTH

COMPARISONS = [
    (PG.MV, PG.PC, "pc-vs-mv", "Per-channel vs MV", "log₂(PC / MV)"),
    (PG.K1, PG.BEST, "best-k-vs-k1", "Best k vs k=1", "log₂(best k / k=1)"),
]

# Colour saturates at this |log2 ratio| (2x); a band beyond it is still drawn,
# at the end colour, and its value is printed in the cell.
CLIP = 1.0


def _band_label(name: str) -> str:
    try:
        v = float(name)
    except (TypeError, ValueError):
        return str(name)
    return f"{v:g}"


def fig_band_grid(grid: FD.BandGrid, corpus: FD.Corpus, spec: tuple, style, root: Path,
                  formats) -> list:
    base, chal, slug, label, cbar = spec
    models, chans = grid.models, grid.channels
    nb, nm = len(chans), len(models)
    # A model that does not train here keeps its row, marked and muted as in
    # every other paradigm figure, but stays out of the share strip.
    flagged = {m for ds, m in corpus.flagged if ds == grid.dataset} & set(models)
    # The models the comparison applies to that have no row: named in a note
    # rather than silently absent.
    pending = [m for ds, m in FD.pending_cells(corpus, base, chal) if ds == grid.dataset]
    unbanded = [m for ds, m in corpus.order if ds == grid.dataset and m not in models
                and m not in pending and m not in PG.DETERMINISTIC
                and corpus.board.applies((ds, m), base) and corpus.board.applies((ds, m), chal)]
    notes = [f"{DAGGER} no seed trains in any paradigm; left out of the share strip"
             if flagged else None,
             "pending, not drawn: " + ", ".join(model_name(m) for m in pending)
             if pending else None,
             "no band vectors: " + ", ".join(model_name(m) for m in unbanded)
             if unbanded else None]
    heat_h = 0.2 * nm + 0.1
    strip_h = 0.75
    bottom = 0.55 + (0.12 if any(notes) else 0.0)
    height = 0.45 + heat_h + strip_h + bottom
    fig = plt.figure(figsize=(WIDTH, height))
    gs = fig.add_gridspec(2, 2, height_ratios=[heat_h, strip_h], width_ratios=[1, 0.018],
                          hspace=0.12, wspace=0.02,
                          left=0.14, right=0.93, top=1 - 0.4 / height, bottom=bottom / height)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    sx = fig.add_subplot(gs[1, 0], sharex=ax)

    data = np.full((nm, nb), np.nan)
    for i, m in enumerate(models):
        for j, ch in enumerate(chans):
            data[i, j] = grid.ratio.get((m, ch), np.nan)
    # The scale is set by the models that train; a flagged row beyond it is
    # drawn at the end colour with its value printed, like any clipped cell.
    scaled = data[[i for i, m in enumerate(models) if m not in flagged] or list(range(nm))]
    vmax = min(CLIP, max(0.05, float(np.nanmax(np.abs(scaled)))
                         if np.isfinite(scaled).any() else CLIP))
    im = ax.imshow(np.clip(data, -vmax, vmax), aspect="auto", cmap=diverging_cmap(),
                   norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
                   interpolation="nearest")
    for i in range(nm):
        for j in range(nb):
            if np.isfinite(data[i, j]) and abs(data[i, j]) > vmax:
                ax.text(j, i, f"{data[i, j]:+.1f}", ha="center", va="center",
                        fontsize=style.context * 0.7, color="white")
    ax.set_yticks(range(nm))
    ax.set_yticklabels([model_name(m) + (f" {DAGGER}" if m in flagged else "") for m in models],
                       fontsize=style.tick)
    for tick, m in zip(ax.get_yticklabels(), models):
        if m in flagged:
            tick.set_color(INK_MUTED)
    ax.tick_params(axis="both", length=0, labelbottom=False)
    for side in ax.spines.values():
        side.set_visible(False)
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=style.tick * 0.9, length=0)
    cb.set_label(cbar, fontsize=style.axis_label * 0.9, color=INK_SECONDARY)

    share = grid.share(skip=flagged)
    vals = [100 * share.get(ch, 0.0) for ch in chans]
    style_axes(sx, style, grid="y")
    sx.bar(range(nb), vals, width=0.72, color=[BURNT_PEACH if v > 0 else VERDIGRIS for v in vals],
           zorder=3)
    sx.axhline(0, color=INK_SECONDARY, lw=0.6, zorder=4)
    sx.set_ylabel("Share of gap (%)", fontsize=style.axis_label * 0.9, color=INK_SECONDARY)
    sx.set_xticks(range(nb))
    names = [_band_label(grid.names.get(ch, ch)) for ch in chans]
    rotate = max(len(n) for n in names) > 3 and nb > 8
    sx.set_xticklabels(names, rotation=90 if rotate else 0,
                       fontsize=style.tick * (0.8 if rotate else 1.0))
    sx.set_xlim(-0.5, nb - 0.5)
    lim = max(5.0, max(abs(v) for v in vals) * 1.15)
    sx.set_ylim(-lim, lim)
    sx.set_xlabel("Band (MHz)" if rotate else "Signal", fontsize=style.axis_label,
                  color=INK_SECONDARY)

    fam = "EMF" if corpus.family == "emf" else "VoIP"
    seeds = sorted(set(grid.n.values()))
    title(fig, f"{label}, band by band · {dataset_name(grid.dataset)}", style,
          f"{fam} · {window_label(corpus.window)} · MSE · "
          f"{'/'.join(map(str, seeds))} matched seeds · positive = baseline better")
    footnotes(fig, notes, style)
    return write(fig, root, family="per-band", group=dataset_name(grid.dataset), subject=slug,
                 configs=[corpus.window], metric="mse", formats=formats)


def fig_bands(corpus: FD.Corpus, bands_csv: Path, style, root: Path, formats) -> list:
    written = []
    for spec in COMPARISONS:
        for grid in FD.band_grids(corpus, bands_csv, spec[0], spec[1]):
            written += fig_band_grid(grid, corpus, spec, style, root, formats)
    return written
