"""P7 and P8: what each paradigm costs, and what it does to the intervals.

P7  cost         accuracy against uncontended training cost (EMF only: VoIP
                 had no solo timing pass). Multivariate and cluster runs
                 trained one at a time, so their logged time is their cost; a
                 per-channel pipeline's cost is timing.csv's solo estimate
                 summed over its bands, with the fastest-to-slowest measured
                 band as a whisker where it is wider than the mark (a note
                 gives the range where it is not). Never the contended
                 per-channel log time.
P8  calibration  per-channel coverage and interval width of the conformal
                 intervals (target 1 - alpha = 90%), in P1's model rows and
                 paradigm lanes.

P7 puts every paradigm relative to the multivariate run of the same cell --
cost as a multiple of its training time, error as a relative change -- so the
three datasets share one origin per model and the question reads directly:
how much more (or less) error for how much more time.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator

from . import figdata as FD
from . import paradigms as PG
from .figkit import (
    INK_PRIMARY, INK_SECONDARY, PARTIAL_ALPHA, SURFACE, dataset_colors, dataset_name,
    fit_texts, footnote, footnotes, layout, legend, mark_kwargs, model_name, pad_range,
    panel_title, paradigm_handles, style_axes, title, window_label, write, xlabel,
)
from .fig_paradigms import (
    FLAG_NOTE, METRIC_NAME, WIDTH, _grid_height, _model_ticks, dot_panel, na_note,
)
from .style import log_ticks

PIPES = (PG.MV, PG.PC, PG.BEST)


def fig_cost(corpus: FD.Corpus, runs_csv: Path, timing_csv: Path, metric: str, style,
             root: Path, formats) -> list:
    """P7: each paradigm's error and cost relative to the multivariate run.

    One facet per model, the multivariate run at the origin: x is the
    pipeline's training cost as a multiple of the multivariate run's (log),
    y its error relative to the multivariate seed mean, on the seeds both
    have (paradigms.compare). Dataset by hue (the dataset palette, so the
    multivariate origin's Charcoal is never also a dataset), paradigm by shape.
    """
    if not Path(timing_csv).exists():
        return []
    costs = FD.costs(corpus, runs_csv, timing_csv)
    if not any(pipe == PG.PC for _, pipe in costs):
        return []
    board = corpus.board
    models = [m for m in corpus.models() if m not in PG.DETERMINISTIC
              and any((c, PG.MV) in costs for c in corpus.order if c[1] == m)]
    if not models:
        return []
    datasets = corpus.datasets
    hue = dataset_colors(datasets)
    ncol = min(3, len(models))
    nrow = -(-len(models) // ncol)
    height = 1.25 + 1.6 * nrow
    fig, axes = plt.subplots(nrow, ncol, figsize=(WIDTH, height), squeeze=False,
                             sharex=True, sharey=True)
    partial = False
    xs, ys = [1.0], [0.0]
    whiskers = []          # (ax, lo, hi, y, colour, alpha, estimate), drawn after layout
    for idx, model in enumerate(models):
        ax = axes[idx // ncol][idx % ncol]
        style_axes(ax, style, grid="both")
        ax.axhline(0, color=INK_SECONDARY, lw=0.6, zorder=2)
        ax.axvline(1, color=INK_SECONDARY, lw=0.6, zorder=2)
        ax.plot([1], [0], zorder=6, **mark_kwargs(PG.MV, style))
        for ds in datasets:
            cell = (ds, model)
            base = costs.get((cell, PG.MV))
            p = board.cells.get(cell, {})
            if base is None or PG.MV not in p:
                continue
            for pipe, marker, filled in ((PG.PC, "o", False), (PG.BEST, "D", True)):
                c = costs.get((cell, pipe))
                if c is None or pipe not in p:
                    continue
                r = PG.compare(p[PG.MV], p[pipe], metric)
                if r is None:
                    continue
                complete = board.complete(cell, PG.MV) and board.complete(cell, pipe)
                partial |= not complete
                alpha = 1.0 if complete else PARTIAL_ALPHA
                x, y = c.seconds / base.seconds, 100 * r["rel"]
                if c.lo is not None:
                    whiskers.append((ax, c.lo / base.seconds, c.hi / base.seconds, y, hue[ds],
                                     alpha, x))
                ax.plot([x], [y], marker=marker, linestyle="None", zorder=5, alpha=alpha,
                        markersize=style.marker * (0.9 if marker == "D" else 1.0),
                        markerfacecolor=hue[ds] if filled else SURFACE,
                        markeredgecolor=hue[ds], markeredgewidth=style.line_width * 0.5)
                xs.append(x)
                ys.append(y)
        panel_title(ax, model_name(model), style)
    for idx in range(len(models), nrow * ncol):
        axes[idx // ncol][idx % ncol].set_visible(False)
    for ax in axes.flat:
        ax.set_xscale("log")
        lo, hi = min(xs) / 1.6, max(xs) * 1.6
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(FixedLocator(log_ticks(lo, hi, 5)))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
        ax.set_ylim(*pad_range(ys, 0.1))
        ax.tick_params(axis="x", labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Δ {METRIC_NAME[metric]} vs MV (%)", fontsize=style.axis_label,
                      color=INK_SECONDARY)
    title(fig, f"{METRIC_NAME[metric]} against training cost", style,
          f"EMF · {window_label(corpus.window)} · relative to the multivariate run · "
          f"per-channel: solo timing pass, Σ bands")
    handles = [Line2D([], [], label="Multivariate", **mark_kwargs(PG.MV, style))]
    handles += [Line2D([], [], marker=mk, linestyle="None", markersize=style.marker * sz,
                       markerfacecolor=SURFACE if not filled else INK_SECONDARY,
                       markeredgecolor=INK_SECONDARY, markeredgewidth=style.line_width * 0.5,
                       label=lab)
                for mk, sz, filled, lab in (("o", 1.0, False, "Per-channel"),
                                            ("D", 0.9, True, "Cluster, best k"))]
    handles += [Line2D([], [], color=hue[ds], lw=2.2, label=dataset_name(ds)) for ds in datasets]
    layout(fig, style, top=0.84, bottom=0.42, left=0.09, wspace=0.08, hspace=0.42)
    drawn, note = _band_whiskers(fig, whiskers, style)
    if drawn:
        handles.append(Line2D([], [], color=INK_SECONDARY, lw=0.8, marker="|",
                              markersize=style.marker * 0.8, label="Fastest–slowest band"))
    if partial:
        # Grey, not Verdigris: in this figure a hue names a dataset.
        handles.append(Line2D([], [], marker="D", linestyle="None", alpha=PARTIAL_ALPHA,
                              markersize=style.marker * 0.9, markerfacecolor=INK_SECONDARY,
                              markeredgecolor=INK_SECONDARY,
                              markeredgewidth=style.line_width * 0.5, label="Still training"))
    legend(fig, handles, style, y=1 - 0.24 / height)
    if note:
        # Only the bottom margin moves: the whiskers were judged on the widths.
        layout(fig, style, top=0.84, bottom=0.52, left=0.09, wspace=0.08, hspace=0.42)
        footnote(fig, note, style)
    xlabel(fig, list(axes.flat), "Training time, × the multivariate run", style)
    return write(fig, root, group=corpus.family, subject="cost",
                 configs=[corpus.window], metric=metric, formats=formats)


def _band_whiskers(fig, whiskers, style) -> tuple:
    """Draw each per-channel bracket that is wider than its mark; (any drawn, note).

    The bracket -- every band at the fastest, or the slowest, measured band's
    seconds per epoch -- is a few percent of the estimate, which on a log axis
    spanning tens of times the multivariate cost hides under the marker. A
    bracket that would hide is not drawn; when none is drawn the note gives
    the range in words instead, so the legend never keys a mark nobody sees.
    """
    if not whiskers:
        return False, None
    mark_px = style.marker * fig.dpi / 72
    drawn = False
    for ax, lo, hi, y, color, alpha, _ in whiskers:
        (x0, _), (x1, _) = ax.transData.transform([(lo, y), (hi, y)])
        if x1 - x0 > 1.2 * mark_px:
            ax.plot([lo, hi], [y, y], color=color, lw=0.8, alpha=alpha, zorder=3,
                    marker="|", markersize=style.marker * 0.8, markevery=[0, 1],
                    solid_capstyle="butt")
            drawn = True
    if drawn:
        return True, None
    lo = min(w[1] / w[6] for w in whiskers) - 1
    hi = max(w[2] / w[6] for w in whiskers) - 1

    def pct(v):
        return f"{100 * v:+.0f}%".replace("-", "\u2212")

    return False, (f"Per-channel cost with every band at the fastest or the slowest band's "
                   f"speed: {pct(lo)} to {pct(hi)}, inside the marks")


def fig_calibration(corpora: list, runs_csv: Path, style, root: Path, formats, *,
                    family: str) -> list:
    """P8: coverage and interval width per model and paradigm, P1's lanes.

    Per dataset a pair of panels, coverage (target 1 - alpha dashed) beside
    width; one row of pairs per window.
    """
    if not corpora:
        return []
    datasets = list(dict.fromkeys(ds for c in corpora for ds in c.datasets))
    models = [m for m in dict.fromkeys(m for c in corpora for m in c.models())
              if m != "Persistence"]
    nrow, nd = len(corpora), len(datasets)
    height = _grid_height(len(models), nrow)
    fig = plt.figure(figsize=(WIDTH, height))
    ratios = []
    for i in range(nd):
        ratios += [1, 1] + ([0.28] if i < nd - 1 else [])
    gs = fig.add_gridspec(nrow, len(ratios), width_ratios=ratios, wspace=0.16,
                          hspace=0.3 if nrow > 1 else 0.2)
    partial = False
    flagged_any = False
    bottom_axes = []
    labels = {}
    for r, corpus in enumerate(corpora):
        cal = FD.calibration(corpus, runs_csv)
        first = None
        for d, dataset in enumerate(datasets):
            flagged = {m for ds, m in corpus.flagged if ds == dataset}
            flagged_any |= bool(flagged)
            for j, (key, name) in enumerate((("coverage", "Coverage (%)"),
                                            ("width", "Width"))):
                ax = fig.add_subplot(gs[r, 3 * d + j], sharey=first)
                first = first or ax
                style_axes(ax, style)
                xs, part, labels[ax] = dot_panel(
                    ax, corpus, dataset, models,
                    lambda cell, pipe, key=key: (cal.get((cell, pipe)) or {}).get(key),
                    style, k_labels=False)
                partial |= part
                if key == "coverage":
                    ax.axvline(FD.TARGET_COVERAGE, color=INK_SECONDARY, lw=0.7,
                               ls=(0, (3, 2)), zorder=2)
                    xs.append(FD.TARGET_COVERAGE)
                if xs:
                    ax.set_xlim(*pad_range(xs, 0.1))
                _model_ticks(ax, models, flagged, style)
                if not (d == 0 and j == 0):
                    ax.tick_params(axis="y", labelleft=False)
                ax.xaxis.set_major_locator(MaxNLocator(3))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.tick_params(axis="x", labelsize=style.tick * 0.9)
                if r == nrow - 1:
                    ax.set_xlabel(name, fontsize=style.axis_label * 0.9, color=INK_SECONDARY)
                    bottom_axes.append(ax)
                if j == 0:
                    head = dataset_name(dataset)
                    if nrow > 1:
                        head += f" · {window_label(corpus.window)}"
                    ax.annotate(head, (1.06, 1.0), xycoords="axes fraction", xytext=(0, 5),
                                textcoords="offset points", ha="center", va="bottom",
                                fontsize=style.panel_title, fontweight="bold",
                                color=INK_PRIMARY)
    fam = "EMF" if family == "emf" else "VoIP"
    windows = ", ".join(window_label(c.window) for c in corpora)
    title(fig, "Coverage and interval width", style,
          f"{fam} · {windows} · conformal, α = 0.1 (target dashed) · per-channel coverage")
    handles = paradigm_handles(style, (PG.MV, PG.PC, PG.K1, PG.BEST), partial=partial)
    legend(fig, handles, style, y=1 - 0.24 / height)
    notes = [FLAG_NOTE if flagged_any else None, na_note(models)]
    layout(fig, style, top=0.86, bottom=0.5 if any(notes) else 0.42, left=0.13)
    for ax, texts in labels.items():
        fit_texts(ax, texts)
    footnotes(fig, notes, style)
    return write(fig, root, group=family, subject="calibration",
                 configs=[c.window for c in corpora], formats=formats)
