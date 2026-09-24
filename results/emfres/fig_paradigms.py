"""The paradigm comparison, drawn: dot plots, gap forests, the scoreboard.

P1  dots        each model under each paradigm, seed spread as ticks (EMF)
P4  dots        the same for VoIP, datasets x windows, Persistence as a rule
P2  gap-forest  every cell's gap with its seed-noise tie band, W/T/L and p
P6  scoreboard  the comparisons' W/T/L counts, EMF and both VoIP windows
P9  seed-skill  every seed's skill over the lookback mean: who collapses
P10 series dots the single-series grid, a separate dataset, on its own

Every number comes from emfres.figdata, which takes it from
emfres.paradigms -- the functions behind tables/paradigms.md.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from . import figdata as FD
from . import paradigms as PG

from .figkit import (
    BAND_FILL, BURNT_PEACH, CHARCOAL, INK_MUTED, INK_PRIMARY, INK_SECONDARY,
    LABEL_BOX, MUTED_COLOR, OUTCOME, PARTIAL_ALPHA, SURFACE, VERDIGRIS, dataset_name,
    fit_texts, footnote, label_color, layout, legend, mark_kwargs, model_name,
    outcome_handles, pad_range, panel_title, paradigm_handles, seed_ticks, style_axes,
    title, window_label, write, xlabel,
)

METRIC_NAME = dict(PG.METRICS)          # {"mae": "MAE", "mse": "MSE"}
WIDTH = 7.16                            # IEEEtran \textwidth, inches

# A reference forecast's rule (Persistence on VoIP).
REF_DASH = (0, (1.5, 1.5))

# Vertical offset of each paradigm's lane inside a model's row.
LANE = {PG.MV: 0.27, PG.PC: 0.0, "cluster": -0.27}

DAGGER = "†"


def _seed_span(corpus: FD.Corpus) -> str:
    seeds = corpus.board.seeds
    if not seeds:
        return "no seeds"
    return f"{len(seeds)} seed{'s' if len(seeds) > 1 else ''}"


def _model_ticks(ax, models, flagged_models, style) -> None:
    n = len(models)
    ax.set_yticks([n - 1 - i for i in range(n)])
    ax.set_yticklabels([model_name(m) + (f" {DAGGER}" if m in flagged_models else "")
                        for m in models])
    for label, m in zip(ax.get_yticklabels(), models):
        if m in flagged_models:
            label.set_color(INK_MUTED)
    ax.set_ylim(-0.6, n - 0.4)
    ax.tick_params(axis="y", length=0)


# --------------------------------------------------------------------------- #
# P1 / P4 / P9: one dot panel
# --------------------------------------------------------------------------- #

def dot_panel(ax, corpus: FD.Corpus, dataset: str, models: list, metric, style,
              *, per_seed: bool = False, k_labels: bool = True) -> tuple:
    """Draw every model's paradigms on one dataset.

    Returns (x values, any partial, labels); the labels hang off their marks
    by a fixed offset, so pass them to figkit.fit_texts once the layout is set.

    Three lanes per model row: multivariate (filled), per-channel (hollow),
    cluster (k = 1 as a small dot, the validation-chosen k as a diamond,
    joined to it by a hairline and labelled with its k on its own outer side,
    away from the k = 1 dot). Seeds are ticks behind the means, or, with
    ``per_seed``, each seed is its own mark and no mean is drawn.
    ``metric`` is a seed-metric name (figdata.marks) or a function
    (cell, pipe) -> figdata.Marks for values the seed metrics do not hold.
    """
    get = metric if callable(metric) else (lambda cell, pipe: FD.marks(corpus, cell, pipe, metric))
    xs: list = []
    labels: list = []
    partial = False
    n = len(models)
    for i, model in enumerate(models):
        y0 = n - 1 - i
        cell = (dataset, model)
        for pipe in (PG.MV, PG.PC):
            m = get(cell, pipe)
            if m is None:
                continue
            y = y0 + LANE[pipe]
            alpha = 1.0 if m.complete else PARTIAL_ALPHA
            partial |= not m.complete
            vals = list(m.seeds.values())
            if per_seed:
                ax.plot(vals, [y] * len(vals), zorder=4,
                        **mark_kwargs(pipe, style, alpha=0.85 * alpha, scale=0.8))
            else:
                seed_ticks(ax, vals, y, CHARCOAL, style, alpha=0.5 * alpha)
                ax.plot([m.mean], [y], zorder=5, **mark_kwargs(pipe, style, alpha=alpha))
            xs += vals
        if model in PG.DETERMINISTIC:
            continue
        y = y0 + LANE["cluster"]
        k1 = get(cell, PG.K1)
        best = get(cell, PG.BEST)
        if k1 is not None and not per_seed:
            ax.plot([k1.mean], [y], zorder=5,
                    **mark_kwargs(PG.K1, style, alpha=1.0 if k1.complete else PARTIAL_ALPHA))
            xs.append(k1.mean)
            partial |= not k1.complete
        if best is not None:
            alpha = 1.0 if best.complete else PARTIAL_ALPHA
            partial |= not best.complete
            vals = list(best.seeds.values())
            if per_seed:
                ax.plot(vals, [y] * len(vals), zorder=4,
                        **mark_kwargs(PG.BEST, style, alpha=0.85 * alpha, scale=0.8))
            else:
                seed_ticks(ax, vals, y, VERDIGRIS, style, alpha=0.55 * alpha)
                if k1 is not None:
                    ax.plot([k1.mean, best.mean], [y, y], color=VERDIGRIS, lw=0.7,
                            alpha=0.7 * alpha, zorder=4, solid_capstyle="butt")
                ax.plot([best.mean], [y], zorder=6, **mark_kwargs(PG.BEST, style, alpha=alpha))
            xs += vals
            if not per_seed and k_labels:
                # The label sits on the diamond's own outer side -- left of it
                # when clustering beat k = 1 -- so it never lands beside the
                # k = 1 dot, and past its seed ticks, so it covers none.
                left = k1 is not None and best.mean < k1.mean
                at = min([best.mean] + vals) if left else \
                    max([best.mean] + vals + ([k1.mean] if k1 is not None else []))
                labels.append(ax.annotate(
                    f"k={best.k}" if best.complete else "partial", (at, y),
                    xytext=(-style.marker * 0.9 if left else style.marker * 0.9, 0),
                    textcoords="offset points", ha="right" if left else "left", va="center",
                    fontsize=style.context * 0.95,
                    color=label_color(VERDIGRIS) if best.complete else INK_MUTED,
                    zorder=7, bbox=LABEL_BOX))
    return xs, partial, labels


def _grid_height(n_models: int, nrow: int) -> float:
    """Inches for nrow facet rows of n_models model rows, with the header and axis."""
    return 1.3 + 0.27 * n_models * nrow + 0.4 * (nrow - 1)


def _flagged_models(corpus: FD.Corpus, dataset: str) -> set:
    return {m for ds, m in corpus.flagged if ds == dataset}


def fig_dots(corpora: list, metric: str, style, root: Path, formats, *,
             family: str) -> list:
    """P1 (one corpus: facets = datasets) and P4 (several windows: rows = windows).

    A model the family runs as a trivial reference (Persistence) is drawn as a
    vertical rule instead of a row.
    """
    if not corpora:
        return []
    ref_models = {"Persistence"}
    datasets = list(dict.fromkeys(ds for c in corpora for ds in c.datasets))
    models = [m for m in dict.fromkeys(m for c in corpora for m in c.models())
              if m not in ref_models]
    nrow, ncol = len(corpora), len(datasets)
    fig, axes = plt.subplots(nrow, ncol, figsize=(WIDTH, _grid_height(len(models), nrow)),
                             squeeze=False, sharey=True)
    partial = False
    flagged_any = False
    ref_drawn: set = set()
    labels = {}
    for r, corpus in enumerate(corpora):
        for c, dataset in enumerate(datasets):
            ax = axes[r][c]
            style_axes(ax, style)
            xs, part, labels[ax] = dot_panel(ax, corpus, dataset, models, metric, style)
            partial |= part
            refs = []
            for ref in ref_models:
                m = FD.marks(corpus, (dataset, ref), PG.MV, metric)
                if m is not None:
                    ax.axvline(m.mean, color=INK_SECONDARY, lw=0.8, ls=REF_DASH, zorder=2)
                    refs.append(m.mean)
                    ref_drawn.add(ref)
            if xs:
                ax.set_xlim(*pad_range(xs + refs, 0.05))
            flagged = _flagged_models(corpus, dataset)
            flagged_any |= bool(flagged)
            _model_ticks(ax, models, flagged, style)
            head = dataset_name(dataset)
            if nrow > 1:
                head += f" · {window_label(corpus.window)}"
            panel_title(ax, head, style)
    windows = ", ".join(window_label(c.window) for c in corpora)
    title(fig, f"{METRIC_NAME[metric]} by paradigm",
          style, f"{family.upper() if family == 'emf' else 'VoIP'} · {windows} · "
                 f"{_seed_span(corpora[0])} · cluster k chosen on validation")
    handles = paradigm_handles(style, (PG.MV, PG.PC, PG.K1, PG.BEST), partial=partial)
    handles += [Line2D([], [], color=INK_SECONDARY, lw=0.8, ls=REF_DASH, label=model_name(ref))
                for ref in sorted(ref_drawn)]
    legend(fig, handles, style, y=1 - 0.24 / fig.get_size_inches()[1])
    layout(fig, style, top=0.86, bottom=0.42, wspace=0.08, hspace=0.3 if nrow > 1 else 0.2)
    for ax, texts in labels.items():
        fit_texts(ax, texts)
    xlabel(fig, axes[-1], METRIC_NAME[metric], style)
    if flagged_any:
        footnote(fig, f"{DAGGER} no seed trains in any paradigm", style)
    return write(fig, root, group=family, subject="dots",
                 configs=[c.window for c in corpora], metric=metric, formats=formats)


# --------------------------------------------------------------------------- #
# P2: gap forest
# --------------------------------------------------------------------------- #

def _forest_rows(corpus: FD.Corpus) -> list:
    """[(y, label, cell or None)]: a dataset header row, then its gradient models."""
    rows, y = [], 0.0
    for dataset in corpus.datasets:
        models = [m for m in corpus.models(dataset) if m not in PG.DETERMINISTIC]
        if not models:
            continue
        rows.append((y, dataset_name(dataset), None))
        y -= 1
        for m in models:
            rows.append((y, m, (dataset, m)))
            y -= 1
        y -= 0.4
    return rows


def _score_text(t: dict) -> str:
    c = t["count"]
    txt = f"{c['challenger']} / {c['tie']} / {c['baseline']}"
    if t["p"] is not None:
        txt += f" · p = {t['p']:.2g}"
    if t["pending"]:
        txt += f" · {t['pending']} pending"
    return txt


def fig_forest(corpus: FD.Corpus, metric: str, style, root: Path, formats) -> list:
    """P2: per-cell relative gap of each comparison, the tie band behind it."""
    rows = _forest_rows(corpus)
    if not rows:
        return []
    comps = [(b, c, lab) for b, c, lab in FD.GAP_COMPARISONS
             if FD.gaps(corpus, b, c, metric)]
    if not comps:
        return []
    height = 1.35 + 0.135 * len(rows)
    fig, axes = plt.subplots(1, len(comps), figsize=(WIDTH, height), sharey=True, squeeze=False)
    axes = axes[0]
    pending_any = False
    for ax, (base, chal, label) in zip(axes, comps):
        style_axes(ax, style)
        by_cell = {g.cell: g for g in FD.gaps(corpus, base, chal, metric)}
        xs, bands = [0.0], []
        for y, _, cell in rows:
            g = by_cell.get(cell) if cell else None
            if g is None:
                continue
            ax.add_patch(Rectangle((-100 * g.band, y - 0.36), 200 * g.band, 0.72,
                                   facecolor=BAND_FILL, edgecolor="none", zorder=1))
            color, filled = OUTCOME[g.outcome]
            ax.plot([100 * g.rel], [y], marker="o", linestyle="None", zorder=4,
                    markersize=style.marker * (0.85 if g.outcome == "pending" else 1.0),
                    markerfacecolor=color if filled else SURFACE, markeredgecolor=color,
                    markeredgewidth=style.line_width * 0.5)
            if g.outcome == "pending":
                pending_any = True
                ax.annotate(f"n={g.n}", (100 * g.rel, y), xytext=(style.marker * 0.9, 0),
                            textcoords="offset points", va="center", ha="left",
                            fontsize=style.context * 0.9, color=INK_MUTED)
            xs.append(100 * g.rel)
            bands.append(100 * g.band)
        ax.axvline(0, color=INK_SECONDARY, lw=0.6, zorder=2)
        # A band wider than the gaps themselves runs off the axis rather than
        # squeezing every gap into the middle (the Transformer's seed spread).
        reach = max(max(abs(x) for x in xs), 0.5)
        extent = xs + [s * b for b in bands if b <= 1.5 * reach for s in (-1, 1)]
        lo, hi = pad_range(extent, 0.1)
        ax.set_xlim(min(lo, -0.1 * reach), max(hi, 0.1 * reach))
        ax.set_xlabel(f"Δ {METRIC_NAME[metric]} (%)", fontsize=style.axis_label,
                      color=INK_SECONDARY)
        t = FD.score(corpus, base, chal, metric)
        panel_title(ax, label, style, sub=_score_text(t))

    ax0 = axes[0]
    ax0.set_yticks([y for y, _, _ in rows])
    ax0.set_yticklabels([model_name(lab) + (f" {DAGGER}" if cell in corpus.flagged else "")
                         if cell else lab for _, lab, cell in rows])
    for tick, (_, _, cell) in zip(ax0.get_yticklabels(), rows):
        if cell is None:
            tick.set_fontweight("bold")
        elif cell in corpus.flagged:
            tick.set_color(INK_MUTED)
    ax0.tick_params(axis="y", length=0)
    ax0.set_ylim(rows[-1][0] - 0.7, rows[0][0] + 0.7)

    fam = "EMF" if corpus.family == "emf" else "VoIP"
    title(fig, f"{METRIC_NAME[metric]} gap per cell", style,
          f"{fam} · {window_label(corpus.window)} · challenger / tie / baseline cells, "
          f"Wilcoxon p")
    legend(fig, outcome_handles(style, pending=pending_any), style, y=1 - 0.24 / height)
    layout(fig, style, top=0.96, bottom=0.42, left=0.14, wspace=0.1)
    if corpus.flagged:
        footnote(fig, f"{DAGGER} no seed trains in any paradigm", style)
    return write(fig, root, group=corpus.family, subject="gap-forest",
                 configs=[corpus.window], metric=metric, formats=formats)


# --------------------------------------------------------------------------- #
# P6: scoreboard
# --------------------------------------------------------------------------- #

SCORE_FILL = [("challenger", VERDIGRIS, "white"), ("tie", MUTED_COLOR, INK_PRIMARY),
              ("baseline", BURNT_PEACH, "white"), ("pending", SURFACE, INK_SECONDARY)]


def _score_rows(corpus: FD.Corpus, metric: str) -> list:
    """[(label, tally, sub)] for each comparison, and its no-training variant.

    The variant without the cells that do not train is listed where it
    changes a count or the p-value; where those cells are only pending on
    one side, dropping them changes nothing counted and it is left out.
    """
    rows = []
    for base, chal, label in FD.SCORE_COMPARISONS:
        t = FD.score(corpus, base, chal, metric)
        if not (t["cells"] or t["pending"]):
            continue
        rows.append((label, t, False))
        if corpus.flagged:
            t2 = FD.score(corpus, base, chal, metric, skip=corpus.flagged)
            if (t2["count"], t2["p"]) != (t["count"], t["p"]):
                names = sorted({model_name(m) for _, m in corpus.flagged})
                rows.append((f"without {', '.join(names)}", t2, True))
    return rows


def fig_scoreboard(corpora: list, style, root: Path, formats) -> list:
    """P6: stacked W/T/L bars per comparison, one block per family window."""
    blocks = [(c, {m: _score_rows(c, m) for m, _ in PG.METRICS}) for c in corpora]
    blocks = [(c, rows) for c, rows in blocks if rows["mae"]]
    if not blocks:
        return []
    heights = [max(len(rows[m]) for m, _ in PG.METRICS) for _, rows in blocks]
    height = 1.0 + 0.24 * sum(heights) + 0.42 * (len(blocks) - 1)
    fig, axes = plt.subplots(len(blocks), 2, figsize=(WIDTH, height), squeeze=False,
                             gridspec_kw={"height_ratios": heights,
                                          "hspace": 0.42 / (0.24 * statistics.fmean(heights)),
                                          "wspace": 0.06})
    for r, (corpus, by_metric) in enumerate(blocks):
        fam = "EMF" if corpus.family == "emf" else "VoIP"
        for c, (metric, name) in enumerate(PG.METRICS):
            ax = axes[r][c]
            rows = by_metric[metric]
            total = max((sum(t["count"].values()) + t["pending"] for _, t, _ in rows), default=1)
            for i, (label, t, sub) in enumerate(rows):
                y = -i
                left = 0
                h = 0.6 if sub else 0.78
                for key, fill, ink in SCORE_FILL:
                    v = t["pending"] if key == "pending" else t["count"][key]
                    if not v:
                        continue
                    ax.barh(y, v, left=left, height=h, color=fill,
                            edgecolor=SURFACE if key != "pending" else INK_MUTED,
                            linewidth=1.2 if key != "pending" else 0.6,
                            hatch="////" if key == "pending" else None, zorder=3)
                    ax.text(left + v / 2, y, str(v), ha="center", va="center",
                            fontsize=style.context * (0.85 if sub else 1.0), color=ink,
                            fontweight="bold" if key in ("challenger", "baseline") else None,
                            zorder=4, bbox=dict(boxstyle="square,pad=0.4", fc=SURFACE,
                                                ec="none") if key == "pending" else None)
                    left += v
                p = "–" if t["p"] is None else f"p = {t['p']:.2g}"
                ax.text(total * 1.02, y, p, ha="left", va="center",
                        fontsize=style.context * (0.85 if sub else 1.0),
                        color=INK_MUTED if sub else INK_SECONDARY)
            ax.set_xlim(0, total * 1.2)
            ax.set_ylim(-max(heights[r], len(rows)) + 0.5, 0.5)
            for side in ("top", "right", "bottom", "left"):
                ax.spines[side].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([-i for i in range(len(rows))])
            if c == 0:
                ax.set_yticklabels([lab for lab, _, _ in rows])
                for tick, (_, _, sub) in zip(ax.get_yticklabels(), rows):
                    tick.set_fontsize(style.tick * (0.85 if sub else 1.0))
                    tick.set_color(INK_MUTED if sub else INK_PRIMARY)
            else:
                ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
            head = f"{fam} · {window_label(corpus.window)} · {name}"
            n = rows[0][1]["cells"] + rows[0][1]["pending"]
            panel_title(ax, head, style, pad=3)
            ax.annotate(f"{n} cells", (1.0, 1.0), xycoords="axes fraction", xytext=(0, 3),
                        textcoords="offset points", ha="right", va="bottom",
                        fontsize=style.context * 0.9, color=INK_MUTED)
    handles = [Patch(facecolor=VERDIGRIS, label="Challenger better"),
               Patch(facecolor=MUTED_COLOR, label="Tie"),
               Patch(facecolor=BURNT_PEACH, label="Baseline better"),
               Patch(facecolor=SURFACE, edgecolor=INK_MUTED, hatch="////", label="Pending")]
    title(fig, "Scoreboard", style, "cells won by the challenger, tied within seed noise, "
                                    "or won by the baseline · Wilcoxon p")
    legend(fig, handles, style, y=1 - 0.24 / height)
    layout(fig, style, top=0.9, bottom=0.05, left=0.2)
    return write(fig, root, group=None, subject="scoreboard",
                 configs=[c.window for c, _ in blocks], formats=formats)


# --------------------------------------------------------------------------- #
# P9: per-seed skill
# --------------------------------------------------------------------------- #

def fig_seed_skill(corpora: list, style, root: Path, formats, *, family: str) -> list:
    """P9: every seed's skill over the lookback mean; the trains threshold dashed."""
    if not corpora:
        return []
    datasets = list(dict.fromkeys(ds for c in corpora for ds in c.datasets))
    models = [m for m in dict.fromkeys(m for c in corpora for m in c.models())
              if m != "Persistence"]
    nrow, ncol = len(corpora), len(datasets)
    fig, axes = plt.subplots(nrow, ncol, figsize=(WIDTH, _grid_height(len(models), nrow)),
                             squeeze=False, sharey=True)
    partial = False
    flagged_any = False
    for r, corpus in enumerate(corpora):
        for c, dataset in enumerate(datasets):
            ax = axes[r][c]
            style_axes(ax, style)
            xs, part, _ = dot_panel(ax, corpus, dataset, models, "skill", style, per_seed=True)
            partial |= part
            ax.axvline(0, color=INK_SECONDARY, lw=0.6, zorder=2)
            thr = FD.trains_threshold(corpus, dataset)
            if thr is not None:
                ax.axvline(thr, color=BURNT_PEACH, lw=0.8, ls=(0, (3, 2)), zorder=2)
                ax.annotate("Transformer trains", (thr, 1.0),
                            xycoords=("data", "axes fraction"),
                            xytext=(2, -1), textcoords="offset points", ha="left", va="top",
                            fontsize=style.context * 0.95, color=label_color(BURNT_PEACH))
                xs.append(thr)
            if xs:
                ax.set_xlim(*pad_range(xs + [0.0], 0.06))
            flagged = _flagged_models(corpus, dataset)
            flagged_any |= bool(flagged)
            _model_ticks(ax, models, flagged, style)
            head = dataset_name(dataset)
            if nrow > 1:
                head += f" · {window_label(corpus.window)}"
            panel_title(ax, head, style)
    windows = ", ".join(window_label(c.window) for c in corpora)
    fam = "EMF" if family == "emf" else "VoIP"
    title(fig, "Skill per seed", style, f"{fam} · {windows} · 0 = the lookback-mean forecast")
    handles = paradigm_handles(style, (PG.MV, PG.PC, PG.BEST), partial=partial, seeds=False)
    legend(fig, handles, style, y=1 - 0.24 / fig.get_size_inches()[1])
    layout(fig, style, top=0.86, bottom=0.42, wspace=0.08, hspace=0.3 if nrow > 1 else 0.2)
    xlabel(fig, axes[-1], "Skill (1 − MSE / lookback-mean MSE)", style)
    if flagged_any:
        footnote(fig, f"{DAGGER} no seed trains in any paradigm", style)
    return write(fig, root, group=family, subject="seed-skill",
                 configs=[c.window for c in corpora], formats=formats)


# --------------------------------------------------------------------------- #
# P10: the single-series grid
# --------------------------------------------------------------------------- #

# ptv is det shifted by one 6-minute step (audit section 2.4): two signals, not three.
SERIES_NOTES = {"ptv": "DET shifted one step"}


def fig_series(runs_csv: Path, metric: str, style, root: Path, formats, *,
               dataset_rank=lambda d: d, model_rank=lambda m: m) -> list:
    """P10: the series grid at its benchmark window, titled as the separate dataset it is."""
    values = FD.series_values(runs_csv)
    if not values:
        return []
    datasets = sorted({ds for ds, _ in values}, key=dataset_rank)
    models = sorted({m for _, m in values}, key=model_rank)
    n = len(models)
    height = 1.4 + 0.2 * n
    fig, axes = plt.subplots(1, len(datasets), figsize=(WIDTH, height), sharey=True,
                             squeeze=False)
    seeds = set()
    for ax, dataset in zip(axes[0], datasets):
        style_axes(ax, style)
        xs = []
        for i, model in enumerate(models):
            by_seed = values.get((dataset, model))
            if not by_seed:
                continue
            seeds |= set(by_seed)
            vals = [v[metric] for v in by_seed.values()]
            y = n - 1 - i
            seed_ticks(ax, vals, y, CHARCOAL, style, alpha=0.6)
            ax.plot([statistics.fmean(vals)], [y], zorder=5, **mark_kwargs(PG.MV, style))
            xs += vals
        if xs:
            ax.set_xlim(*pad_range(xs, 0.06))
        ax.set_yticks([n - 1 - i for i in range(n)])
        ax.set_yticklabels([model_name(m) for m in models])
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, n - 0.4)
        # Every panel gets the note line (empty where there is none), so the
        # titles stay level.
        panel_title(ax, dataset_name(dataset), style,
                    sub=SERIES_NOTES.get(dataset, "") if any(
                        d in SERIES_NOTES for d in datasets) else None)
    title(fig, f"Univariate series, a separate dataset · {METRIC_NAME[metric]}", style,
          f"{window_label(FD.SERIES_WINDOW)} · {len(seeds)} seeds · never paired with "
          f"the paradigm grids")
    handles = [Line2D([], [], label="Mean over seeds", **mark_kwargs(PG.MV, style)),
               Line2D([], [], marker="|", linestyle="None", color=CHARCOAL,
                      markersize=style.marker * 1.1, label="Seed")]
    legend(fig, handles, style, y=1 - 0.24 / height)
    layout(fig, style, top=0.98, bottom=0.42, wspace=0.08)
    xlabel(fig, axes[0], METRIC_NAME[metric], style)
    return write(fig, root, family="series", group=None, subject="dots",
                 configs=[FD.SERIES_WINDOW], metric=metric, formats=formats)
