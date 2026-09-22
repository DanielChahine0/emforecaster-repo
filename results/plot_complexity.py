#!/usr/bin/env python3
"""Model-complexity poster figures from tables.md's Multivariate section.

Two figures land in ``figures/complexity/``:

``multivariate_complexity.png``
    A two-panel strip: trainable-parameter bars (log scale, each bar carrying
    its human-readable count) beside the total-training-time panel, one line
    per dataset - ARIMA keeps an empty, annotated slot in both.

``multivariate_mae-vs-tt_params.png``
    A bubble scatter of the accuracy/cost/size trade-off: MAE (averaged over
    the three datasets) against total training time on a log axis, bubble
    area proportional to the parameter count. ARIMA, with neither logged,
    sits out with a note.

Everything reads the L336/O96 multivariate cells of ``tables.md`` and reuses
plot_metrics' palette, panel machinery, and save conventions, so the figures
match the rest of the repo's set.

Usage
-----
    python3 plot_complexity.py                 # slide proportions, PNG
    python3 plot_complexity.py --style paper   # IEEEtran column sizing
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_metrics as pm
from emfres.chrome import figure_header, footnote, shared_legend
from emfres.naming import artifact_path

HERE = Path(__file__).resolve().parent
SECTION = "Multivariate"
GRID = "336/96"
PARAMS_COLUMN = "L336/O96"

# The parameter bars wear one recessive slate: the two-panel figure keys its
# colours to the datasets (panel B), so panel A stays out of that palette and
# lets the annotated counts carry the story.
BAR_COLOR = pm.INK_SECONDARY

# Bubble area maps linearly onto the parameter count, floored so the smallest
# model (LSTM, 20.7k) stays visible next to EMForecaster's 2.4M.
MAX_BUBBLE_AREA = 2600.0   # pt^2, assigned to the largest parameter count
MIN_BUBBLE_AREA = 55.0     # pt^2 floor for visibility
SIZE_KEY = [(20_000, "20k"), (400_000, "400k"), (2_400_000, "2.4M")]

# Spot checks against the frozen tables (archive/2026-09-22_tables.md, the
# three-seed Phase 2 means): if any of these stops matching, either the parser
# or the tables changed, and the run aborts loudly instead of drawing a wrong
# figure.
CHECK_PARAMS = {
    "EMForecaster": 2_416_959, "DLinear": 64_704, "TSMixer": 56_065,
    "PatchTST": 72_182, "Transformer": 372_738, "LSTM": 20_726,
}
CHECK_TIME_S = {
    ("ED2022", "EMForecaster"): 13 * 60 + 21,
    ("ED2022", "DLinear"): 60,
    ("EMF2023", "Transformer"): 40 * 60 + 55,
    ("EMF2023", "DLinear"): 41,
}
CHECK_MAE = {
    ("ED2022", "EMForecaster"): 0.3358,
    ("ED2022", "ARIMA"): 0.6076,
}


# --------------------------------------------------------------------------- #
# Parsing and checking
# --------------------------------------------------------------------------- #


def human_count(n: float) -> str:
    """``2416959`` -> ``2.42M``; ``64704`` -> ``64.7k``."""
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return f"{n / 1e3:.1f}k"


def params_from_records(records) -> dict[str, float]:
    """Trainable-weight counts, keyed by short model name.

    This replaces a 45-line second copy of the table walker that existed only
    because the 4.1 parameters table has a different shape (``Model |
    L48/O96 | ...``) from the metric tables, so plot_metrics' parser skipped
    it. Read from frames/cells.csv the shape difference simply does not
    arise. ARIMA has no count -- it is not gradient-trained -- and stays
    absent, which verify() asserts.
    """
    out: dict[str, float] = {}
    lookback, horizon = pm.parse_grid(GRID)
    for r in records:
        if (r.metric != "Parameters" or r.section != SECTION
                or r.lookback != lookback or r.horizon != horizon):
            continue
        if math.isnan(r.value):
            continue
        out.setdefault(pm.short_model(r.model), r.value)
    return out

def cell(values: dict[str, dict[str, float]], dataset: str, model: str) -> float:
    """Look one (dataset, model) cell up by short names in an aggregate dict."""
    for d, per_model in values.items():
        if pm.short_dataset(d) != dataset:
            continue
        for m, v in per_model.items():
            if pm.short_model(m) == model:
                return v
    return float("nan")


def verify(params, tvals, mvals) -> None:
    problems = []
    for model, expected in CHECK_PARAMS.items():
        got = params.get(model, float("nan"))
        if not math.isclose(got, expected, rel_tol=1e-9):
            problems.append(f"params[{model}] = {got!r}, expected {expected}")
    if "ARIMA" in params:
        problems.append(f"params[ARIMA] = {params['ARIMA']!r}, expected absent")
    # The time and MAE constants were read off the rendered tables, which
    # round to the second and to 4 dp respectively. Assert at that precision:
    # the frames carry more, and demanding an exact match would only be
    # re-asserting the rounding that used to be applied on the way in.
    for (dataset, model), expected in CHECK_TIME_S.items():
        got = cell(tvals, dataset, model)
        if round(got) != expected:
            problems.append(f"time[{dataset}, {model}] = {got:.1f}s, expected {expected}s")
    for (dataset, model), expected in CHECK_MAE.items():
        got = cell(mvals, dataset, model)
        if round(got, 4) != expected:
            problems.append(f"MAE[{dataset}, {model}] = {got:.4f}, expected {expected}")
    if problems:
        raise SystemExit("values disagree with the spot checks:\n  " + "\n  ".join(problems))


# --------------------------------------------------------------------------- #
# Figure 1: parameter bars | training-time lines
# --------------------------------------------------------------------------- #


def count_tick(v: float) -> str:
    return f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k"


def draw_params_panel(ax, models: list[str], params: dict[str, float], style) -> None:
    """Log-scale bars of trainable weights, one annotated count per bar.

    A model with no gradient training (ARIMA) keeps its slot, annotated the
    way draw_panel marks an unlogged column.
    """
    x = np.arange(len(models))
    vals = np.array([params.get(m, float("nan")) for m in models], dtype=float)
    finite = vals[np.isfinite(vals)]
    lo = 10.0 ** math.floor(math.log10(finite.min() * 0.9))
    hi = finite.max() * 2.2

    pm.style_axes(ax, style)
    # Bars read against horizontal rules, not the line panels' vertical ones.
    ax.grid(False)
    ax.grid(axis="y", color=pm.GRIDLINE, linewidth=style.line_width * 0.32, zorder=0)
    ax.set_axisbelow(True)
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.45, len(models) - 0.55)

    for xi, v in zip(x, vals):
        if np.isfinite(v):
            ax.bar(xi, v - lo, bottom=lo, width=0.62, color=BAR_COLOR, zorder=3)
            ax.annotate(
                human_count(v), (xi, v),
                textcoords="offset points", xytext=(0, 5),
                ha="center", va="bottom",
                fontsize=style.value_label, fontweight="bold",
                color=pm.INK_PRIMARY,
            )
        else:
            ax.annotate(
                "not gradient-trained", (xi, 0),
                xycoords=("data", "axes fraction"),
                textcoords="offset points", xytext=(0, 8),
                ha="center", rotation=90,
                fontsize=style.context * 0.85, color=pm.INK_MUTED,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=28, ha="right", fontsize=style.tick)
    ticks = [t for t in (1e2, 1e3, 1e4, 1e5, 1e6, 1e7) if lo <= t <= hi]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: count_tick(v)))
    ax.tick_params(axis="y", labelsize=style.tick)

    ax.text(
        0.5, 1.08, "Trainable parameters", transform=ax.transAxes,
        fontsize=style.panel_title, fontweight="bold", color=pm.INK_PRIMARY,
        ha="center", va="baseline",
    )


def complexity_figure(models, params, time_series, style, formats, outdir) -> list[Path]:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=style.combined_size)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.70, bottom=0.27, wspace=0.30)

    draw_params_panel(ax_a, models, params, style)

    panel = pm.Panel(
        "Training time", "Training time", True, models, time_series, [(336, 96)]
    )
    handles = pm.draw_panel(
        ax_b, panel, style,
        title="Training time", subtitle=None,
        highlight=None, muted=set(),
        show_legend=False, label_points=False,
    )

    figure_header(fig, style, "Multivariate: model complexity and training cost",
                  pm.grid_line([(336, 96)]))

    shared_legend(fig, pm.dot_handles(handles, style), style)

    pm.add_x_label(fig, [ax_a, ax_b], style, "Model")
    footnote(
        fig,
        "parameter counts: first logged value per model (channel counts differ across datasets)",
        style,
    )
    directory, base = artifact_path(outdir, "complexity", subject="multivariate",
                                    configs=[pm.parse_grid(GRID)],
                                    metric="params-vs-tt")
    directory.mkdir(parents=True, exist_ok=True)
    return pm.save(fig, directory, base, formats)


# --------------------------------------------------------------------------- #
# Figure 2: MAE against training time, bubble area = parameters
# --------------------------------------------------------------------------- #


def bubble_area(count: float, scale: float) -> float:
    return max(MIN_BUBBLE_AREA, count * scale)


# Axes-fraction x of the three size-key circles, tried in order. The top right
# was always empty while the slowest model was also a well-trained one; with
# its collapsed seeds averaged in, the Transformer sits there instead.
SIZE_KEY_SLOTS = [(0.58, 0.72, 0.885), (0.09, 0.23, 0.395)]


def size_key_x(fig, ax, labels, points, divisor, scale, base, width_pt, height_pt):
    """The first key slot that no bubble or point label overlaps."""
    to_axes = ax.transAxes.inverted()
    renderer = fig.canvas.get_renderer()
    taken = [to_axes.transform(t.get_window_extent(renderer)) for t in labels]
    for _, _, t, y, p in points:
        cx, cy = to_axes.transform(ax.transData.transform((t / divisor, y)))
        r = math.sqrt(bubble_area(p, scale) / math.pi)
        taken.append(np.array([[cx - r / width_pt, cy - r / height_pt],
                               [cx + r / width_pt, cy + r / height_pt]]))
    radii = [math.sqrt(bubble_area(c, scale) / math.pi) for c, _ in SIZE_KEY]
    for slot in SIZE_KEY_SLOTS:
        x0 = slot[0] - radii[0] / width_pt
        x1 = slot[-1] + radii[-1] / width_pt
        y0, y1 = base - 0.12, base + 2 * radii[-1] / height_pt
        if not any(b[0][0] < x1 and b[1][0] > x0 and b[0][1] < y1 and b[1][1] > y0
                   for b in taken):
            return slot
    return SIZE_KEY_SLOTS[0]


def bubble_figure(models, params, tvals, mvals, datasets, style, formats, outdir) -> list[Path]:
    # One point per model: both means run over the datasets, like the
    # trade-off view in plot_metrics.
    points, missing = [], []
    for i, model in enumerate(models):
        ts = [v for d in datasets if np.isfinite(v := cell(tvals, d, model))]
        ys = [v for d in datasets if np.isfinite(v := cell(mvals, d, model))]
        p = params.get(model, float("nan"))
        if ts and ys and np.isfinite(p):
            points.append((i, model, float(np.mean(ts)), float(np.mean(ys)), p))
        else:
            missing.append(model)

    divisor, unit = pm.time_unit(max(t for _, _, t, _, _ in points))
    xs = [t / divisor for _, _, t, _, _ in points]
    ys = [y for _, _, _, y, _ in points]
    scale = MAX_BUBBLE_AREA / max(p for _, _, _, _, p in points)

    fig, ax = plt.subplots(figsize=style.single_size)
    fig.subplots_adjust(left=0.155, right=0.94, top=0.76, bottom=0.2)

    # Log time axis with room on the right for the point labels; the y-range
    # opens downward so the largest bubble clears the x-axis.
    xlo, xhi = min(xs) / 1.5, max(xs) * 2.6
    xticks = pm.log_ticks(xlo, xhi)
    xdigits = 0 if all(float(t).is_integer() for t in xticks) else 1
    yspread = max(ys) - min(ys)
    ylo, yhi = min(ys) - 0.25 * yspread, max(ys) + 0.15 * yspread
    yticks = pm.linear_ticks(ylo, yhi)
    ydigits = pm.digits_for_step(yticks[1] - yticks[0]) if len(yticks) > 1 else 2

    # Larger bubbles go down first so smaller neighbours stay on top.
    order = sorted(points, key=lambda p: -bubble_area(p[4], scale))
    labels = []
    for rank, (i, name, t, y, p) in enumerate(order):
        color, second_cycle = pm.series_style(i)
        area = bubble_area(p, scale)
        radius_pt = math.sqrt(area / math.pi)
        ax.scatter(
            [t / divisor], [y], s=[area],
            facecolor=pm.SURFACE if second_cycle else color,
            edgecolor=color if second_cycle else pm.SURFACE,
            linewidth=1.6, zorder=4 + rank, clip_on=False,
        )
        labels.append(ax.annotate(
            name, (t / divisor, y),
            textcoords="offset points", xytext=(radius_pt + 5, 0),
            ha="left", va="center", fontsize=style.value_label,
            color=pm.label_color(color), fontweight="bold",
            zorder=5 + len(order), annotation_clip=False,
        ))

    pm.style_axes(ax, style)
    ax.set_xscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.{xdigits}f}"))
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.{ydigits}f}"))
    ax.set_ylabel("MAE", fontsize=style.axis_label, color=pm.INK_PRIMARY, labelpad=7)
    ax.tick_params(axis="both", labelsize=style.tick)

    if ylo > 0:
        pm.draw_axis_break(ax, style, below=(yticks[0] - ylo) / (yhi - ylo) if yticks else 1.0)

    if missing:
        ax.text(
            0.975, 0.035, f"{', '.join(missing)}: not logged", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=style.context * 0.9, color=pm.INK_MUTED,
        )

    pm.spread_labels(ax, labels, style.value_label * 1.45)

    # --- bubble-size key: bottom-aligned circles in an empty top corner ------ #
    pos = ax.get_position()
    width_pt = pos.width * fig.get_size_inches()[0] * 72.0
    height_pt = pos.height * fig.get_size_inches()[1] * 72.0
    base = 0.72
    key_x = size_key_x(fig, ax, labels, points, divisor, scale, base, width_pt, height_pt)
    for (count, label), xk in zip(SIZE_KEY, key_x):
        area = bubble_area(count, scale)
        radius_pt = math.sqrt(area / math.pi)
        ax.scatter(
            [xk], [base + radius_pt / height_pt], s=[area],
            transform=ax.transAxes, facecolor="none",
            edgecolor=pm.INK_MUTED, linewidth=1.2, zorder=3, clip_on=False,
        )
        ax.text(
            xk, base - 0.05, label, transform=ax.transAxes,
            ha="center", va="top", fontsize=style.context, color=pm.INK_SECONDARY,
        )

    ax.text(
        0.5, 1.22, "MAE against training time",
        transform=ax.transAxes, fontsize=style.panel_title, fontweight="bold",
        color=pm.INK_PRIMARY, ha="center", va="baseline",
    )
    # Two lines, so the header stays inside \columnwidth rather than stretching
    # the saved figure past the axes.
    for y, line in zip(
        (1.13, 1.05),
        (pm.grid_line([(336, 96)]), "bubble area = trainable parameters"),
    ):
        ax.text(
            0.5, y, line, transform=ax.transAxes, fontsize=style.subtitle,
            color=pm.INK_SECONDARY, ha="center", va="baseline",
        )

    pm.add_x_label(
        fig, [ax], style, f"Training time ({pm.TIME_UNIT_ABBREV.get(unit, unit)})"
    )
    fig.text(
        0.985, 0.012,
        "datasets averaged · smallest bubbles floored for visibility",
        ha="right", va="bottom", fontsize=style.footnote, color=pm.INK_MUTED,
    )
    directory, base = artifact_path(outdir, "complexity", subject="multivariate",
                                    configs=[pm.parse_grid(GRID)],
                                    metric="mae-vs-tt", view="size-params")
    directory.mkdir(parents=True, exist_ok=True)
    return pm.save(fig, directory, base, formats)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=Path, default=HERE / "frames" / "cells.csv")
    parser.add_argument("--outdir", type=Path, default=HERE / "figures")
    parser.add_argument("--style", choices=sorted(pm.STYLES), default="slide")
    parser.add_argument("--formats", default="png", help="comma-separated output formats")
    args = parser.parse_args()

    from emfres.records import COMPLEXITY_COLUMNS, records_from_cells
    records = records_from_cells(args.frames, HERE, metrics=COMPLEXITY_COLUMNS)
    models_full, datasets_full, mvals, _ = pm.aggregate(records, SECTION, "MAE", GRID)
    _, _, tvals, _ = pm.aggregate(records, SECTION, "Training time", GRID)
    params = params_from_records(records)
    if not models_full:
        parser.error(f"no {SECTION} L{GRID} cells found in {args.frames}")
    verify(params, tvals, mvals)
    print("values match the spot checks")

    models = [pm.short_model(m) for m in models_full]
    datasets = [pm.short_dataset(d) for d in datasets_full]
    time_series = [
        (pm.short_dataset(d), [tvals[d][m] for m in models_full]) for d in datasets_full
    ]

    style = pm.STYLES[args.style]
    plt.rcParams.update({**pm.rc_fonts(), "font.size": style.tick})

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    args.outdir.mkdir(parents=True, exist_ok=True)
    for path in complexity_figure(models, params, time_series, style, formats, args.outdir):
        print(f"wrote {path}")
    for path in bubble_figure(models, params, tvals, mvals, datasets, style, formats, args.outdir):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
