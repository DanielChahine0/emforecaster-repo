#!/usr/bin/env python3
"""Conformal-prediction trustworthiness figures from ``tables.md``.

Three panels over the Multivariate L = 336 / O = 96 grid, models along the
x-axis and one line per dataset (the ``dataset split`` convention):

* Marginal coverage % (``sl_test/ic``) - closer to the nominal 90% target is
  better, so the panel carries a dashed reference line at 90 instead of a
  "lower is better" axis note.
* Joint coverage % (``sl_test/jc``) - logged uncorrected (per-step alpha,
  ``conf.corrected = False``), so its values sit far below 90% by design;
  the footnote carries that caveat and the panel gets no target line.
* Mean interval width (``sl_test/interval_width_mean``) - narrower intervals
  at the same coverage are sharper, so this one IS lower-is-better.

The parsed cells are cross-checked against a handful of hard-coded values
straight from the tables before anything is drawn - a parser drift fails
loudly rather than plotting the wrong section.

Usage
-----
    python3 plot_trustworthiness.py                 # slide proportions, PNG
    python3 plot_trustworthiness.py --style paper   # IEEEtran column sizing
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe
from matplotlib.ticker import FixedLocator, FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_metrics as pm
from emfres.chrome import figure_header, footnote, shared_legend

HERE = Path(__file__).resolve().parent
SECTION = "Multivariate"
GRID = "336/96"
NOMINAL = 90.0  # target marginal coverage % (alpha = 0.1)

# Dataset lines run close in places (joint coverage's ED2022/EMF2023 braid,
# the near-tied interval widths at ARIMA), so each series is nudged sideways
# and haloed in the surface colour - near-ties land side by side instead of
# stacking, and the crossing line stays readable.
DODGE = 0.10  # per-series x offset, in category units
HALO = 2.4    # halo stroke width, as a multiple of the line width

# The three conformal subsections, in pm.METRICS shape: (key, heading pattern,
# long panel title, is_time).
TRUST_METRICS = [
    ("Coverage", re.compile(r"^3\.1\b.*Marginal coverage"), "Marginal coverage", False),
    ("Joint coverage", re.compile(r"^3\.2\b.*Joint coverage"), "Joint coverage", False),
    ("Interval width", re.compile(r"^3\.3\b.*Mean interval width"), "Mean interval width", False),
]

# Spot checks against the Multivariate tables, keyed (metric, dataset, model).
# If the parser drifts into another section's 3.1/3.2/3.3 these stop the run.
EXPECTED = {
    ("Coverage", "ED2022", "EMForecaster"): 84.75,
    ("Coverage", "ED2022", "DLinear"): 84.89,
    ("Coverage", "ED2022", "TSMixer"): 85.39,
    ("Coverage", "ED2022", "PatchTST"): 85.06,
    ("Coverage", "ED2022", "Transformer"): 85.45,
    ("Coverage", "ED2022", "LSTM"): 85.31,
    ("Coverage", "ED2022", "ARIMA"): 86.43,
    ("Coverage", "EMF2023", "EMForecaster"): 90.31,
    ("Coverage", "EMF2023", "DLinear"): 91.06,
    ("Joint coverage", "ED2022", "EMForecaster"): 11.47,
    ("Joint coverage", "ED2022", "ARIMA"): 21.65,
    ("Joint coverage", "EMF2023", "EMForecaster"): 10.67,
    ("Joint coverage", "EMF2023", "DLinear"): 13.81,
    ("Interval width", "ED2022", "EMForecaster"): 1.273,
    ("Interval width", "ED2022", "ARIMA"): 2.460,
    ("Interval width", "EMF2023", "EMForecaster"): 1.862,
    ("Interval width", "EMF2023", "DLinear"): 2.095,
}


def parse_trust_tables(tables: Path) -> list[pm.Record]:
    """Reuse pm.parse_tables with the conformal subsections swapped in."""
    saved = pm.METRICS
    pm.METRICS = TRUST_METRICS
    try:
        return pm.parse_tables(tables)
    finally:
        pm.METRICS = saved


def build_panel(records, key: str, long_title: str) -> pm.Panel:
    models, datasets, values, configs = pm.aggregate(records, SECTION, key, GRID)
    if not models:
        sys.exit(f"no {key!r} rows found in section {SECTION!r} at L/O {GRID}")
    x_labels = [pm.short_model(m) for m in models]
    series = [
        (pm.short_dataset(d), [values[d][m] for m in models]) for d in datasets
    ]
    # Verify the spot checks that touch this metric.
    for (metric, dataset, model), expected in EXPECTED.items():
        if metric != key:
            continue
        row = dict(zip(x_labels, dict(series)[dataset]))
        if not math.isclose(row[model], expected, abs_tol=1e-9):
            sys.exit(
                f"parser cross-check failed: {metric} / {dataset} / {model} "
                f"= {row[model]}, expected {expected}"
            )
    return pm.Panel(key, long_title, False, x_labels, series, configs)


def draw_trust_panel(
    ax,
    panel: pm.Panel,
    style: pm.Style,
    *,
    title: str,
    subtitle: str | None,
    ylim: tuple[float, float] | None = None,
    tick_target: int = 6,
    ref_line: bool = False,
    show_legend: bool,
):
    """pm.draw_panel's look, with a fixed-scale option and the 90% target line.

    The stock panel fits the y-scale to the data; coverage instead pins an
    80-95 window so the nominal line stays inside the frame.
    """
    x = np.arange(len(panel.x_labels))
    finite = [v for _, vals in panel.series for v in vals if np.isfinite(v)]

    if ylim is not None:
        lo, hi = ylim
    else:
        pad = 0.09 * (max(finite) - min(finite))
        lo, hi = min(finite) - pad, max(finite) + pad
        if show_legend:  # empty band at the top for the legend to sit in
            hi += 0.28 * (hi - lo)
    ticks = pm.linear_ticks(lo, hi, target=tick_target)
    digits = pm.digits_for_step(ticks[1] - ticks[0]) if len(ticks) > 1 else 2
    if ticks and all(float(t).is_integer() for t in ticks):
        digits = 0
    fmt = lambda v: f"{v:.{digits}f}"

    # --- the nominal-coverage reference, under the data lines ----------------- #
    avoid = []
    if ref_line:
        ax.axhline(
            NOMINAL, color=pm.INK_MUTED, linestyle=(0, (6, 4)),
            linewidth=style.line_width * 0.75, zorder=3,
        )
        avoid.append(ax.annotate(
            "90% target",
            xy=(0.03, NOMINAL), xycoords=("axes fraction", "data"),
            xytext=(0, -6), textcoords="offset points",
            ha="left", va="top", fontsize=style.context,
            fontstyle="italic", color=pm.INK_SECONDARY, zorder=3.5,
        ))

    # --- series, in the house dot-line style ---------------------------------- #
    handles = []
    occupied: list[tuple[float, float]] = []
    n_series = len(panel.series)
    for i, (name, vals) in enumerate(panel.series):
        y = np.array(vals, dtype=float)
        xs = x + (i - (n_series - 1) / 2.0) * DODGE
        color, second_cycle = pm.series_style(i)
        # Line and markers as separate artists: a halo on a marker-bearing
        # line erases the segment around every dot and fakes a dash pattern,
        # so the line alone gets the halo and the dots stay plain.
        # Dots ride above every line so a crossing never buries a marker.
        (line,) = ax.plot(
            xs, y,
            color=color,
            linestyle=(0, (2.4, 1.4)) if second_cycle else "-",
            linewidth=style.line_width,
            solid_capstyle="round",
            solid_joinstyle="round",
            dash_capstyle="round",
            zorder=4 + i,
            clip_on=False,
        )
        line.set_path_effects([
            pe.Stroke(linewidth=style.line_width * HALO, foreground=pm.SURFACE),
            pe.Normal(),
        ])
        (dots,) = ax.plot(
            xs, y,
            linestyle="none",
            marker="o",
            markersize=style.marker,
            markerfacecolor=pm.SURFACE if second_cycle else color,
            markeredgecolor=color if second_cycle else "none",
            markeredgewidth=style.line_width * 0.9 if second_cycle else 0.0,
            color=color,
            label=name,
            zorder=10 + i,
            clip_on=False,
        )
        handles.append(dots)
        occupied += pm.densify(list(zip(xs.tolist(), y.tolist())))

    # --- frame ----------------------------------------------------------------- #
    pm.style_axes(ax, style)
    n = len(panel.x_labels)
    ax.set_xlim(-0.45, n - 0.55)
    ax.set_ylim(lo, hi)
    ax.set_xticks(x)
    ax.set_xticklabels(panel.x_labels, rotation=28, ha="right", fontsize=style.tick)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v)))
    ax.tick_params(axis="y", labelsize=style.tick)
    if lo > 0:
        pm.draw_axis_break(ax, style, below=(ticks[0] - lo) / (hi - lo) if ticks else 1.0)

    outside = show_legend and pm.place_legend(ax, handles, style, occupied, avoid)
    lift = 0.15 if outside else 0.0
    ax.text(
        0.5, (1.16 if subtitle else 1.08) + lift, title, transform=ax.transAxes,
        fontsize=style.panel_title, fontweight="bold", color=pm.INK_PRIMARY,
        ha="center", va="baseline",
    )
    if subtitle:
        ax.text(
            0.5, 1.06 + lift, subtitle, transform=ax.transAxes,
            fontsize=style.subtitle, color=pm.INK_SECONDARY,
            ha="center", va="baseline",
        )
    return handles


PANEL_SPECS = [
    # (stem slug, metric key, panel title, fixed ylim, tick target, ref line)
    (
        "multivariate_coverage", "Coverage", "Marginal coverage (%)",
        (80.0, 95.0), 4, True,
    ),
    (
        "multivariate_joint-coverage", "Joint coverage", "Joint coverage (%)",
        None, 6, False,
    ),
    (
        "multivariate_interval-width", "Interval width", "Mean interval width",
        None, 6, False,
    ),
]

FOOTNOTE = "split-conformal, alpha = 0.1 · joint coverage is per-step (uncorrected), low values by design"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=Path, default=HERE / "tables.md")
    parser.add_argument("--style", choices=sorted(pm.STYLES), default="slide")
    parser.add_argument("--format", action="append", dest="formats",
                        help="png (default), pdf, svg; repeatable")
    args = parser.parse_args()

    style = pm.STYLES[args.style]
    formats = args.formats or ["png"]

    plt.rcParams.update({**pm.rc_fonts(), "font.size": style.tick})

    records = parse_trust_tables(args.tables)
    panels = {
        key: build_panel(records, key, long_title)
        for _, key, long_title, _, _, _ in PANEL_SPECS
    }

    outdir = HERE / "figures" / "trustworthiness"
    outdir.mkdir(parents=True, exist_ok=True)

    subhead = pm.grid_line(panels["Coverage"].configs)

    # --- 1) Combined strip: marginal | joint | interval width, shared legend -- #
    fig, axes = plt.subplots(1, len(PANEL_SPECS), figsize=style.combined_size)
    axes = list(np.atleast_1d(axes))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.66, bottom=0.27, wspace=0.26)
    handles = []
    for ax, (_, key, long_title, ylim, target, ref) in zip(axes, PANEL_SPECS):
        handles = draw_trust_panel(
            ax, panels[key], style,
            title=long_title, subtitle=None,
            ylim=ylim, tick_target=target, ref_line=ref, show_legend=False,
        )
    figure_header(fig, style, "Multivariate: conformal prediction trustworthiness",
                  subhead, top=0.985, gap=1.5)
    shared_legend(fig, pm.dot_handles(handles, style), style)
    pm.add_x_label(fig, axes, style, "Model")
    footnote(fig, FOOTNOTE, style, y=0.008)
    for path in pm.save(fig, outdir, "multivariate_trustworthiness", formats):
        print(f"  wrote {path}")

    # --- 2) One standalone figure per panel ----------------------------------- #
    for stem, key, long_title, ylim, target, ref in PANEL_SPECS:
        fig, ax = plt.subplots(figsize=style.single_size)
        fig.subplots_adjust(left=0.125, right=0.965, top=0.84, bottom=0.26)
        draw_trust_panel(
            ax, panels[key], style,
            title=long_title, subtitle=subhead,
            ylim=ylim, tick_target=target, ref_line=ref, show_legend=True,
        )
        pm.add_x_label(fig, [ax], style, "Model")
        for path in pm.save(fig, outdir, stem, formats):
            print(f"  wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
