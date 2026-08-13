#!/usr/bin/env python3
"""Plot the ED2022 lookback ladder: MSE / MAE / training time against L.

One strip of three panels with the lookback length L on the x-axis and one
line per (model, setup) pair: the multivariate runs solid with filled markers,
the per-channel runs dashed with hollow ones, colour carrying the model in
both. Unlike plot_metrics.py this reads the logs directly - the ladder's
off-grid windows (L = 48..480 by 48) never land in tables.md.

Per-channel accuracy comes from each cell's aggregate.json (channel-averaged,
the same convention as the multivariate test MSE/MAE). Per-channel training
time is the SUM of the cell's 27 band trainings - the cost of the approach,
not its wall-clock, since the bands run through an 8-wide pool.

Usage
-----
    python3 plot_L_ladder.py                 # slide proportions, PNG
    python3 plot_L_ladder.py --style paper   # IEEEtran column sizing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import plot_metrics as pm
from emfres.chrome import figure_header, footnote, shared_legend
from emfres.vocab import parse_duration

REPO = Path(__file__).resolve().parent.parent
LOOKBACKS = list(range(48, 481, 48))
HORIZON = 96
SEED = 69421
DATASET = "ed2022"
MODELS = ["DLinear", "PatchTST", "TSMixer"]

MV_ROOT = REPO / "logs" / "multivariate"
PC_ROOT = REPO / "logs" / "univariate (per channel)"
CL_ROOT = REPO / "logs" / "cluster"
K_CLUSTER = 5

def running_seconds(text: str | None) -> float:
    """Seconds from the logger's ``43min 10s`` / ``1h 5min 2s`` / ``22s`` form.

    Delegates to the shared grammar. The local copy this replaces matched
    ``h`` only on a word boundary, so it silently dropped the hours from a
    ``1hr 44min 16s`` value and under-reported that run by an hour.
    """
    seconds = parse_duration(text)
    return float("nan") if seconds is None else seconds


def scalar(value) -> float:
    """The logger stores test metrics as one-element lists."""
    if value is None:
        return float("nan")
    if isinstance(value, list):
        return float(value[0]) if value else float("nan")
    return float(value)


def mv_cell(model: str, lookback: int) -> tuple[float, float, float]:
    """(mse, mae, train seconds) of one multivariate cell, NaN where unlogged."""
    path = MV_ROOT / model / DATASET / f"L{lookback}_O{HORIZON}_cp" / str(SEED) / "log.json"
    if not path.exists():
        return (float("nan"),) * 3
    log = json.loads(path.read_text())
    return (
        scalar(log.get("sl_test/mse")),
        scalar(log.get("sl_test/mae")),
        running_seconds(log.get("parameters/running_time")),
    )


def cluster_cell(model: str, lookback: int) -> tuple[float, float, float]:
    """(mse, mae, train seconds) of one CCM cluster cell at k = K_CLUSTER."""
    path = (CL_ROOT / model / DATASET / f"L{lookback}_O{HORIZON}_cp"
            / f"k{K_CLUSTER}" / str(SEED) / "log.json")
    if not path.exists():
        return (float("nan"),) * 3
    log = json.loads(path.read_text())
    return (
        scalar(log.get("sl_test/mse")),
        scalar(log.get("sl_test/mae")),
        running_seconds(log.get("parameters/running_time")),
    )


def pc_cell(model: str, lookback: int) -> tuple[float, float, float]:
    """(mse, mae, summed band-train seconds) of one per-channel cell."""
    cell = PC_ROOT / model / DATASET / f"L{lookback}_O{HORIZON}_cp"
    agg_path = cell / "aggregate.json"
    if not agg_path.exists():
        return (float("nan"),) * 3
    agg = json.loads(agg_path.read_text())
    if not agg.get("complete"):
        print(f"  [warn] {cell} incomplete: {agg.get('missing_channels')}", file=sys.stderr)
    seconds = 0.0
    for band_log in sorted(cell.glob(f"ch*/{SEED}/log.json")):
        seconds += running_seconds(
            json.loads(band_log.read_text()).get("parameters/running_time")
        )
    return (
        float(agg.get("mse_mean", "nan")),
        float(agg.get("mae_mean", "nan")),
        seconds if seconds > 0 else float("nan"),
    )


def ladder_style(i: int) -> tuple[str, bool, str]:
    """Colour follows the model, the dashed second cycle marks the per-channel setup."""
    return pm.SERIES_COLORS[i % len(MODELS)], i >= len(MODELS), "o"


def build_panels() -> list[pm.Panel]:
    series: list[tuple[str, list[tuple[float, float, float]]]] = []
    for model in MODELS:
        series.append((f"{model} · MV", [mv_cell(model, L) for L in LOOKBACKS]))
    for model in MODELS:
        series.append((f"{model} · per ch.", [pc_cell(model, L) for L in LOOKBACKS]))

    configs = [(L, HORIZON) for L in LOOKBACKS]
    x_labels = [str(L) for L in LOOKBACKS]
    panels = []
    for slot, (key, is_time) in enumerate(
        [("MSE", False), ("MAE", False), ("Training time", True)]
    ):
        panels.append(pm.Panel(
            key, key, is_time, x_labels,
            [(name, [cells[slot] for cells in values]) for name, values in series],
            configs,
        ))
    return panels


DOTTED = (0, (1.0, 1.6))


def slim_markers(handles, style: pm.Style) -> None:
    """Thin the hollow rings to suit the ladder's small markers.

    draw_panel sizes the ring stroke for plot_metrics' large markers; at the
    ladder's point size that stroke would fill the circle solid.
    """
    for h in handles:
        if h.get_markeredgecolor() not in ("none", None):
            h.set_markeredgewidth(style.line_width * 0.5)


def model_panels(model: str) -> list[pm.Panel]:
    """The three metric panels of one model: series = its three setups."""
    cells = {
        "multivariate": [mv_cell(model, L) for L in LOOKBACKS],
        "per-channel": [pc_cell(model, L) for L in LOOKBACKS],
        f"cluster (k = {K_CLUSTER})": [cluster_cell(model, L) for L in LOOKBACKS],
    }
    configs = [(L, HORIZON) for L in LOOKBACKS]
    x_labels = [str(L) for L in LOOKBACKS]
    return [
        pm.Panel(
            key, key, is_time, x_labels,
            [(name, [c[slot] for c in values]) for name, values in cells.items()],
            configs,
        )
        for slot, (key, is_time) in enumerate(
            [("MSE", False), ("MAE", False), ("Training time", True)]
        )
    ]


def by_model_figure(style: pm.Style, formats: list[str], outdir: Path) -> None:
    """One 3x3 sheet: a row per model, its three setups overlaid per panel.

    Colour still follows the model - the whole row wears its series colour,
    matching the combined strip; the setup rides on line style and marker:
    solid/filled circle = multivariate, dashed/hollow circle = per-channel,
    dotted/hollow square = cluster. Metric titles run across the top row only;
    the rotated row label carries the model, so identity is never colour-alone,
    and one neutral-ink legend up top keys the setups for every row at once.
    """
    width, row_height = style.combined_size
    fig, axes = plt.subplots(len(MODELS), 3,
                             figsize=(width, row_height * 0.62 * len(MODELS) + 1.5))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.11,
                        wspace=0.30, hspace=0.44)

    for row, model in enumerate(MODELS):
        colour = pm.SERIES_COLORS[row]
        # One hue per row, the second cycle marking both non-MV setups; the
        # cluster line is then restyled dotted-and-square below.
        row_style = lambda i, c=colour: (c, i >= 1, "o")
        for col, panel in enumerate(model_panels(model)):
            handles = pm.draw_panel(
                axes[row][col], panel, style,
                title=panel.title if row == 0 else "",
                subtitle=None, highlight=None, muted=set(),
                show_legend=False, label_points=False, styler=row_style,
            )
            handles[2].set_linestyle(DOTTED)
            handles[2].set_marker("s")
            handles[2].set_markersize(style.marker * 0.88)
            slim_markers(handles, style)
        pos = axes[row][0].get_position()
        fig.text(
            0.016, (pos.y0 + pos.y1) / 2, model,
            rotation=90, ha="center", va="center",
            fontsize=style.panel_title, fontweight="bold", color=pm.INK_PRIMARY,
        )

    fig.text(0.5, 0.988, "ED2022: each model's lookback ladder, all setups",
             ha="center", va="top", fontsize=style.title, fontweight="bold",
             color=pm.INK_PRIMARY)
    fig.text(0.5, 0.988 - (style.title * 1.5) / 72 / fig.get_size_inches()[1],
             f"forecast horizon {HORIZON} · seed {SEED}",
             ha="center", va="top", fontsize=style.subtitle, color=pm.INK_SECONDARY)

    # Hue belongs to the rows, so the setup key is neutral ink: the line style
    # and marker are the encoding, and they read the same in every row.
    def key_handle(label, linestyle, marker, filled):
        return Line2D(
            [], [], color=pm.INK_PRIMARY, linestyle=linestyle,
            linewidth=style.line_width * 0.9, marker=marker,
            markersize=style.legend * 0.85,
            markerfacecolor=pm.INK_PRIMARY if filled else pm.SURFACE,
            markeredgecolor="none" if filled else pm.INK_PRIMARY,
            markeredgewidth=0.0 if filled else style.line_width * 0.8,
            label=label,
        )

    shared_legend(
        fig,
        [
            key_handle("multivariate", "-", "o", True),
            key_handle("per-channel", (0, (2.4, 1.4)), "o", False),
            key_handle(f"cluster (k = {K_CLUSTER})", DOTTED, "s", False),
        ],
        style, anchor=(0.5, 0.94), handlelength=2.6, handletextpad=0.8,
    )
    pm.add_x_label(fig, list(axes[-1]), style, "Lookback length")
    footnote(
        fig,
        "per-channel time = sum of the 27 band trainings · "
        "the lookback-336 cells were timed under different GPU load",
        style, y=0.008,
    )
    for path in pm.save(fig, outdir, f"{DATASET}_O{HORIZON}_L_ladder_by_model", formats):
        print(f"  wrote {path.relative_to(REPO)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--style", choices=sorted(pm.STYLES), default="slide")
    parser.add_argument("--format", action="append", dest="formats",
                        help="png (default), pdf, svg; repeatable")
    args = parser.parse_args()
    # Ten points per line, up to three lines per hue: the ladder wants far
    # smaller markers than plot_metrics' five-point categorical panels.
    style = replace(pm.STYLES[args.style], marker=pm.STYLES[args.style].marker * 0.5)
    formats = args.formats or ["png"]

    plt.rcParams.update({**pm.rc_fonts(), "font.size": style.tick})

    # The ladder pairs each model's two setups instead of cycling five colours,
    # so it hands draw_panel its own styler rather than the default wheel.
    panels = build_panels()
    fig, axes = plt.subplots(1, len(panels), figsize=style.combined_size)
    axes = list(np.atleast_1d(axes))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.70, bottom=0.21, wspace=0.30)
    handles = []
    for ax, panel in zip(axes, panels):
        handles = pm.draw_panel(
            ax, panel, style,
            title=panel.title, subtitle=None,
            highlight=None, muted=set(),
            show_legend=False, label_points=False, styler=ladder_style,
        )
        slim_markers(handles, style)

    figure_header(
        fig, style,
        "ED2022: accuracy and training cost by lookback length",
        f"forecast horizon {HORIZON} · seed {SEED} · "
        f"solid = multivariate, dashed = per-channel",
    )
    # The subtitle keys the setups, so the legend needs only one dot per model:
    # the solid multivariate lines, drawn first, carry the hue.
    for handle, model in zip(handles, MODELS):
        handle.set_label(model)
    shared_legend(fig, pm.dot_handles(handles[: len(MODELS)], style), style,
                  ncol=len(MODELS))
    pm.add_x_label(fig, axes, style, "Lookback length")
    footnote(
        fig,
        "per-channel time = sum of the 27 band trainings · "
        "the lookback-336 cells were timed under different GPU load",
        style,
    )

    outdir = REPO / "results" / "figures" / "L ladder"
    outdir.mkdir(parents=True, exist_ok=True)
    for path in pm.save(fig, outdir, f"{DATASET}_O{HORIZON}_L_ladder", formats):
        print(f"  wrote {path.relative_to(REPO)}")

    by_model_figure(style, formats, outdir)


if __name__ == "__main__":
    main()
