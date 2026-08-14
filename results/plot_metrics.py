#!/usr/bin/env python3
"""Turn ``frames/cells.csv`` into MSE / MAE / training-time figures.

Each figure puts the models along the x-axis (ordered by year of publication)
and one line per dataset, so a panel reads as "how this metric moves across
model generations, per dataset". Three panels - MSE, MAE, total training time -
are emitted both as one combined strip and as standalone single figures, twice:
``figures/by-dataset/`` keeps datasets as the lines with models along the
x-axis, and ``figures/by-model/`` flips the roles. Each also gets an
``figures/overlay/`` panel putting both setups' MAE together (multivariate
solid over per-channel dashed). The clustering study puts the cluster count k
on the x-axis and lands in ``figures/cluster-sweep/``.

Paths are generated, never typed: emfres.naming builds every one of them from
the slice actually plotted, so the window is always in the filename. A pooled
run writes ``l48-480x48_o96_mse.png`` where ``--grid 336/96`` writes
``l336_o96_mse.png``, side by side in the same directory. This replaces a
hand-typed ``--outdir``, under which the two were indistinguishable.

Usage
-----
    python3 plot_metrics.py                          # every section, PNG
    python3 plot_metrics.py --section multivariate
    python3 plot_metrics.py --grid 336/96            # keep one (L, O) config
    python3 plot_metrics.py --dataset ED2022         # keep one dataset (repeatable)
    python3 plot_metrics.py --mute DET               # gray one dataset as reference
    python3 plot_metrics.py --style paper            # IEEEtran column sizing
    python3 plot_metrics.py --source tables          # legacy: read tables.md
    python3 plot_metrics.py --list                   # show what was parsed

frames/cells.csv is the single source of truth; rebuild it with
``python results/make.py extract``.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter


# --------------------------------------------------------------------------- #
# Shared library
# --------------------------------------------------------------------------- #
#
# Design tokens, figure furniture, and the drawing primitives now live in
# ``emfres``. They are re-exported here by name because plot_complexity,
# plot_trustworthiness, and plot_L_ladder all reach for them as ``pm.<name>``.

from emfres.style import (  # noqa: E402
    AXIS_RULE, FONT_STACKS, GRIDLINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY,
    LADDER, LEGEND_EDGE, LEGEND_FILL, MUTED_COLOR, SERIES_COLORS, STYLES,
    SURFACE, TIME_UNIT_ABBREV, Style, digits_for_step, label_color,
    linear_ticks, log_ticks, rc_fonts, series_style, time_unit,
)
from emfres.chrome import (  # noqa: E402
    add_legend, add_x_label, axes_bottom, densify, dot_handles, draw_axis_break,
    figure_header, footnote, frame_legend, grid_line, place_legend, save,
    shared_legend, spread_labels, style_axes,
)
from emfres.panels import (  # noqa: E402
    Panel, draw_panel, draw_tradeoff_panel,
)
from emfres.naming import artifact_path, STEM_SLUGS  # noqa: E402



# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

from emfres.vocab import is_duration, parse_duration  # noqa: E402


def clean_cell(cell: str) -> str:
    """Strip the markdown emphasis the tables use to mark best / runner-up."""
    text = cell.strip()
    text = re.sub(r"</?u>", "", text)
    text = text.replace("**", "").replace("`", "").replace("$", "")
    return text.strip()


def parse_value(text: str) -> float:
    """Parse a plain number or a ``1h 44m 16s`` duration into a float.

    Durations come back in seconds. ``-`` (not logged) becomes NaN.
    """
    text = clean_cell(text)
    if not text or text in {"-", "--", "n/a", "N/A"}:
        return float("nan")
    if is_duration(text):
        return parse_duration(text)
    try:
        return float(text.replace(",", "").rstrip("%"))
    except ValueError:
        return float("nan")


def split_row(line: str) -> list[str]:
    return [clean_cell(c) for c in line.strip().strip("|").split("|")]


@dataclass
class Record:
    section: str
    metric: str
    dataset: str
    lookback: int
    horizon: int
    model: str
    model_order: tuple[int, int]
    value: float
    cluster: int | None = None  # k, in the k-grouped-channels tables only


# Which subsection heading feeds which panel, its panel title, and whether it
# is a duration (log axis, time-formatted ticks).
METRICS = [
    ("MSE", re.compile(r"^1\.1\b.*\bMSE\b"), "MSE", False),
    ("MAE", re.compile(r"^1\.2\b.*\bMAE\b"), "MAE", False),
    ("Training time", re.compile(r"^2\.1\b.*Total training time"), "Training time", True),
]

# Fixed x-axis order: EMForecaster first, then the baselines. Models missing
# from this list (new baselines) fall in after, in table-column order.
MODEL_ORDER = [
    "EMForecaster", "PatchTST", "DLinear", "TSMixer", "Transformer", "LSTM", "ARIMA",
]


def model_rank(name: str) -> int:
    try:
        return MODEL_ORDER.index(re.sub(r"\s*\(\d{4}\)\s*$", "", name).strip())
    except ValueError:
        return len(MODEL_ORDER)


# Sections whose tables share the Dataset | L | O | <models...> shape.
SECTION_ALIASES = {
    "univariate": "Univariate Series",
    "perchannel": "Univariate (Per Channel)",
    "multivariate": "Multivariate",
}

# The k-grouped-channels study: the same shape plus a k column, so its figures
# put the cluster count on the x-axis instead of the model generation.
CLUSTER_SECTION = "Cluster (k-Grouped Channels)"

# The clustering study reads on one dataset: its figures default to ED2022
# (matching the sweep default in forecasting/grids.py), and --dataset overrides.
CLUSTER_DEFAULT_DATASET = "ED2022"


def parse_tables(path: Path) -> list[Record]:
    """Walk ``tables.md`` and pull out every (section, metric, dataset, model) cell."""
    records: list[Record] = []
    section = ""
    header: list[str] | None = None
    metric: str | None = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()

        if line.startswith("# "):
            section, header, metric = clean_cell(line[2:]), None, None
            continue
        if line.startswith("### "):
            subsection = clean_cell(line[4:]).replace("·", "").replace("  ", " ")
            header = None
            metric = next(
                (name for name, pat, _, _ in METRICS if pat.match(subsection)), None
            )
            continue
        if line.startswith("## "):
            header, metric = None, None
            continue

        if not line.startswith("|") or metric is None:
            if not line.startswith("|"):
                header = None
            continue

        cells = split_row(line)
        if set("".join(cells)) <= set("-: "):  # the |---|---| separator row
            continue
        if header is None:
            header = cells
            continue

        # Only the wide "Dataset | L | O | <one column per model>" tables, plus
        # an optional k (cluster count) column before the models. The
        # comparison tree uses "Dataset | L | O | Model | ..." instead - skip it.
        if len(header) < 4 or header[0] != "Dataset" or header[1:3] != ["L", "O"]:
            continue
        if header[3] == "Model":
            continue
        has_k = header[3] == "k"
        if has_k and len(header) < 5:
            continue

        dataset, lookback, horizon = cells[0], int(cells[1]), int(cells[2])
        cluster = int(cells[3]) if has_k else None
        first_model = 4 if has_k else 3
        for col, (model, cell) in enumerate(
            zip(header[first_model:], cells[first_model:])
        ):
            records.append(
                Record(
                    section=section,
                    metric=metric,
                    dataset=dataset,
                    lookback=lookback,
                    horizon=horizon,
                    model=model,
                    model_order=(model_rank(model), col),
                    value=parse_value(cell),
                    cluster=cluster,
                )
            )

    return records


# --------------------------------------------------------------------------- #
# Shaping
# --------------------------------------------------------------------------- #


def short_model(name: str) -> str:
    """``EMForecaster (2025)`` -> ``EMForecaster``; years move to the caption."""
    return re.sub(r"\s*\(\d{4}\)\s*$", "", name).strip()


def short_dataset(name: str) -> str:
    """``EMF2023 (University-5 w/ PTV)`` -> ``EMF2023``."""
    return re.sub(r"\s*\(.*\)\s*$", "", name).strip() or name


def model_year(name: str) -> str | None:
    found = re.search(r"\((\d{4})\)", name)
    return found.group(1) if found else None


def pair_label(dataset: str, model: str, many_datasets: bool, many_models: bool) -> str:
    """One legend entry per (dataset, model) line; a constant half drops out."""
    parts = []
    if many_datasets:
        parts.append(short_dataset(dataset))
    if many_models or not many_datasets:
        parts.append(short_model(model))
    return " · ".join(parts)



_GRID_RE = re.compile(r"^\s*L?(\d+)\s*[/x]\s*O?(\d+)\s*$", re.IGNORECASE)


def parse_grid(grid: str) -> tuple[int, int]:
    """(lookback, horizon) from ``336/96``, ``L336/O96`` or ``336x96``.

    The tables and the log directories both spell this window with L/O
    prefixes, so a reader who copies one in should not get a traceback.
    """
    m = _GRID_RE.match(grid)
    if not m:
        raise ValueError(
            f"cannot read --grid {grid!r}; expected something like 336/96 or L336/O96"
        )
    return int(m.group(1)), int(m.group(2))


def aggregate(records, section: str, metric: str, grid: str | None):
    """Mean each (dataset, model) cell over the (L, O) configs still in play."""
    rows = [r for r in records if r.section == section and r.metric == metric]
    if grid:
        lookback, horizon = parse_grid(grid)
        rows = [r for r in rows if r.lookback == lookback and r.horizon == horizon]
    if not rows:
        return [], [], {}, []

    models = [m for _, m in sorted({(r.model_order, r.model) for r in rows})]
    datasets = list(dict.fromkeys(r.dataset for r in rows))  # table order
    configs = sorted({(r.lookback, r.horizon) for r in rows})

    values: dict[str, dict[str, float]] = {}
    for dataset in datasets:
        values[dataset] = {}
        for model in models:
            cells = [
                r.value
                for r in rows
                if r.dataset == dataset and r.model == model and not np.isnan(r.value)
            ]
            values[dataset][model] = float(np.mean(cells)) if cells else float("nan")
    return models, datasets, values, configs


def aggregate_clusters(records, section: str, metric: str, grid: str | None):
    """Mean each (dataset, model, k) cell over the (L, O) configs still in play."""
    rows = [
        r
        for r in records
        if r.section == section and r.metric == metric and r.cluster is not None
    ]
    if grid:
        lookback, horizon = parse_grid(grid)
        rows = [r for r in rows if r.lookback == lookback and r.horizon == horizon]
    if not rows:
        return [], [], {}, []

    ks = sorted({r.cluster for r in rows})
    models = [m for _, m in sorted({(r.model_order, r.model) for r in rows})]
    datasets = list(dict.fromkeys(r.dataset for r in rows))  # table order
    pairs = [(d, m) for d in datasets for m in models]
    configs = sorted({(r.lookback, r.horizon) for r in rows})

    values: dict[tuple[str, str], dict[int, float]] = {}
    for pair in pairs:
        dataset, model = pair
        values[pair] = {}
        for k in ks:
            cells = [
                r.value
                for r in rows
                if r.dataset == dataset
                and r.model == model
                and r.cluster == k
                and not np.isnan(r.value)
            ]
            values[pair][k] = float(np.mean(cells)) if cells else float("nan")
    return ks, pairs, values, configs












def draw_no_cluster_baselines(ax, panel: Panel, base_vals, style: Style) -> None:
    """Dotted per-model reference lines: the plain multivariate backbone.

    The cluster panels sweep k inside the CCM architecture, so on their own
    they cannot say whether clustering beats not clustering - CCM's k = 1 is
    still CCM, not the ordinary model. Each series gets a dotted line in its
    own hue at the plain (no-CCM) multivariate value for the same dataset,
    windows and metric; time panels re-derive draw_panel's unit divisor so
    the line lands on the same scale.
    """
    finite = [v for _, vals in panel.series for v in vals if np.isfinite(v)]
    divisor = time_unit(max(finite))[0] if panel.is_time else 1.0
    lo, hi = ax.get_ylim()
    pad = (hi - lo) * 0.04
    for i, value in enumerate(base_vals):
        if value is None or not np.isfinite(value):
            continue
        value = value / divisor
        color, _ = series_style(i)
        ax.axhline(value, color=color, linestyle=(0, (1.4, 1.8)),
                   linewidth=style.line_width * 0.85, zorder=3)
        if panel.is_time:  # log axis: extend multiplicatively
            lo, hi = min(lo, value / 1.15), max(hi, value * 1.15)
        else:
            lo, hi = min(lo, value - pad), max(hi, value + pad)
    ax.set_ylim(lo, hi)


def build_figures(
    records,
    section: str,
    *,
    outdir: Path,
    grid: str | None,
    formats: list[str],
    style: Style,
    highlight: str | None,
    muted: set[str],
    label_points: bool,
    family: str,
    group: str | None,
    split: str,
    subject: str | None = None,
    dataset_token: str | None = None,
    k_token=None,
    title_text: str | None,
    subtitle_text: str | None,
) -> list[Path]:
    panels: list[Panel] = []
    context = None  # the sole dataset of a cluster study, surfaced in subtitles
    baselines: dict[str, list] = {}  # panel key -> no-clustering value per series
    for key, _, long_title, is_time in METRICS:
        if split == "cluster":  # lines = (dataset, model) pairs, columns = k
            ks, pairs, kvalues, configs = aggregate_clusters(records, section, key, grid)
            if not ks:
                continue
            many_datasets = len({d for d, _ in pairs}) > 1
            many_models = len({m for _, m in pairs}) > 1
            if not many_datasets:
                context = short_dataset(pairs[0][0])
            x_labels = [str(k) for k in ks]
            series = [
                (
                    pair_label(d, m, many_datasets, many_models),
                    [kvalues[(d, m)][k] for k in ks],
                )
                for d, m in pairs
            ]
            # The plain multivariate cell beside each (dataset, model) pair,
            # at the same windows: the no-clustering reference the k sweep is
            # judged against (drawn dotted by draw_no_cluster_baselines).
            base_rows = [r for r in records if (r.lookback, r.horizon) in set(configs)]
            _, _, base_map, _ = aggregate(base_rows, OVERLAY_MV, key, grid)
            baselines[key] = [base_map.get(d, {}).get(m) for d, m in pairs]
            panels.append(Panel(key, long_title, is_time, x_labels, series, configs))
            continue
        models, datasets, values, configs = aggregate(records, section, key, grid)
        if not models:
            continue
        if split == "model split":  # lines = models, columns = datasets
            x_labels = [short_dataset(d) for d in datasets]
            series = [
                (short_model(m), [values[d][m] for d in datasets]) for m in models
            ]
        else:  # dataset split: lines = datasets, columns = models
            x_labels = [short_model(m) for m in models]
            series = [
                (short_dataset(d), [values[d][m] for m in models]) for d in datasets
            ]
        panels.append(Panel(key, long_title, is_time, x_labels, series, configs))

    if not panels:
        print(f"  no data for section {section!r}", file=sys.stderr)
        return []

    def sub(configs: list[tuple[int, int]]) -> str:
        line = grid_line(configs)
        if context:
            line = f"{context} · {line}"
        if any(v is not None for vals in baselines.values() for v in vals):
            line += " · dotted = no clustering"
        return line

    first = panels[0]
    written: list[Path] = []
    x_title = {"model split": "Dataset", "cluster": "Channel clusters (k)"}.get(
        split, "Model"
    )
    axis = "cluster count" if split == "cluster" else "model generation"
    headline = title_text or f"{section}: accuracy and training cost by {axis}"
    subhead = subtitle_text or sub(first.configs)

    # --- 1) Combined strip: MSE | MAE | training time, one shared legend ------ #
    legend_rows = int(np.ceil(len(first.series) / 4))
    fig, axes = plt.subplots(1, len(panels), figsize=style.combined_size)
    axes = list(np.atleast_1d(axes))
    fig.subplots_adjust(
        left=0.055, right=0.985,
        top=0.70 if legend_rows == 1 else 0.63,
        bottom=0.27, wspace=0.30,
    )
    handles: list[Line2D] = []
    for ax, panel in zip(axes, panels):
        handles = draw_panel(
            ax, panel, style,
            title=panel.title,
            subtitle=None,
            highlight=highlight,
            muted=muted,
            show_legend=False,
            label_points=label_points,
        )
        if any(v is not None for v in baselines.get(panel.key, [])):
            draw_no_cluster_baselines(ax, panel, baselines[panel.key], style)

    figure_header(fig, style, headline, subhead)
    legend = fig.legend(
        handles=dot_handles(handles, style),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=min(len(handles), 4),
        frameon=True,
        fancybox=True,
        fontsize=style.legend,
        handlelength=1.4,
        handletextpad=0.7,
        borderpad=0.7,
        columnspacing=2.0,
    )
    frame = legend.get_frame()
    frame.set_facecolor(LEGEND_FILL)
    frame.set_edgecolor(LEGEND_EDGE)
    frame.set_linewidth(1.0)
    frame.set_boxstyle("round,pad=0.5,rounding_size=0.8")
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    add_x_label(fig, axes, style, x_title)
    directory, base = artifact_path(
        outdir, family, group=group, subject=subject,
        dataset=dataset_token, configs=first.configs, k=k_token,
    )
    directory.mkdir(parents=True, exist_ok=True)
    written += save(fig, directory, f"{base}_strip", formats)

    # --- 2) One standalone figure per metric --------------------------------- #
    for panel in panels:
        fig, ax = plt.subplots(figsize=style.single_size)
        fig.subplots_adjust(left=0.125, right=0.965, top=0.84, bottom=0.26)
        draw_panel(
            ax, panel, style,
            title=panel.title,
            subtitle=subtitle_text or sub(panel.configs),
            highlight=highlight,
            muted=muted,
            show_legend=True,
            label_points=label_points,
        )
        if any(v is not None for v in baselines.get(panel.key, [])):
            draw_no_cluster_baselines(ax, panel, baselines[panel.key], style)
        add_x_label(fig, [ax], style, x_title)
        directory, base = artifact_path(
            outdir, family, group=group, subject=subject,
            dataset=dataset_token, configs=panel.configs, k=k_token,
            metric=panel.key,
        )
        directory.mkdir(parents=True, exist_ok=True)
        written += save(fig, directory, base, formats)

    return written


# --------------------------------------------------------------------------- #
# Overlay view: multivariate MAE drawn over the per-channel MAE
# --------------------------------------------------------------------------- #

OVERLAY_MV = "Multivariate"
OVERLAY_PC = "Univariate (Per Channel)"


def setup_handles(style: Style) -> list[Line2D]:
    """The second key: what a line's dress means, with the hue held out of it.

    Colour is spoken for - it carries identity - so these run in ink and show
    the line itself rather than the legend's usual dot.
    """
    common = dict(color=INK_SECONDARY, marker="o", markersize=style.legend * 0.9,
                  linewidth=style.line_width * 0.85)
    return [
        Line2D([], [], linestyle="-", markerfacecolor=INK_SECONDARY,
               markeredgecolor="none", label="Multivariate", **common),
        Line2D([], [], linestyle=(0, (2.4, 1.4)), markerfacecolor=SURFACE,
               markeredgecolor=INK_SECONDARY,
               markeredgewidth=style.line_width * 0.9, label="Per-channel",
               **common),
    ]


def place_legend_pair(fig, left, right, y: float, gap: float = 0.025) -> None:
    """Centre two legend boxes side by side, tops aligned on ``y``.

    Their widths are only known once drawn, so the pair is measured and then
    re-anchored - the split decides how many entries the identity key holds.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widths = [
        lg.get_window_extent(renderer).width / fig.bbox.width for lg in (left, right)
    ]
    x = 0.5 - (sum(widths) + gap) / 2
    for legend, width in zip((left, right), widths):
        legend.set_bbox_to_anchor((x + width / 2, y))
        x += width + gap


def build_overlay_figure(
    records,
    *,
    outdir: Path,
    split: str,
    grid: str | None,
    formats: list[str],
    style: Style,
    title_text: str | None,
    subtitle_text: str | None,
) -> list[Path]:
    """One MAE panel holding both setups: multivariate lines over per-channel.

    Hue keeps its usual identity (datasets in the dataset split, models in the
    model split); the setup rides on the second-cycle idiom - multivariate
    solid with filled markers, per-channel dashed with hollow ones - so two
    nearly-equal lines stay tellable apart, and a second legend beside the
    identity key spells the dress out. The panel sits near square - a little
    wider than the single size, a little taller - and its y-axis runs log,
    which squeezes the classical models' offset at the top so the spread among
    the deep models carries more of the height. Only the (L, O) configs both
    sections share enter the mean, keeping the setups comparable; the multivariate
    draw last, on top.
    """
    shared = (
        {(r.lookback, r.horizon) for r in records if r.section == OVERLAY_MV}
        & {(r.lookback, r.horizon) for r in records if r.section == OVERLAY_PC}
    )
    rows = [r for r in records if (r.lookback, r.horizon) in shared]
    models, datasets, mv_vals, mv_configs = aggregate(rows, OVERLAY_MV, "MAE", grid)
    pc_models, pc_datasets, pc_vals, pc_configs = aggregate(rows, OVERLAY_PC, "MAE", grid)
    # What actually entered the mean, after --grid. `shared` is computed before
    # that filter, so captioning with it told a --grid run it had averaged the
    # whole ladder.
    drawn = sorted(set(mv_configs) & set(pc_configs))
    if not models or not pc_models:
        print("  no shared MV / per-channel MAE cells", file=sys.stderr)
        return []
    models = [m for m in models if m in pc_models]
    datasets = [d for d in datasets if d in pc_datasets]

    if split == "model split":  # lines = models, columns = datasets
        names, short, x_title = models, short_model, "Dataset"
        x_labels = [short_dataset(d) for d in datasets]
        series_of = lambda vals, m: [vals[d][m] for d in datasets]
    else:  # dataset split: lines = datasets, columns = models
        names, short, x_title = datasets, short_dataset, "Model"
        x_labels = [short_model(m) for m in models]
        series_of = lambda vals, d: [vals[d][m] for m in models]

    n = len(names)
    series = [(f"{short(nm)} · per-channel", series_of(pc_vals, nm)) for nm in names]
    series += [(f"{short(nm)} · multivariate", series_of(mv_vals, nm)) for nm in names]
    panel = Panel("MAE", "MAE", False, x_labels, series, drawn)

    def styler(i: int):
        """Hue = identity; per-channel (first half) dashed and hollow.

        Sixth and seventh models revisit the colour wheel, so they trade the
        usual dashed second cycle - taken here by the setup - for a shape.
        """
        base = i % n
        color, _ = series_style(base)
        return color, i < n, {5: "s", 6: "^"}.get(base, "o")

    # The header runs title / subtitle / the two legends, so its height is
    # stacked out of the font sizes rather than guessed as a figure fraction -
    # the two presets set very different type. The identity key narrows to
    # three columns to leave the setup key room beside it; whichever of the
    # two runs taller sets the header.
    ncol = min(n, 3)
    legend_rows = max(int(np.ceil(n / ncol)), 2)  # the setup key is two rows
    subtitle = subtitle_text or grid_line(drawn)
    width = style.single_size[0] * 1.2  # near square, with the header on top
    height = style.single_size[1] * 1.3
    fig, ax = plt.subplots(figsize=(width, height))
    frac = lambda points: points / 72 / height

    title_y = 0.985
    subtitle_y = title_y - frac(style.panel_title * 1.7)
    legend_y = subtitle_y - frac(
        style.subtitle * 1.45 * (subtitle.count("\n") + 1) + 10
    )
    fig.subplots_adjust(
        left=0.11, right=0.975,
        top=legend_y - frac(style.legend * 2.8 * legend_rows), bottom=0.15,
    )

    handles = draw_panel(
        ax, panel, style,
        title="", subtitle=None, highlight=None, muted=set(),
        show_legend=False, label_points=False, styler=styler, log_y=True,
    )
    for handle, nm in zip(handles[n:], names):  # legend keys identity via MV lines
        handle.set_label(short(nm))

    fig.text(0.5, title_y, title_text or "MAE",
             ha="center", va="top", fontsize=style.panel_title, fontweight="bold",
             color=INK_PRIMARY)
    fig.text(0.5, subtitle_y, subtitle, ha="center", va="top",
             fontsize=style.subtitle, color=INK_SECONDARY, linespacing=1.45)
    box = dict(loc="upper center", bbox_to_anchor=(0.5, legend_y), frameon=True,
               fancybox=True, fontsize=style.legend, handletextpad=0.7,
               borderpad=0.7, columnspacing=1.6)
    identity = frame_legend(fig.legend(
        handles=dot_handles(handles[n:], style), ncol=ncol, handlelength=1.4, **box,
    ))
    setup = frame_legend(fig.legend(
        handles=setup_handles(style), ncol=1, handlelength=2.4, **box,
    ))
    place_legend_pair(fig, identity, setup, legend_y)

    if any(short_model(m) == "ARIMA" for m in models):
        fig.text(0.985, 0.008, "ARIMA is channel-independent - both setups coincide",
                 ha="right", va="bottom", fontsize=style.footnote, color=INK_MUTED)
    add_x_label(fig, [ax], style, x_title)
    directory, base = artifact_path(outdir, "overlay", group=split,
                                    subject="mv-vs-perchannel", configs=drawn,
                                    metric="MAE")
    directory.mkdir(parents=True, exist_ok=True)
    return save(fig, directory, base, formats)


# --------------------------------------------------------------------------- #
# Trade-off view: training time against accuracy, datasets averaged
# --------------------------------------------------------------------------- #


def tradeoff_points(records, section: str, metric: str, grid: str | None):
    """One point per model: (label, mean seconds, mean value, colour, hollow).

    Both means run over the datasets (and configs), so each model collapses to a
    single cost/accuracy position. Colours reuse the model-split assignment,
    hollow markers included.
    """
    models, datasets, mvals, configs = aggregate(records, section, metric, grid)
    _, _, tvals, _ = aggregate(records, section, "Training time", grid)
    points = []
    for i, model in enumerate(models):
        ys = [mvals[d][model] for d in datasets if np.isfinite(mvals[d][model])]
        ts = [tvals[d][model] for d in datasets if np.isfinite(tvals[d][model])]
        color, second_cycle = series_style(i)
        points.append((
            short_model(model),
            float(np.mean(ts)) if ts else float("nan"),
            float(np.mean(ys)) if ys else float("nan"),
            color,
            second_cycle,
        ))
    return points, configs




def build_tradeoff_figures(
    records,
    section: str,
    *,
    outdir: Path,
    grid: str | None,
    formats: list[str],
    style: Style,
    group: str | None,
    title_text: str | None,
    subtitle_text: str | None,
) -> list[Path]:
    panels = []
    for key, _, long_title, is_time in METRICS:
        if is_time:
            continue
        points, configs = tradeoff_points(records, section, key, grid)
        if points and any(np.isfinite(t) and np.isfinite(y) for _, t, y, _, _ in points):
            panels.append((key, long_title, points, configs))

    if not panels:
        print(f"  no data for section {section!r}", file=sys.stderr)
        return []

    subhead = subtitle_text or grid_line(panels[0][3])
    headline = title_text or f"{section}: accuracy against training cost"
    written: list[Path] = []

    # --- 1) Combined pair: TT vs MSE | TT vs MAE ------------------------------ #
    fig, axes = plt.subplots(1, len(panels), figsize=style.combined_size)
    axes = list(np.atleast_1d(axes))
    fig.subplots_adjust(left=0.055, right=0.97, top=0.78, bottom=0.22, wspace=0.26)
    unit = ""
    for ax, (key, long_title, points, _) in zip(axes, panels):
        unit = draw_tradeoff_panel(ax, key, long_title, None, points, style)
    fig.text(0.5, 0.985, headline, ha="center", va="top",
             fontsize=style.title, fontweight="bold", color=INK_PRIMARY)
    fig.text(0.5, 0.985 - (style.title * 1.5) / 72 / style.combined_size[1], subhead,
             ha="center", va="top", fontsize=style.subtitle, color=INK_SECONDARY)
    add_x_label(fig, axes, style, f"Training time ({TIME_UNIT_ABBREV.get(unit, unit)})")
    directory, base = artifact_path(outdir, "tradeoff", group=group,
                                    configs=panels[0][3])
    directory.mkdir(parents=True, exist_ok=True)
    written += save(fig, directory, f"{base}_strip", formats)

    # --- 2) One standalone figure per metric --------------------------------- #
    for key, long_title, points, configs in panels:
        fig, ax = plt.subplots(figsize=style.single_size)
        fig.subplots_adjust(left=0.125, right=0.94, top=0.84, bottom=0.2)
        unit = draw_tradeoff_panel(
            ax, key, long_title, subtitle_text or grid_line(configs), points, style
        )
        add_x_label(fig, [ax], style, f"Training time ({TIME_UNIT_ABBREV.get(unit, unit)})")
        directory, base = artifact_path(outdir, "tradeoff", group=group,
                                        configs=configs, metric=f"{key}-vs-tt")
        directory.mkdir(parents=True, exist_ok=True)
        written += save(fig, directory, base, formats)

    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=Path, default=here / "frames" / "cells.csv",
                        help="aggregated cells to plot (default: frames/cells.csv)")
    parser.add_argument("--tables", type=Path,
                        default=here / "archive" / "2026-08-11_tables.md",
                        help="legacy markdown source; only used with --source tables")
    parser.add_argument("--source", choices=["frames", "tables"], default="frames",
                        help="where the numbers come from (default: frames)")
    parser.add_argument("--outdir", type=Path, default=here / "figures")
    parser.add_argument(
        "--section", default="all", choices=["all", *SECTION_ALIASES, "cluster"],
        help="which results tree to plot (default: everything)",
    )
    parser.add_argument(
        "--grid",
        help="keep a single lookback/horizon config, e.g. 336/96 (default: average them)",
    )
    parser.add_argument("--formats", default="png", help="comma-separated output formats")
    parser.add_argument(
        "--style", default="slide", choices=list(STYLES),
        help="slide: presentation proportions (default); paper: IEEEtran column sizing",
    )
    parser.add_argument("--font", default="sans", choices=list(FONT_STACKS))
    parser.add_argument(
        "--highlight", default="",
        help="optionally bold this model's tick label and label its values",
    )
    parser.add_argument(
        "--dataset", action="append", default=[], metavar="NAME",
        help="keep only this dataset, e.g. ED2022 (repeatable; default: all)",
    )
    parser.add_argument(
        "--mute", action="append", default=[], metavar="SERIES",
        help="draw this series (dataset or model) in grey as a reference (repeatable)",
    )
    parser.add_argument(
        "--labels", action="store_true", help="print the value beside every point"
    )
    parser.add_argument("--title", help="headline above the combined figure")
    parser.add_argument("--subtitle", help="grey line under the headline")
    parser.add_argument("--list", action="store_true", help="show what was parsed and exit")
    args = parser.parse_args(argv)

    if args.source == "frames" and not args.frames.exists():
        parser.error(f"frames not found: {args.frames} -- run `python results/make.py extract`")
    if args.source == "tables" and not args.tables.exists():
        parser.error(f"tables file not found: {args.tables}")

    if args.grid:
        try:
            parse_grid(args.grid)
        except ValueError as exc:
            parser.error(str(exc))

    if args.source == "frames":
        from emfres.records import records_from_cells
        records = records_from_cells(args.frames, here)
        where = args.frames
    else:
        records = parse_tables(args.tables)
        where = args.tables
    if not records:
        parser.error(f"no metric records found in {where}")

    if args.dataset:
        wanted = {name.lower() for name in args.dataset}
        available = sorted({short_dataset(r.dataset) for r in records})
        records = [r for r in records if short_dataset(r.dataset).lower() in wanted]
        if not records:
            parser.error(
                f"no rows match --dataset {', '.join(args.dataset)}; "
                f"available: {', '.join(available)}"
            )

    if args.list:
        for section in dict.fromkeys(r.section for r in records):
            print(f"\n{section}")
            for key, _, _, _ in METRICS:
                if section == CLUSTER_SECTION:
                    ks, pairs, _, configs = aggregate_clusters(
                        records, section, key, args.grid
                    )
                    if ks:
                        print(
                            f"  {key:<14} {len(pairs)} series x k = "
                            f"{{{', '.join(map(str, ks))}}}"
                            f"  [{', '.join(f'L{l}/O{o}' for l, o in configs)}]"
                        )
                    continue
                models, datasets, _, configs = aggregate(records, section, key, args.grid)
                if models:
                    print(
                        f"  {key:<14} {len(datasets)} datasets x {len(models)} models"
                        f"  [{', '.join(f'L{l}/O{o}' for l, o in configs)}]"
                    )
        return 0

    style = STYLES[args.style]
    plt.rcParams.update({**rc_fonts(args.font), "font.size": style.tick})

    sections = (
        list(SECTION_ALIASES.values())
        if args.section == "all"
        else []
        if args.section == "cluster"
        else [SECTION_ALIASES[args.section]]
    )

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    for split, family in (("dataset split", "by-dataset"), ("model split", "by-model")):
        outdir = args.outdir
        for section in sections:
            stem = next(k for k, v in SECTION_ALIASES.items() if v == section)
            print(f"{family} / {section}:")
            for path in build_figures(
                records,
                section,
                outdir=outdir,
                grid=args.grid,
                formats=formats,
                style=style,
                highlight=args.highlight or None,
                muted=set(args.mute),
                label_points=args.labels,
                family=family,
                group=STEM_SLUGS.get(stem, stem),
                split=split,
                title_text=args.title,
                subtitle_text=args.subtitle,
            ):
                print(f"  wrote {path}")
        if args.section in ("all", "multivariate", "perchannel"):
            print(f"overlay / {family}:")
            for path in build_overlay_figure(
                records,
                outdir=outdir,
                split=family,
                grid=args.grid,
                formats=formats,
                style=style,
                title_text=args.title,
                subtitle_text=args.subtitle,
            ):
                print(f"  wrote {path}")

    outdir = args.outdir
    for section in sections:
        stem = next(k for k, v in SECTION_ALIASES.items() if v == section)
        print(f"tradeoff / {section}:")
        for path in build_tradeoff_figures(
            records,
            section,
            outdir=outdir,
            grid=args.grid,
            formats=formats,
            style=style,
            group=STEM_SLUGS.get(stem, stem),
            title_text=args.title,
            subtitle_text=args.subtitle,
        ):
            print(f"  wrote {path}")

    if args.section in ("all", "cluster"):
        # Without an explicit --dataset the cluster figures keep only the
        # study's dataset; nine (dataset, model) lines drown the k trend.
        cluster_records = records if args.dataset else [
            r for r in records
            if short_dataset(r.dataset).lower() == CLUSTER_DEFAULT_DATASET.lower()
        ]
        # One dataset per cluster figure, so the dataset is a filename token
        # rather than a directory -- and naming it is what stops a bare and a
        # --dataset run writing the same bytes to two different places.
        cluster_ds = short_dataset(cluster_records[0].dataset) if cluster_records else None
        print(f"cluster-sweep / {CLUSTER_SECTION}:")
        for path in build_figures(
            cluster_records,
            CLUSTER_SECTION,
            outdir=args.outdir,
            grid=args.grid,
            formats=formats,
            style=style,
            highlight=args.highlight or None,
            muted=set(args.mute),
            label_points=args.labels,
            family="cluster-sweep",
            group=None,
            dataset_token=cluster_ds,
            k_token="all",
            split="cluster",
            title_text=args.title,
            subtitle_text=args.subtitle,
        ):
            print(f"  wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
