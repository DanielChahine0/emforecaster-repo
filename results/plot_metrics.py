#!/usr/bin/env python3
"""Turn the result tables in ``tables.md`` into MSE / MAE / training-time figures.

Each figure puts the models along the x-axis (ordered by year of publication) and
one line per dataset, so a panel reads as "how this metric moves across model
generations, per dataset". Three panels - MSE, MAE, total training time - are
emitted both as one combined strip and as standalone single figures, twice:
``figures/dataset split/`` keeps datasets as the lines with models along the
x-axis, and ``figures/model split/`` flips the roles. Each split also gets
``mv-vs-perchannel_mae``, one panel overlaying both setups' MAE (multivariate
solid over per-channel dashed). The clustering study (k-grouped channels)
instead puts the cluster count k on the x-axis and lands in
``figures/cluster/``.

Usage
-----
    python3 plot_metrics.py                          # every section, PNG
    python3 plot_metrics.py --section multivariate
    python3 plot_metrics.py --grid 336/96            # keep one (L, O) config
    python3 plot_metrics.py --dataset ED2022         # keep one dataset (repeatable)
    python3 plot_metrics.py --mute DET               # gray one dataset as reference
    python3 plot_metrics.py --style paper            # IEEEtran column sizing
    python3 plot_metrics.py --list                   # show what was parsed

The tables are the single source of truth: nothing is duplicated in this file.
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
# Design tokens
# --------------------------------------------------------------------------- #

# The five style-example colours - Verdigris, Burnt Peach, Charcoal Blue,
# Sandy Brown, Tuscan Sun - ordered for contrast on a white surface. The
# leading Verdigris / Burnt Peach / Charcoal trio is the dataset-split set and
# passes the all-pairs check (deltaE >= 42.0, >= 28.0 under simulated CVD);
# across all five, adjacent pairs hold deltaE >= 23.2 normal-vision, though the
# twin ambers compress to 6.2 for deutan viewers - the legend and the paper's
# tables are the relief. Sandy Brown and Tuscan Sun sit below 3:1 contrast.
# Uniform circle markers, like the reference - hue carries identity, and
# series past the fifth restate the wheel dashed-and-hollow (series_style).
SERIES_COLORS = [
    "#2a9d8f", "#e76f51", "#264653", "#f4a261", "#e9c46a",
]
# A deliberately recessive mist step for a series named with --mute: it stays
# legible and keeps its legend entry, but stops competing for attention.
MUTED_COLOR = "#d7e0e4"

INK_PRIMARY = "#264653"
INK_SECONDARY = "#54707e"
INK_MUTED = "#8ba1ab"
GRIDLINE = "#e4eaee"
AXIS_RULE = "#264653"
SURFACE = "#ffffff"
LEGEND_FILL = "#fbfcfd"
LEGEND_EDGE = "#d5dde2"


def series_style(i: int) -> tuple[str, bool]:
    """Colour and second-cycle flag for series ``i``.

    The palette stays fixed at the five theme colours; a sixth or seventh
    series repeats the wheel one step ahead - so the lead colour, the proposed
    model's, never duplicates - and the True flag asks for a dashed line with a
    hollow marker to carry the difference.
    """
    cycle, slot = divmod(i, len(SERIES_COLORS))
    return SERIES_COLORS[(slot + cycle) % len(SERIES_COLORS)], cycle > 0


def label_color(color: str) -> str:
    """Darken a series colour until small bold text in it reads on white (3:1).

    Lines and markers keep the true palette; only annotations pay the tax, so
    the light saffron / sandy / sage slots stay legible as text.
    """
    rgb = np.array(mcolors.to_rgb(color))

    def contrast(c: np.ndarray) -> float:
        linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return 1.05 / (linear @ [0.2126, 0.7152, 0.0722] + 0.05)

    while contrast(rgb) < 3.0:
        rgb *= 0.97
    return mcolors.to_hex(rgb)


FONT_STACKS = {
    # IBM Plex Sans sets every figure in the repo; the rest of the list is the
    # fallback chain for a machine that does not have it. Install the family
    # with results/install_plex_sans.py - Google Fonts ships only the variable
    # font, whose bold matplotlib cannot reach.
    "sans": [
        "IBM Plex Sans",
        "Lato", "Source Sans 3", "Source Sans Pro", "Open Sans", "Noto Sans",
        "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans",
    ],
    "serif": ["IBM Plex Serif", "Times New Roman", "STIXGeneral", "DejaVu Serif"],
}


def rc_fonts(family: str = "sans") -> dict:
    """rcParams that put every glyph - mathtext included - in one family.

    Every figure script shares this, so the typeface is named in one place.
    Mathtext runs through the ``custom`` fontset rather than a stock one:
    otherwise the paradigm schematic's $f_1$ would arrive in DejaVu beside
    Plex captions.
    """
    lead = FONT_STACKS[family][0]
    return {
        "font.family": "sans-serif" if family == "sans" else "serif",
        "font.sans-serif": FONT_STACKS["sans"],
        "font.serif": FONT_STACKS["serif"],
        # Every custom slot is resolved at draw time, so name the family in all
        # of them - the unused calligraphic one otherwise warns its way to a
        # cursive fallback on each run.
        "mathtext.fontset": "custom",
        "mathtext.rm": lead,
        "mathtext.it": f"{lead}:italic",
        "mathtext.bf": f"{lead}:bold",
        "mathtext.bfit": f"{lead}:italic:bold",
        "mathtext.sf": lead,
        "mathtext.cal": f"{lead}:italic",
        "text.color": INK_PRIMARY,
        "axes.labelcolor": INK_SECONDARY,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42,  # embed real glyphs, not type-3 outlines
        "ps.fonttype": 42,
    }


@dataclass(frozen=True)
class Style:
    """Everything that changes between the presentation and the LaTeX preset."""

    combined_size: tuple[float, float]
    single_size: tuple[float, float]
    title: float
    subtitle: float
    panel_title: float
    axis_label: float
    context: float
    tick: float
    legend: float
    footnote: float
    line_width: float
    marker: float

    @property
    def value_label(self) -> float:
        return self.tick * 0.95


STYLES = {
    # Presentation / preprint proportions: heavy strokes, generous margins.
    "slide": Style(
        combined_size=(11.5, 5.4),
        single_size=(6.4, 5.0),
        title=17,
        subtitle=11.5,
        panel_title=13,
        axis_label=11.5,
        context=9,
        tick=10,
        legend=10.5,
        footnote=7.5,
        line_width=2.8,
        marker=8.5,
    ),
    # IEEEtran: \columnwidth ~ 3.5in, \textwidth ~ 7.16in, no downscaling.
    "paper": Style(
        combined_size=(7.16, 3.7),
        single_size=(3.5, 3.3),
        title=10.5,
        subtitle=8,
        panel_title=9,
        axis_label=8,
        context=6.5,
        tick=7,
        legend=7,
        footnote=5.6,
        line_width=1.8,
        marker=5.0,
    ),
}


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

TIME_UNITS = {"h": 3600.0, "m": 60.0, "s": 1.0}
_TIME_RE = re.compile(r"^(?:\d+(?:\.\d+)?\s*[hms]\s*)+$", re.IGNORECASE)
_TIME_PART_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([hms])", re.IGNORECASE)


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
    if _TIME_RE.match(text):
        return sum(
            float(v) * TIME_UNITS[u.lower()] for v, u in _TIME_PART_RE.findall(text)
        )
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


@dataclass
class Panel:
    key: str
    title: str
    is_time: bool
    x_labels: list[str]  # display-ready column names
    series: list[tuple[str, list[float]]]  # (display label, value per column)
    configs: list[tuple[int, int]]


def aggregate(records, section: str, metric: str, grid: str | None):
    """Mean each (dataset, model) cell over the (L, O) configs still in play."""
    rows = [r for r in records if r.section == section and r.metric == metric]
    if grid:
        lookback, horizon = (int(v) for v in grid.lower().replace("l", "").split("/"))
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
        lookback, horizon = (int(v) for v in grid.lower().replace("l", "").split("/"))
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


# --------------------------------------------------------------------------- #
# Scales: one unit and one number format per axis
# --------------------------------------------------------------------------- #

# A 1-2-3-5 ladder, so every axis draws its ticks from the same shape.
LADDER = [m * 10**k for k in range(-3, 6) for m in (1, 2, 3, 5)]


def time_unit(seconds_max: float) -> tuple[float, str]:
    """One unit for the whole axis - mixing '45s' with '2m' reads inconsistently."""
    if seconds_max < 120:
        return 1.0, "seconds"
    if seconds_max < 10800:
        return 60.0, "minutes"
    return 3600.0, "hours"


# Compact axis-label spellings of the time_unit names.
TIME_UNIT_ABBREV = {"seconds": "s", "minutes": "min", "hours": "h"}


def log_ticks(lo: float, hi: float, target: int = 6) -> list[float]:
    """Ladder values inside [lo, hi], thinned toward ``target`` ticks."""
    inside = [t for t in LADDER if lo <= t <= hi]
    if len(inside) <= target:
        return inside
    step = int(np.ceil(len(inside) / target))
    return inside[::step]


def linear_ticks(lo: float, hi: float, target: int = 6) -> list[float]:
    """Evenly spaced ticks on a round step - a log ladder on a linear axis reads wrong."""
    raw = (hi - lo) / target
    magnitude = 10 ** np.floor(np.log10(raw))
    step = next(
        (m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw), 10 * magnitude
    )
    first = np.ceil(lo / step) * step
    return [round(first + i * step, 10) for i in range(int((hi - first) / step) + 1)]


def digits_for_step(step: float) -> int:
    """One decimal count for every tick on the axis."""
    for threshold, digits in ((10, 0), (1, 1), (0.1, 2)):
        if step >= threshold:
            return digits
    return 3


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #


def style_axes(ax, style: Style) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    rule = style.line_width * 0.55
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_RULE)
        ax.spines[side].set_linewidth(rule)
    ax.tick_params(colors=AXIS_RULE, length=4, width=rule, labelcolor=INK_PRIMARY)
    ax.grid(axis="x", color=GRIDLINE, linewidth=style.line_width * 0.32, zorder=0)
    ax.set_axisbelow(True)


def draw_axis_break(ax, style: Style, below: float) -> None:
    """Kink the y-axis spine into a small zigzag - the truncated-scale mark.

    Two sharp teeth poke left of the spine just above the origin: the
    conventional sign that a linear axis does not run down to zero. ``below``
    is the axes-fraction height of the lowest tick; the glyph centres itself
    under it, clear of the tick and its label.
    """
    fig = ax.figure
    pos = ax.get_position()
    width_pt = pos.width * fig.get_size_inches()[0] * 72.0
    height_pt = pos.height * fig.get_size_inches()[1] * 72.0
    # Sized to sit under the tick labels: a touch shorter than the tick font,
    # so the lowest number reads taller than the glyph.
    half = 0.32 * style.tick / height_pt
    reach = 0.36 * style.tick / width_pt
    centre = min(0.085, max(0.04, below / 2))
    ys = [centre + half, centre + half / 2, centre, centre - half / 2, centre - half]
    xs = [0.0, -reach, 0.0, -reach, 0.0]

    spine = ax.spines["left"]
    # Blank the straight spine behind the glyph first, so the axis line
    # really kinks instead of showing through the teeth.
    ax.add_artist(Line2D(
        [0.0, 0.0], [ys[0], ys[-1]], transform=ax.transAxes,
        color=SURFACE, linewidth=spine.get_linewidth() * 3.0,
        solid_capstyle="butt", clip_on=False, zorder=spine.get_zorder() + 0.1,
    ))
    ax.add_artist(Line2D(
        xs, ys, transform=ax.transAxes,
        color=AXIS_RULE, linewidth=spine.get_linewidth(),
        solid_capstyle="butt", solid_joinstyle="miter",
        clip_on=False, zorder=spine.get_zorder() + 0.2,
    ))


def densify(points: list[tuple[float, float]], per_segment: int = 10):
    """Sample along each polyline segment - a legend must dodge lines, not just markers."""
    dense = []
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        for t in np.linspace(0, 1, per_segment):
            dense.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return dense + points


# (location, columns) tried in order; the first placement that touches no data
# wins. The stacked box reads best when a corner is free; the single row fits the
# thin band of headroom reserved at the top when no corner is.
LEGEND_PLACEMENTS = [
    ("upper right", 1), ("upper left", 1), ("lower left", 1), ("lower right", 1),
    ("upper left", 0), ("upper right", 0), ("lower left", 0),
]


def place_legend(ax, handles, style: Style, points, avoid) -> bool:
    """Try each corner against the legend's real box; hang it above the axes if none is clear.

    Counting points per quadrant is not enough - a legend needs clearance, and a
    flat line crossing a sparse corner still runs under the box.
    """
    fig = ax.figure
    fig.canvas.draw()
    marks = [
        ax.transData.transform((px, py)) for px, py in points if np.isfinite(py)
    ]
    boxes = [a.get_window_extent(fig.canvas.get_renderer()) for a in avoid]

    best = (None, float("inf"))
    for loc, ncol in LEGEND_PLACEMENTS:
        legend = add_legend(ax, handles, style, loc, ncol or len(handles))
        fig.canvas.draw()
        frame = legend.get_window_extent(fig.canvas.get_renderer()).expanded(1.14, 1.3)
        hits = sum(frame.x0 <= px <= frame.x1 and frame.y0 <= py <= frame.y1 for px, py in marks)
        hits += sum(frame.overlaps(box) for box in boxes)
        if hits < best[1]:
            best = ((loc, ncol), hits)
        if hits == 0:
            return False
        legend.remove()

    if best[1] == 0:
        loc, ncol = best[0]
        add_legend(ax, handles, style, loc, ncol or len(handles))
        return False

    # Nothing inside is clear: a horizontal strip above the axes never collides.
    legend = add_legend(ax, handles, style, "lower left", len(handles))
    legend.set_bbox_to_anchor((0.0, 1.02, 1.0, 0.12), transform=ax.transAxes)
    legend.set_ncols(len(handles))
    return True


def draw_panel(
    ax,
    panel: Panel,
    style: Style,
    *,
    title: str,
    subtitle: str | None,
    highlight: str | None,
    muted: set[str],
    show_legend: bool,
    label_points: bool,
    styler=None,
    log_y: bool = False,
) -> list[Line2D]:
    x = np.arange(len(panel.x_labels))
    finite = [v for _, vals in panel.series for v in vals if np.isfinite(v)]

    # --- scale: one unit, one number format, for ticks and value labels alike --
    # The panel title names the metric, so the y-axis carries only what the
    # title cannot: the time unit. MSE / MAE axes stay label-free.
    if panel.is_time:
        divisor, unit = time_unit(max(finite))
        series = [(name, [v / divisor for v in vals]) for name, vals in panel.series]
        span = [v / divisor for v in finite]
        lo, hi = min(span) / 1.3, max(span) * 1.3
        if show_legend:  # empty band at the top for the legend to sit in
            hi *= (hi / lo) ** 0.20
        ticks = log_ticks(lo, hi)
        digits = 0 if all(float(t).is_integer() for t in ticks) else 1
        axis_label = TIME_UNIT_ABBREV.get(unit, unit)
    else:
        series = panel.series
        span = list(finite)
        pad = 0.09 * (max(span) - min(span))
        lo, hi = min(span) - pad, max(span) + pad
        if show_legend:  # empty band at the top for the legend to sit in
            hi += 0.24 * (hi - lo)
        ticks = linear_ticks(lo, hi)
        digits = digits_for_step(ticks[1] - ticks[0]) if len(ticks) > 1 else 2
        axis_label = None

    fmt = lambda v: f"{v:.{digits}f}"
    highlight_at = next(
        (i for i, name in enumerate(panel.x_labels) if name == highlight), None
    )
    label_side = -1 if highlight_at == 0 else 1  # labels face the nearer plot edge

    # Endpoint labels carry one more decimal if the axis format would print two
    # series as the same number.
    label_digits = digits
    if highlight_at is not None:
        endpoints = [vals[highlight_at] for name, vals in series if name not in muted]
        while label_digits < digits + 2 and len(
            {f"{v:.{label_digits}f}" for v in endpoints if np.isfinite(v)}
        ) < len([v for v in endpoints if np.isfinite(v)]):
            label_digits += 1
    label_fmt = lambda v: f"{v:.{label_digits}f}"

    # --- series -------------------------------------------------------------- #
    handles: list[Line2D] = []
    occupied: list[tuple[float, float]] = []
    value_labels = []
    for i, (name, vals) in enumerate(series):
        y = np.array(vals, dtype=float)
        is_muted = name in muted
        if styler is not None:
            color, second_cycle, marker = styler(i)
        else:
            (color, second_cycle), marker = series_style(i), "o"
        if is_muted:
            color = MUTED_COLOR
        depth = 2 if is_muted else 4 + i

        (line,) = ax.plot(
            x,
            y,
            color=color,
            linestyle=(0, (2.4, 1.4)) if second_cycle else "-",
            marker=marker,
            markersize=style.marker,
            markerfacecolor=SURFACE if second_cycle else color,
            markeredgecolor=color if second_cycle else "none",
            markeredgewidth=style.line_width * 0.9 if second_cycle else 0.0,
            linewidth=style.line_width,
            solid_capstyle="round",
            solid_joinstyle="round",
            dash_capstyle="round",
            label=name,
            zorder=depth,
            clip_on=False,
        )
        handles.append(line)
        occupied += densify(list(zip(x.tolist(), y.tolist())))

        # The proposed model gets the only value label on the line - the marks
        # themselves stay uniform, like the reference.
        if highlight_at is not None and not is_muted and np.isfinite(y[highlight_at]):
            value_labels.append(ax.annotate(
                label_fmt(y[highlight_at]),
                (x[highlight_at], y[highlight_at]),
                textcoords="offset points",
                xytext=(label_side * style.marker * 1.2, 0),
                ha="left" if label_side > 0 else "right",
                va="center",
                fontsize=style.value_label,
                color=label_color(color),
                fontweight="bold",
                zorder=depth + 0.7,
                annotation_clip=False,
            ))

        if label_points:
            for xi, yi in zip(x, y):
                if np.isfinite(yi) and xi != highlight_at:
                    ax.annotate(
                        fmt(yi),
                        (xi, yi),
                        textcoords="offset points",
                        xytext=(0, style.marker),
                        ha="center",
                        fontsize=style.value_label * 0.9,
                        color=INK_SECONDARY,
                    )

    # --- frame --------------------------------------------------------------- #
    style_axes(ax, style)
    n = len(panel.x_labels)
    # Room beside the highlighted column for its value label.
    if highlight_at is None:
        ax.set_xlim(-0.45, n - 0.55)
    elif label_side > 0:
        ax.set_xlim(-0.45, n - 1 + 0.85)
    else:
        ax.set_xlim(-0.85, n - 1 + 0.45)
    ax.set_ylim(lo, hi)
    ax.set_xticks(x)
    # Short label sets sit flat; long ones tilt to keep from colliding.
    longest = max(map(len, panel.x_labels))
    flat = longest <= 4 or (n <= 4 and longest <= 8)
    ax.set_xticklabels(
        panel.x_labels,
        rotation=0 if flat else 28,
        ha="center" if flat else "right",
        fontsize=style.tick,
    )
    if highlight_at is not None:
        label = ax.get_xticklabels()[highlight_at]
        label.set_fontweight("bold")
        label.set_color(INK_PRIMARY)

    if panel.is_time or log_y:
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v)))
    if axis_label:
        ax.set_ylabel(axis_label, fontsize=style.axis_label, color=INK_PRIMARY, labelpad=7)
    ax.tick_params(axis="y", labelsize=style.tick)

    # A linear scale that does not reach zero says so with the conventional
    # zigzag break in the spine; a log axis has no zero to truncate, so it
    # shows its scale through the uneven tick spacing instead.
    if not panel.is_time and not log_y and lo > 0:
        draw_axis_break(ax, style, below=(ticks[0] - lo) / (hi - lo) if ticks else 1.0)

    # A column with no logged run keeps its slot so the panels stay aligned.
    for xi in x:
        if not any(np.isfinite(vals[xi]) for _, vals in panel.series):
            ax.annotate(
                "not logged",
                (xi, 0),
                xycoords=("data", "axes fraction"),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                rotation=90,
                fontsize=style.context * 0.85,
                color=INK_MUTED,
            )

    # A series with nothing logged in this panel keeps its legend entry; the
    # panel itself says why the line is absent.
    missing = [name for name, vals in panel.series if not any(np.isfinite(v) for v in vals)]
    if missing:
        ax.text(
            0.975, 0.035, f"{', '.join(missing)}: not logged", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=style.context * 0.9, color=INK_MUTED,
        )

    spread_labels(ax, value_labels, style.value_label * 1.3)
    outside = show_legend and place_legend(ax, handles, style, occupied, value_labels)

    lift = 0.15 if outside else 0.0
    ax.text(
        0.5, (1.16 if subtitle else 1.08) + lift, title, transform=ax.transAxes,
        fontsize=style.panel_title, fontweight="bold", color=INK_PRIMARY,
        ha="center", va="baseline",
    )
    if subtitle:
        ax.text(
            0.5, 1.06 + lift, subtitle, transform=ax.transAxes,
            fontsize=style.subtitle, color=INK_SECONDARY,
            ha="center", va="baseline",
        )

    return handles


def spread_labels(ax, labels, min_gap_pt: float) -> None:
    """Push overlapping endpoint labels apart - two series can land on one value."""
    if len(labels) < 2:
        return
    fig = ax.figure
    fig.canvas.draw()
    to_points = 72 / fig.dpi
    gap_px = min_gap_pt / to_points
    ordered = sorted(labels, key=lambda a: ax.transData.transform(a.xy)[1])
    previous = -np.inf
    for ann in ordered:
        anchor = ax.transData.transform(ann.xy)[1]
        current = anchor + ann.xyann[1] / to_points
        if current - previous < gap_px:
            current = previous + gap_px
            ann.xyann = (ann.xyann[0], (current - anchor) * to_points)
        previous = current


def dot_handles(handles, style: Style):
    """Plain colour dots, the way the reference legend keys its series.

    Second-cycle series keep their hollow ring so the key mirrors the line.
    """
    return [
        Line2D(
            [], [], linestyle="None",
            marker=h.get_marker(), markersize=style.legend * 1.15,
            markerfacecolor=h.get_markerfacecolor(),
            markeredgecolor=h.get_markeredgecolor(),
            markeredgewidth=h.get_markeredgewidth(),
            label=h.get_label(),
        )
        for h in handles
    ]


def add_legend(ax, handles, style: Style, loc: str, ncol: int = 1):
    legend = ax.legend(
        handles=dot_handles(handles, style),
        loc=loc,
        ncol=ncol,
        columnspacing=1.8,
        frameon=True,
        fancybox=True,
        fontsize=style.legend,
        handlelength=1.4,
        handletextpad=0.7,
        borderpad=0.85,
        labelspacing=0.65,
        borderaxespad=0.9,
    )
    frame = legend.get_frame()
    frame.set_facecolor(LEGEND_FILL)
    frame.set_edgecolor(LEGEND_EDGE)
    frame.set_linewidth(1.0)
    frame.set_boxstyle("round,pad=0.5,rounding_size=0.8")
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)
    return legend


# --------------------------------------------------------------------------- #
# Figure furniture
# --------------------------------------------------------------------------- #


def axes_bottom(fig, axes) -> float:
    """Figure fraction of the lowest ink in ``axes`` (rotated tick labels included)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    return min(ax.get_tightbbox(renderer).y0 for ax in axes) / fig.bbox.height


def add_x_label(fig, axes, style: Style, label: str) -> None:
    """Shared x-axis label, centred below the rotated tick labels."""
    height = fig.get_size_inches()[1]
    centre = np.mean([ax.get_position().x0 + ax.get_position().width / 2 for ax in axes])
    y = axes_bottom(fig, axes) - 0.07 / height
    fig.text(
        centre, y, label, ha="center", va="top",
        fontsize=style.axis_label, color=INK_PRIMARY, fontweight="bold",
    )


def grid_line(configs: list[tuple[int, int]]) -> str:
    """``lookback length 192, 336 · forecast horizon 96`` from the configs in play.

    The symbols L and O belong to the tables; a caption spells them out. An
    evenly spaced run of four or more windows collapses to its endpoints and
    step - the ladder's ten lookbacks, listed, ran wider than the figure.
    """

    def fmt(values: list[int]) -> str:
        if len(values) == 1:
            return str(values[0])
        steps = {b - a for a, b in zip(values, values[1:])}
        if len(values) >= 4 and len(steps) == 1:
            return f"{values[0]}-{values[-1]} every {steps.pop()}"
        return ", ".join(map(str, values))

    lookbacks = sorted({l for l, _ in configs})
    horizons = sorted({o for _, o in configs})
    return f"lookback length {fmt(lookbacks)} · forecast horizon {fmt(horizons)}"


def save(fig, outdir: Path, stem: str, formats: list[str]) -> list[Path]:
    written = []
    for fmt in formats:
        path = outdir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=300, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.06)
        written.append(path)
    plt.close(fig)
    return written


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
    stem: str,
    split: str,
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

    fig.text(0.5, 0.99, headline, ha="center", va="top",
             fontsize=style.title, fontweight="bold", color=INK_PRIMARY)
    fig.text(0.5, 0.99 - (style.title * 1.35) / 72 / style.combined_size[1], subhead,
             ha="center", va="top", fontsize=style.subtitle, color=INK_SECONDARY)
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
    written += save(fig, outdir, stem, formats)

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
        slug = panel.key.lower().replace(" ", "-")
        written += save(fig, outdir, f"{stem}_{slug}", formats)

    return written


# --------------------------------------------------------------------------- #
# Overlay view: multivariate MAE drawn over the per-channel MAE
# --------------------------------------------------------------------------- #

OVERLAY_MV = "Multivariate"
OVERLAY_PC = "Univariate (Per Channel)"


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
    nearly-equal lines stay tellable apart. The panel runs wide - the full
    combined-strip width at the usual single height - and its y-axis runs
    log: the width spreads the models apart, and the log squeezes the
    classical models' offset at the top so the spread among the deep models
    carries more of the height. Only the (L, O) configs both sections share
    enter the mean, keeping the setups comparable; the multivariate series
    draw last, on top.
    """
    shared = (
        {(r.lookback, r.horizon) for r in records if r.section == OVERLAY_MV}
        & {(r.lookback, r.horizon) for r in records if r.section == OVERLAY_PC}
    )
    rows = [r for r in records if (r.lookback, r.horizon) in shared]
    models, datasets, mv_vals, _ = aggregate(rows, OVERLAY_MV, "MAE", grid)
    pc_models, pc_datasets, pc_vals, _ = aggregate(rows, OVERLAY_PC, "MAE", grid)
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
    panel = Panel("MAE", "MAE", False, x_labels, series, sorted(shared))

    def styler(i: int):
        """Hue = identity; per-channel (first half) dashed and hollow.

        Sixth and seventh models revisit the colour wheel, so they trade the
        usual dashed second cycle - taken here by the setup - for a shape.
        """
        base = i % n
        color, _ = series_style(base)
        return color, i < n, {5: "s", 6: "^"}.get(base, "o")

    # The header runs title / two subtitle lines / legend, so its height is
    # stacked out of the font sizes rather than guessed as a figure fraction -
    # the two presets set very different type.
    legend_rows = int(np.ceil(n / 4))
    subtitle = subtitle_text or (
        f"{grid_line(sorted(shared))}\nsolid = multivariate, dashed = per-channel"
    )
    width = style.combined_size[0]  # wide panel: room between the models
    height = style.single_size[1]
    height += 0.30 if legend_rows == 1 else 0.55  # the extra subtitle line
    fig, ax = plt.subplots(figsize=(width, height))
    frac = lambda points: points / 72 / height

    title_y = 0.985
    subtitle_y = title_y - frac(style.panel_title * 1.7)
    legend_y = subtitle_y - frac(
        style.subtitle * 1.45 * (subtitle.count("\n") + 1) + 10
    )
    fig.subplots_adjust(
        left=0.075, right=0.975,
        top=legend_y - frac(style.legend * 2.8 * legend_rows), bottom=0.24,
    )

    handles = draw_panel(
        ax, panel, style,
        title="", subtitle=None, highlight=None, muted=set(),
        show_legend=False, label_points=False, styler=styler, log_y=True,
    )
    for handle, nm in zip(handles[n:], names):  # legend keys identity via MV lines
        handle.set_label(short(nm))

    fig.text(0.5, title_y, title_text or "MAE: multivariate vs per-channel",
             ha="center", va="top", fontsize=style.panel_title, fontweight="bold",
             color=INK_PRIMARY)
    fig.text(0.5, subtitle_y, subtitle, ha="center", va="top",
             fontsize=style.subtitle, color=INK_SECONDARY, linespacing=1.45)
    legend = fig.legend(
        handles=dot_handles(handles[n:], style),
        loc="upper center", bbox_to_anchor=(0.5, legend_y),
        ncol=min(n, 4), frameon=True, fancybox=True, fontsize=style.legend,
        handlelength=1.4, handletextpad=0.7, borderpad=0.7, columnspacing=1.6,
    )
    frame = legend.get_frame()
    frame.set_facecolor(LEGEND_FILL)
    frame.set_edgecolor(LEGEND_EDGE)
    frame.set_linewidth(1.0)
    frame.set_boxstyle("round,pad=0.5,rounding_size=0.8")
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    if any(short_model(m) == "ARIMA" for m in models):
        fig.text(0.985, 0.008, "ARIMA is channel-independent - both setups coincide",
                 ha="right", va="bottom", fontsize=style.footnote, color=INK_MUTED)
    add_x_label(fig, [ax], style, x_title)
    return save(fig, outdir, "mv-vs-perchannel_mae", formats)


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


def draw_tradeoff_panel(ax, key: str, title: str, subtitle: str | None, points, style: Style) -> str:
    """Scatter of models in cost/accuracy space; returns the time unit used."""
    plotted = [p for p in points if np.isfinite(p[1]) and np.isfinite(p[2])]
    missing = [p[0] for p in points if not (np.isfinite(p[1]) and np.isfinite(p[2]))]

    divisor, unit = time_unit(max(t for _, t, _, _, _ in plotted))
    xs = [t / divisor for _, t, _, _, _ in plotted]
    ys = [y for _, _, y, _, _ in plotted]

    # Log time axis, with room on the right for the point labels.
    xlo, xhi = min(xs) / 1.5, max(xs) * 2.6
    xticks = log_ticks(xlo, xhi)
    xdigits = 0 if all(float(t).is_integer() for t in xticks) else 1
    pad = 0.12 * (max(ys) - min(ys))
    ylo, yhi = min(ys) - pad, max(ys) + pad
    yticks = linear_ticks(ylo, yhi)
    ydigits = digits_for_step(yticks[1] - yticks[0]) if len(yticks) > 1 else 2

    labels = []
    for name, t, y, color, second_cycle in plotted:
        ax.plot(
            [t / divisor], [y],
            marker="o", markersize=style.marker * 1.35, color=color,
            markerfacecolor=SURFACE if second_cycle else color,
            markeredgecolor=color if second_cycle else "none",
            markeredgewidth=style.line_width * 0.9 if second_cycle else 0.0,
            linestyle="None", zorder=4, clip_on=False,
        )
        labels.append(ax.annotate(
            name, (t / divisor, y),
            textcoords="offset points", xytext=(style.marker * 1.1, 0),
            ha="left", va="center", fontsize=style.value_label,
            color=label_color(color),
            fontweight="bold", zorder=5, annotation_clip=False,
        ))

    style_axes(ax, style)
    ax.set_xscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.{xdigits}f}"))
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.{ydigits}f}"))
    ax.tick_params(axis="both", labelsize=style.tick)

    if ylo > 0:
        draw_axis_break(ax, style, below=(yticks[0] - ylo) / (yhi - ylo) if yticks else 1.0)

    if missing:
        ax.text(
            0.975, 0.035, f"{', '.join(missing)}: not logged", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=style.context * 0.9, color=INK_MUTED,
        )

    spread_labels(ax, labels, style.value_label * 1.45)

    ax.text(
        0.5, 1.16 if subtitle else 1.08, title, transform=ax.transAxes,
        fontsize=style.panel_title, fontweight="bold", color=INK_PRIMARY,
        ha="center", va="baseline",
    )
    if subtitle:
        ax.text(
            0.5, 1.06, subtitle, transform=ax.transAxes,
            fontsize=style.subtitle, color=INK_SECONDARY,
            ha="center", va="baseline",
        )
    return unit


def build_tradeoff_figures(
    records,
    section: str,
    *,
    outdir: Path,
    grid: str | None,
    formats: list[str],
    style: Style,
    stem: str,
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
    written += save(fig, outdir, stem, formats)

    # --- 2) One standalone figure per metric --------------------------------- #
    for key, long_title, points, configs in panels:
        fig, ax = plt.subplots(figsize=style.single_size)
        fig.subplots_adjust(left=0.125, right=0.94, top=0.84, bottom=0.2)
        unit = draw_tradeoff_panel(
            ax, key, long_title, subtitle_text or grid_line(configs), points, style
        )
        add_x_label(fig, [ax], style, f"Training time ({TIME_UNIT_ABBREV.get(unit, unit)})")
        written += save(fig, outdir, f"{stem}_{key.lower()}-vs-tt", formats)

    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=Path, default=here / "tables.md")
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

    if not args.tables.exists():
        parser.error(f"tables file not found: {args.tables}")

    records = parse_tables(args.tables)
    if not records:
        parser.error(f"no metric tables found in {args.tables}")

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

    for split in ("dataset split", "model split"):
        outdir = args.outdir / split
        outdir.mkdir(parents=True, exist_ok=True)
        for section in sections:
            stem = next(k for k, v in SECTION_ALIASES.items() if v == section)
            print(f"{split} / {section}:")
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
                stem=stem,
                split=split,
                title_text=args.title,
                subtitle_text=args.subtitle,
            ):
                print(f"  wrote {path}")
        if args.section in ("all", "multivariate", "perchannel"):
            print(f"{split} / multivariate vs per-channel:")
            for path in build_overlay_figure(
                records,
                outdir=outdir,
                split=split,
                grid=args.grid,
                formats=formats,
                style=style,
                title_text=args.title,
                subtitle_text=args.subtitle,
            ):
                print(f"  wrote {path}")

    outdir = args.outdir / "time tradeoff"
    outdir.mkdir(parents=True, exist_ok=True)
    for section in sections:
        stem = next(k for k, v in SECTION_ALIASES.items() if v == section)
        print(f"time tradeoff / {section}:")
        for path in build_tradeoff_figures(
            records,
            section,
            outdir=outdir,
            grid=args.grid,
            formats=formats,
            style=style,
            stem=stem,
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
        outdir = args.outdir / "cluster"
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"cluster / {CLUSTER_SECTION}:")
        for path in build_figures(
            cluster_records,
            CLUSTER_SECTION,
            outdir=outdir,
            grid=args.grid,
            formats=formats,
            style=style,
            highlight=args.highlight or None,
            muted=set(args.mute),
            label_points=args.labels,
            stem="cluster",
            split="cluster",
            title_text=args.title,
            subtitle_text=args.subtitle,
        ):
            print(f"  wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
