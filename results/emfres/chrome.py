"""Figure furniture: axes styling, legends, labels, and saving.

Everything that dresses a figure but draws none of its data. Collecting it here
is what lets one framed-legend implementation serve every plotter instead of
the five near-copies this replaces.

Moved verbatim out of plot_metrics.py (Phase 1 of the results/ restructure).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

from .style import (
    AXIS_RULE, FONT_STACKS, GRIDLINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY,
    LADDER, LEGEND_EDGE, LEGEND_FILL, MUTED_COLOR, SERIES_COLORS, STYLES,
    SURFACE, TIME_UNIT_ABBREV, Style, digits_for_step, label_color,
    linear_ticks, log_ticks, rc_fonts, series_style, time_unit,
)

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


def frame_legend(legend):
    """The soft rounded box every legend in these figures sits in."""
    frame = legend.get_frame()
    frame.set_facecolor(LEGEND_FILL)
    frame.set_edgecolor(LEGEND_EDGE)
    frame.set_linewidth(1.0)
    frame.set_boxstyle("round,pad=0.5,rounding_size=0.8")
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)
    return legend


def add_legend(ax, handles, style: Style, loc: str, ncol: int = 1):
    return frame_legend(ax.legend(
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
    ))


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
