"""The two drawing primitives every figure family is built from.

``draw_panel`` renders a categorical line panel; ``draw_scatter_panel`` renders
a cost/accuracy scatter. Both take their series colouring through ``styler`` so
a caller can pair series by hue without reaching in and rebinding a module
global — which is what plot_L_ladder used to do.

Moved verbatim out of plot_metrics.py (Phase 1 of the results/ restructure).
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

from .chrome import (
    add_legend, densify, dot_handles, draw_axis_break, frame_legend,
    place_legend, spread_labels, style_axes,
)
from .style import (
    AXIS_RULE, GRIDLINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY, LADDER,
    LEGEND_EDGE, LEGEND_FILL, MUTED_COLOR, SERIES_COLORS, SURFACE,
    TIME_UNIT_ABBREV, Style, digits_for_step, label_color, linear_ticks,
    log_ticks, series_style, time_unit,
)

@dataclass
class Panel:
    key: str
    title: str
    is_time: bool
    x_labels: list[str]  # display-ready column names
    series: list[tuple[str, list[float]]]  # (display label, value per column)
    configs: list[tuple[int, int]]



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
