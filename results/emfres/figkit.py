"""Shared furniture of the paradigm-comparison figures.

The paradigm, not the model, owns the colour: multivariate is Charcoal Blue
filled, per-channel the same hue hollow, the cluster pipelines Verdigris (a
diamond for the validation-chosen k, a small dot for k = 1). Models are rows
or facets, always named in text, so no figure asks the reader to tell seven
models apart by hue. Outcomes of a comparison use Verdigris (challenger
better) and Burnt Peach (baseline better), with a tie drawn hollow in muted
ink -- a shape difference as well as a hue one, since Verdigris and the muted
ink are too close under colour-vision deficiency to carry it alone. A figure
that colours by dataset (k-gain, cost) takes style.DATASET_COLORS, whose hues
are none of these, so no dataset reads as a paradigm or an outcome.

Type comes from style.rc_fonts (IBM Plex Sans); sizes from a style.Style
preset, "paper" (IEEEtran) by default.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.markers import MarkerStyle  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from . import paradigms as PG  # noqa: E402
from .chrome import frame_legend, save  # noqa: E402
from .naming import artifact_path  # noqa: E402
from .style import (  # noqa: E402
    AXIS_RULE, DATASET_COLORS, GRIDLINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY,
    MUTED_COLOR, SERIES_COLORS, STYLES, SURFACE, Style, label_color, rc_fonts,
)

VERDIGRIS, BURNT_PEACH, CHARCOAL = SERIES_COLORS[0], SERIES_COLORS[1], SERIES_COLORS[2]

# The figure family directory: figures/paradigm-compare/<emf|voip>/...
FAMILY_DIR = "paradigm-compare"

# pipe -> (colour, marker, filled, relative size, legend label)
PARADIGM = {
    PG.MV: (CHARCOAL, "o", True, 1.0, "Multivariate"),
    PG.PC: (CHARCOAL, "o", False, 1.0, "Per-channel"),
    PG.K1: (VERDIGRIS, "o", True, 0.5, "Cluster k=1"),
    PG.BEST: (VERDIGRIS, "D", True, 0.95, "Cluster, best k"),
}

OUTCOME = {
    "challenger": (VERDIGRIS, True),
    "baseline": (BURNT_PEACH, True),
    "tie": (INK_SECONDARY, False),
    "pending": (INK_MUTED, False),
}

# Faded: a pipeline still short of seeds (or a sweep still short of k values).
PARTIAL_ALPHA = 0.4

# The tie band's fill: one step darker than the gridlines, so it reads as a
# region, not a rule.
BAND_FILL = "#dfe6ea"


def diverging_cmap() -> LinearSegmentedColormap:
    """Verdigris (challenger better) -> neutral mist -> Burnt Peach (baseline better)."""
    return LinearSegmentedColormap.from_list(
        "paradigm-diverging", [VERDIGRIS, "#9fd0c9", "#f3f5f6", "#f4b8a6", BURNT_PEACH])


def setup(style_name: str = "paper") -> Style:
    style = STYLES[style_name]
    plt.rcParams.update({**rc_fonts("sans"), "font.size": style.tick,
                         "axes.titlesize": style.panel_title})
    return style


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #

def _er():
    results = str(Path(__file__).resolve().parent.parent)
    if results not in sys.path:
        sys.path.insert(0, results)
    import extract_results as er
    return er


def model_name(model: str) -> str:
    """``RF_EMF_Transformer`` -> ``Transformer``: the display name without its year."""
    return re.sub(r"\s*\(\d{4}\)\s*$", "", _er().model_label(model)).strip()


def dataset_name(dataset: str) -> str:
    """``emf_2023_university_5_with_ptv`` -> ``EMF2023``."""
    return re.sub(r"\s*\(.*\)\s*$", "", _er().dataset_label(dataset)).strip() or dataset


def ranks() -> dict:
    er = _er()
    return {"dataset_rank": er.dataset_rank, "model_rank": er.model_rank}


def window_label(window: tuple) -> str:
    return f"L{window[0]}/O{window[1]}"


# --------------------------------------------------------------------------- #
# Marks
# --------------------------------------------------------------------------- #

def mark_kwargs(pipe: str, style: Style, *, alpha: float = 1.0, scale: float = 1.0) -> dict:
    color, marker, filled, size, _ = PARADIGM[pipe]
    return dict(marker=marker, markersize=style.marker * size * scale, linestyle="None",
                markerfacecolor=color if filled else SURFACE, markeredgecolor=color,
                markeredgewidth=style.line_width * 0.5, alpha=alpha)


def dataset_colors(datasets) -> dict:
    """{dataset: colour} in the given order, from the dataset palette.

    Never the paradigm or outcome hues, so a figure that colours by dataset
    keeps them free for what they mean elsewhere.
    """
    return {ds: DATASET_COLORS[i % len(DATASET_COLORS)] for i, ds in enumerate(datasets)}


def paradigm_handles(style: Style, pipes, *, partial: bool = False, seeds: bool = True) -> list:
    handles = [Line2D([], [], label=PARADIGM[p][4], **mark_kwargs(p, style)) for p in pipes]
    if seeds:
        handles.append(Line2D([], [], marker="|", linestyle="None", color=INK_SECONDARY,
                              markersize=style.marker * 1.1, markeredgewidth=0.9,
                              label="Seed"))
    if partial:
        handles.append(Line2D([], [], label="Still training",
                              **mark_kwargs(PG.BEST, style, alpha=PARTIAL_ALPHA)))
    return handles


def outcome_handles(style: Style, band: bool = True, pending: bool = False) -> list:
    out = []
    for key, label in (("challenger", "Challenger better"), ("tie", "Tie"),
                       ("baseline", "Baseline better")) + ((("pending", "Pending"),) if pending else ()):
        color, filled = OUTCOME[key]
        out.append(Line2D([], [], marker="o", linestyle="None", markersize=style.marker,
                          markerfacecolor=color if filled else SURFACE, markeredgecolor=color,
                          markeredgewidth=style.line_width * 0.5, label=label))
    if band:
        out.append(Patch(facecolor=BAND_FILL, edgecolor="none", label="Tie band (seed sd)"))
    return out


def seed_ticks(ax, xs, y, color, style: Style, alpha: float = 0.55) -> None:
    ax.plot(list(xs), [y] * len(list(xs)), marker="|", linestyle="None", color=color,
            markersize=style.marker * 1.05, markeredgewidth=0.8, alpha=alpha, zorder=3)


# --------------------------------------------------------------------------- #
# Axes and figure furniture
# --------------------------------------------------------------------------- #

def style_axes(ax, style: Style, grid: str = "x") -> None:
    """Hairline spines and grid, one shade off the surface."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_RULE)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=AXIS_RULE, length=2.5, width=0.6, labelcolor=INK_PRIMARY,
                   labelsize=style.tick)
    if grid:
        ax.grid(axis=grid, color=GRIDLINE, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def panel_title(ax, text: str, style: Style, sub: str | None = None, pad: float = 4) -> None:
    ax.set_title(text, fontsize=style.panel_title, fontweight="bold", color=INK_PRIMARY,
                 pad=pad + (style.context * 1.35 if sub is not None else 0))
    if sub:
        ax.annotate(sub, (0.5, 1.0), xycoords="axes fraction", xytext=(0, pad),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=style.context, color=INK_SECONDARY)


def title(fig, text: str, style: Style, sub: str | None = None, y: float = 0.995) -> None:
    """Bold title, muted qualifier after it, on one line at the top left."""
    t = fig.text(0.01, y, text, ha="left", va="top", fontsize=style.title * 0.95,
                 fontweight="bold", color=INK_PRIMARY)
    if sub:
        fig.canvas.draw()
        box = t.get_window_extent(fig.canvas.get_renderer())
        x = box.x1 / fig.bbox.width + 0.012
        fig.text(x, y - 0.002, sub, ha="left", va="top", fontsize=style.subtitle,
                 color=INK_SECONDARY)


def legend(fig, handles, style: Style, y: float, ncol: int | None = None):
    """A framed legend row centred at figure height ``y`` (its top)."""
    return frame_legend(fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
        ncol=len(handles) if ncol is None else ncol, frameon=True, fancybox=True,
        fontsize=style.legend, handlelength=1.3, handletextpad=0.5, borderpad=0.45,
        columnspacing=1.3))


def xlabel(fig, axes, text: str, style: Style) -> None:
    """One x-axis label centred under a row of facets (their scales may differ)."""
    axes = [ax for ax in axes if ax.get_visible()]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(renderer) for ax in axes]
    x0 = min(ax.get_position().x0 for ax in axes)
    x1 = max(ax.get_position().x1 for ax in axes)
    y = min(b.y0 for b in boxes) / fig.bbox.height
    fig.text((x0 + x1) / 2, y - 0.01, text, ha="center", va="top",
             fontsize=style.axis_label, color=INK_SECONDARY)


def layout(fig, style: Style, *, top: float, bottom: float, left: float = 0.13,
           right: float = 0.99, **kw) -> None:
    """subplots_adjust with the top and bottom margins given in inches."""
    h = fig.get_size_inches()[1]
    fig.subplots_adjust(left=left, right=right, top=1 - top / h, bottom=bottom / h, **kw)


def footnote(fig, text: str, style: Style, y: float = 0.0) -> None:
    fig.text(0.01, y, text, ha="left", va="bottom", fontsize=style.footnote, color=INK_MUTED)


def write(fig, root: Path, *, group: str | None, subject: str, configs=None,
          metric: str | None = None, family: str = FAMILY_DIR, formats=("png",),
          view: str | None = None) -> list:
    directory, stem = artifact_path(root, family, group=group, subject=subject,
                                    configs=configs, metric=metric, view=view)
    directory.mkdir(parents=True, exist_ok=True)
    return save(fig, directory, stem, list(formats))


def pad_range(values, frac: float = 0.06, right: float = 0.0) -> tuple:
    lo, hi = min(values), max(values)
    span = (hi - lo) or abs(hi) * 0.05 or 1.0
    return lo - frac * span, hi + (frac + right) * span


# A label's box: opaque, so a rule or gridline under it never crosses the text.
LABEL_BOX = dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none")


def fit_texts(ax, texts, pad: float = 2.0, rounds: int = 4) -> bool:
    """Widen a linear x axis until every text in ``texts`` lies inside it.

    Labels hang off data points by a fixed offset in points, so how much room
    they need depends on the final axes width: call this after the figure's
    layout. Each round widens by the overhang at the current scale, which
    over-covers it slightly as the scale compresses, so a few rounds settle.
    Returns whether the axis was widened.
    """
    texts = [t for t in texts if t is not None]
    if not texts:
        return False
    renderer = ax.figure.canvas.get_renderer()
    widened = False
    for _ in range(rounds):
        box = ax.get_window_extent(renderer)
        ext = [t.get_window_extent(renderer) for t in texts]
        over_l = max(0.0, box.x0 + pad - min(e.x0 for e in ext))
        over_r = max(0.0, max(e.x1 for e in ext) - (box.x1 - pad))
        if over_l < 0.5 and over_r < 0.5:
            break
        lo, hi = ax.get_xlim()
        per_px = (hi - lo) / box.width
        ax.set_xlim(lo - over_l * per_px, hi + over_r * per_px)
        widened = True
    return widened


def _obstacles(ax, renderer, ignore=(), rules: bool = True) -> list:
    """Display-space boxes (x0, y0, x1, y1) a label must not cover on ``ax``.

    Every marker, as drawn (a seed tick is as narrow as its stroke, a
    diamond wider than a dot of its size), and every straight rule (a line
    whose points share one x or one y: axvline, axhline, a hairline between
    two marks) except those in ``ignore``; with ``rules`` False, no vertical
    rule. Gridlines are not ax.lines and are left to the label's opaque box.
    """
    px = ax.figure.dpi / 72
    boxes = []
    for line in ax.lines:
        if line in ignore or not line.get_visible():
            continue
        pts = line.get_transform().transform(line.get_xydata())
        if line.get_marker() not in (None, "None", "", " "):
            ms = MarkerStyle(line.get_marker())
            e = ms.get_path().get_extents(ms.get_transform())
            size = line.get_markersize() * px
            edge = line.get_markeredgewidth() * px / 2
            boxes += [(x + e.x0 * size - edge, y + e.y0 * size - edge,
                       x + e.x1 * size + edge, y + e.y1 * size + edge) for x, y in pts]
        elif len(pts) >= 2 and line.get_linestyle() not in ("None", "", " "):
            (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
            w = line.get_linewidth() * px / 2
            if (x1 - x0 < 0.5 and rules) or y1 - y0 < 0.5:
                boxes.append((x0 - w, y0 - w, x1 + w, y1 + w))
    return boxes


def place_clear(ax, text: str, candidates, pad: float = 1.5, ignore=(), **kw):
    """Annotate at the first candidate spot whose text box clears the marks.

    ``candidates`` are (xy, xycoords, xytext offset in points, ha, va). The
    text must fit inside the axes and cover no marker, no straight rule
    (figkit._obstacles; ``ignore`` names rules it may touch, such as the one
    it labels) and no text already on the axes. Where no candidate is clear
    the first is kept: its opaque box still keeps what it covers off the
    text. Call after the layout, the axis limits and the axes' other labels
    are final.
    """
    renderer = ax.figure.canvas.get_renderer()
    marks = _obstacles(ax, renderer, ignore)
    for other in ax.texts:
        if other.get_visible() and other.get_text():
            e = other.get_window_extent(renderer)
            marks.append((e.x0, e.y0, e.x1, e.y1))
    frame = ax.get_window_extent(renderer)
    kept = None
    for xy, coords, offset, ha, va in candidates:
        t = ax.annotate(text, xy, xycoords=coords, xytext=offset, textcoords="offset points",
                        ha=ha, va=va, bbox=LABEL_BOX, **kw)
        e = t.get_window_extent(renderer)
        inside = frame.x0 <= e.x0 and e.x1 <= frame.x1 and frame.y0 <= e.y0 and e.y1 <= frame.y1
        clear = inside and not any(e.x0 - pad < b[2] and b[0] < e.x1 + pad and
                                   e.y0 - pad < b[3] and b[1] < e.y1 + pad for b in marks)
        if clear:
            if kept is not None:
                kept.remove()
            return t
        if kept is None:
            kept = t
        else:
            t.remove()
    return kept


# Where a lane label draws: over the seed ticks (3) and the hairline (4) it
# steps clear of, under the means (5) and diamonds (6), so a mark always
# stays whole even where no clear spot exists.
LABEL_Z = 4.5


@dataclass
class LaneLabel:
    """A text written in a dot plot's lane, beside the marks it names.

    ``lo``..``hi`` (data x) spans those marks and ``at`` is the one it names
    (a diamond); the text starts ``gap`` points past ``hi`` on the right
    (side +1) or ``lo`` on the left (side -1). ``sides`` are the sides it may
    take, preferred first; with ``prefer`` False the nearer one wins. ``own``
    are the marks lo..hi spans and the hairline between them: its starting
    spot lies past them by construction, so they are not obstacles. Every
    other mark is, the seed ticks of the marks it names included.
    """
    text: object
    y: float
    lo: float
    hi: float
    at: float
    sides: tuple = (1,)
    prefer: bool = True
    own: tuple = ()
    gap: float = 4.5


def lane_label(ax, text: str, y: float, lo: float, hi: float, at: float, *,
               sides=(1,), prefer: bool = True, own=(), gap: float = 4.5,
               **kw) -> LaneLabel:
    """Annotate ``text`` at its first spot; settle_labels moves it clear of the marks."""
    side = sides[0]
    t = ax.annotate(text, (hi if side > 0 else lo, y), xytext=(side * gap, 0),
                    textcoords="offset points", ha="left" if side > 0 else "right",
                    va="center", bbox=LABEL_BOX, **{"zorder": LABEL_Z, **kw})
    return LaneLabel(t, y, lo, hi, at, tuple(sides), prefer, tuple(own), gap)


def _place_lane_labels(ax, labels, near: float) -> None:
    renderer = ax.figure.canvas.get_renderer()
    px = ax.figure.dpi / 72
    to_disp = ax.transData.transform
    placed: list = []
    for lab in labels:
        t = lab.text
        marks = _obstacles(ax, renderer, ignore=lab.own, rules=False) + placed
        e = t.get_window_extent(renderer)
        w, h = e.width, e.height
        # The opaque box's margin (LABEL_BOX pad): what it would cut away.
        pad = 0.1 * t.get_fontsize() * px
        step = pad + 1.5 * px
        xat, yd = to_disp((lab.at, lab.y))
        y0, y1 = yd - h / 2, yd + h / 2
        spots = []
        for side in lab.sides:
            x = to_disp((lab.hi if side > 0 else lab.lo, lab.y))[0] + side * lab.gap * px
            for _ in range(200):
                x0, x1 = (x, x + w) if side > 0 else (x - w, x)
                hits = [b for b in marks if x0 - pad < b[2] and b[0] < x1 + pad and
                        y0 - pad < b[3] and b[1] < y1 + pad]
                if not hits:
                    break
                x = max(b[2] for b in hits) + step if side > 0 else min(b[0] for b in hits) - step
            spots.append((abs(x - xat), side, x))
        if lab.prefer and spots[0][0] <= near * px:
            _, side, x = spots[0]
        else:
            _, side, x = min(spots, key=lambda s: s[0])
        t.xy = (ax.transData.inverted().transform((x - side * lab.gap * px, yd))[0], lab.y)
        t.xyann = (side * lab.gap, 0)
        t.set_horizontalalignment("left" if side > 0 else "right")
        x0, x1 = (x, x + w) if side > 0 else (x - w, x)
        placed.append((x0 - pad, y0 - pad, x1 + pad, y1 + pad))


def settle_labels(ax, labels, near: float, rounds: int = 4) -> None:
    """Put every lane label where its box covers no mark, and fit the axis to them.

    Each label starts past the marks it names on each side it may take and,
    while its box (with the opaque margin) meets a marker, a hairline or a
    label already placed (_obstacles), steps past the farthest of those; a
    vertical reference rule it crosses runs under its opaque box, as a
    gridline does. It keeps its preferred side unless that leaves it more
    than ``near`` points from the mark it names; then, or with ``prefer``
    False, the nearer side wins. The axis is then widened to hold the labels
    (fit_texts), which moves the marks under them, so the two alternate
    until the axis stays put. Call after the figure's layout and axis limits.
    """
    labels = [lab for lab in labels if lab is not None]
    if not labels:
        return
    for _ in range(rounds):
        _place_lane_labels(ax, labels, near)
        if not fit_texts(ax, [lab.text for lab in labels]):
            return
    _place_lane_labels(ax, labels, near)


def footnotes(fig, notes, style: Style) -> None:
    """The figure's notes on one line at the bottom left, if it has any."""
    notes = [n for n in notes if n]
    if notes:
        footnote(fig, "  ·  ".join(notes), style)


__all__ = [
    "AXIS_RULE", "BAND_FILL", "BURNT_PEACH", "CHARCOAL", "DATASET_COLORS", "FAMILY_DIR",
    "GRIDLINE", "INK_MUTED", "INK_PRIMARY", "INK_SECONDARY", "LABEL_BOX", "LABEL_Z",
    "MUTED_COLOR", "OUTCOME", "PARADIGM", "PARTIAL_ALPHA", "SERIES_COLORS", "SURFACE",
    "VERDIGRIS", "LaneLabel", "dataset_colors", "dataset_name", "diverging_cmap",
    "fit_texts", "footnote", "footnotes", "label_color", "lane_label", "legend",
    "mark_kwargs", "model_name", "layout", "outcome_handles", "pad_range", "panel_title",
    "paradigm_handles", "place_clear", "ranks", "seed_ticks", "settle_labels", "setup",
    "style_axes", "title", "window_label", "write", "xlabel",
]
