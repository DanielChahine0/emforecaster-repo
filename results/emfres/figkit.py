"""Shared furniture of the paradigm-comparison figures.

The paradigm, not the model, owns the colour: multivariate is Charcoal Blue
filled, per-channel the same hue hollow, the cluster pipelines Verdigris (a
diamond for the validation-chosen k, a small dot for k = 1). Models are rows
or facets, always named in text, so no figure asks the reader to tell seven
models apart by hue. Outcomes of a comparison use Verdigris (challenger
better) and Burnt Peach (baseline better), with a tie drawn hollow in muted
ink -- a shape difference as well as a hue one, since Verdigris and the muted
ink are too close under colour-vision deficiency to carry it alone.

Type comes from style.rc_fonts (IBM Plex Sans); sizes from a style.Style
preset, "paper" (IEEEtran) by default.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from . import paradigms as PG  # noqa: E402
from .chrome import frame_legend, save  # noqa: E402
from .naming import artifact_path  # noqa: E402
from .style import (  # noqa: E402
    AXIS_RULE, GRIDLINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY, MUTED_COLOR,
    SERIES_COLORS, STYLES, SURFACE, Style, label_color, rc_fonts,
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


__all__ = [
    "AXIS_RULE", "BAND_FILL", "BURNT_PEACH", "CHARCOAL", "FAMILY_DIR", "GRIDLINE",
    "INK_MUTED", "INK_PRIMARY", "INK_SECONDARY", "MUTED_COLOR", "OUTCOME", "PARADIGM",
    "PARTIAL_ALPHA", "SERIES_COLORS", "SURFACE", "VERDIGRIS", "dataset_name",
    "diverging_cmap", "footnote", "label_color", "legend", "mark_kwargs", "model_name",
    "layout", "outcome_handles", "pad_range", "panel_title", "paradigm_handles", "ranks",
    "seed_ticks", "setup", "style_axes", "title", "window_label", "write", "xlabel",
]
