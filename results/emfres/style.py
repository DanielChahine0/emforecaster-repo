"""Design tokens, typography, and axis scales.

Every figure in results/ reads its palette, its Style preset, and its tick
ladder from here, so the look is named in exactly one place.

Moved verbatim out of plot_metrics.py (Phase 1 of the results/ restructure).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib import colors as mcolors

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
