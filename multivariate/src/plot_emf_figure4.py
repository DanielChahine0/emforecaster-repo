#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EMFusion Figure-4 style EMF (V/m) time-trend plots.

This is a standalone editable copy that does not modify the original
parse_continuous_periods_window.py (which reads raw .txt and only makes
fixed zoom-window plots). Edit the configuration block below to change the
dataset, periods, colours, and so on.

It reproduces the layout of Fig. 4 of the EMFusion paper and extends it: for
each breakdown the EMF level (V/m) is plotted as a time trend, shown both over
the whole continuous measurement period and over a roughly 2-week zoom.

Each breakdown produces a full-period panel and a zoom panel:
  1. Operators: W3, Iliad, TIM, VF
  2. Technologies: 2G, 3G, 4G, 5G
  3. Frequencies: every individual frequency band available, coloured by
     frequency with a colour bar instead of a 29-entry legend

It writes into result/figure4_<dataset>_<timestamp>/plots/ one combined stacked
figure (emf_figure4_<dataset>.png and .pdf) and six separate per-panel files
(emf_<breakdown>_<dataset>_<full|zoom>.png and .pdf).

The processed CSVs in datasets/processed/ are already E-field values in V/m
(per frequency), so no dBm to V/m conversion is done here. EMF_2023_university
also ships the EMF_2G/3G/4G/5G, operator, and EMF_Total columns precomputed; we
use them when present and otherwise compute them on the fly by RSS combining.
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Configuration (edit the values below).

# Project root (multivariate/) anchored to this file, so the script works
# regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Name of the timestamp column in the CSV.
TIMESTAMP_COL = "Timestamp"

# Datasets to plot. Each entry is a dict:
#   csv          path to a processed CSV
#   label        short label used in the plot titles / log output
#   folder       output sub-folder under result/EMF levels/
#   full_period  {"start", "end"} for the full-period panels, or None to use
#                the dataset's own min/max timestamp
#   zoom_window  {"start", "end"} for the 2-week zoom, or None to centre it on
#                the dataset's temporal midpoint
# The 2023 university entry keeps its curated continuous window (the longest
# continuous stretch, matching EMFusion Fig. 4); the UH/ED datasets auto-derive
# their windows from their own date ranges.
DATASETS = [
    {
        "csv": PROJECT_ROOT / "datasets/processed/EMF_2023_university_5_with_PTV.csv",
        "label": "EMF_2023_university",
        "folder": "EMF2023",
        "full_period": {
            "start": dt.datetime(2023, 7, 26, 16, 49, 2),
            "end":   dt.datetime(2024, 3, 11, 5, 8, 28),
        },
        "zoom_window": {
            "start": dt.datetime(2023, 11, 10, 0, 0, 0),
            "end":   dt.datetime(2023, 11, 25, 0, 0, 0),
        },
    },
    {"csv": PROJECT_ROOT / "datasets/processed/UH2022.csv", "label": "UH_2022",
     "folder": "UH2022", "full_period": None, "zoom_window": None},
    {"csv": PROJECT_ROOT / "datasets/processed/ED2022.csv", "label": "ED_2022",
     "folder": "ED2022", "full_period": None, "zoom_window": None},
    {"csv": PROJECT_ROOT / "datasets/processed/UH2024.csv", "label": "UH_2024",
     "folder": "UH2024", "full_period": None, "zoom_window": None},
    {"csv": PROJECT_ROOT / "datasets/processed/ED2024.csv", "label": "ED_2024",
     "folder": "ED2024", "full_period": None, "zoom_window": None},
]

# Length of the auto-derived zoom window, in days.
ZOOM_DAYS = 14

# Set per dataset inside main(); the plotting helpers read these when drawing
# the zoom markers and naming the output files.
DATASET_LABEL = DATASETS[0]["label"]
FULL_PERIOD = DATASETS[0]["full_period"]
ZOOM_WINDOW = DATASETS[0]["zoom_window"]

# Drop points outside +/- N std (per-series) before plotting. Fig. 4 shows the
# raw spikes, so this is off by default; set to a number (e.g. 4) to clean up.
REMOVE_OUTLIERS_STD = None

# Per-frequency extreme-spike removal for the "all frequencies" panels only.
# Some raw per-frequency columns contain isolated corrupted samples (e.g. a
# single 119 V/m reading at 1835 MHz) that flatten the whole panel. Any value
# above FREQ_CLIP_FACTOR x the FREQ_CLIP_QUANTILE of the bulk data is treated as
# an artifact and dropped (NaN) before plotting. Set FREQ_CLIP_QUANTILE = None
# to disable. Operator/technology panels use the clean precomputed aggregations
# and are never clipped.
FREQ_CLIP_QUANTILE = 0.999
FREQ_CLIP_FACTOR = 3.0

# How to render the "all frequencies" panels:
#   "colorbar" colours each line by frequency and adds a Frequency (MHz) colorbar
#   "legend" draws a (crowded) multi-column legend listing every frequency
FREQ_LEGEND_MODE = "colorbar"

# Line styling per panel kind.
STYLE = {
    "full": dict(lw=0.4, alpha=0.9),   # dense full-period overview
    "zoom": dict(lw=0.9, alpha=0.9),   # ~2 weeks, fewer points
}
FREQ_STYLE = {
    "full": dict(lw=0.35, alpha=0.85),
    "zoom": dict(lw=0.7, alpha=0.85),
}

# Fig. 4 colours (blue / orange / green / red) keyed by series, in legend order.
OPERATOR_ORDER = ["W3", "Iliad", "TIM", "VF"]
TECH_ORDER = ["2G", "3G", "4G", "5G"]
SERIES_COLORS = {
    "W3": "#1f77b4", "Iliad": "#ff7f0e", "TIM": "#2ca02c", "VF": "#d62728",
    "2G": "#1f77b4", "3G": "#ff7f0e", "4G": "#2ca02c", "5G": "#d62728",
}

# Colour for the dashed vertical lines marking the zoom window on full panels.
ZOOM_MARKER_COLOR = "#1f3b5c"

# Colormap for the all-frequencies panels.
FREQ_CMAP = "turbo"

# Band maps, used as a fallback when a dataset lacks EMF_* columns (the chosen
# EMF_2023_university CSV already has them). Same mapping as
# parse_continuous_periods_window.py for the 2024 (decimal) frequency set.
TECH_BANDS = {
    "2G": [935.0, 945.0, 955.0],
    "3G": [2120.0, 2137.5, 2150.0, 2162.5],
    "4G": [763.0, 773.0, 783.0, 796.0, 806.0, 816.0,
           1835.0, 1850.0, 1870.0,
           2585.0, 2635.0, 2647.5, 2662.5, 2680.0],
    "5G": [3449.0, 3469.0, 3489.0, 3547.5, 3589.0, 3630.0, 3680.0, 3760.0],
}
OPERATOR_BANDS = {
    "Iliad": [763.0, 1835.0, 2150.0, 2635.0, 3630.0],
    "TIM":   [773.0, 806.0, 935.0, 2137.5, 2662.5, 3760.0, 3469.0],
    "VF":    [783.0, 816.0, 945.0, 1870.0, 2162.5, 2647.5, 3680.0],
    "W3":    [796.0, 955.0, 1850.0, 2120.0, 2585.0, 2680.0, 3610.0],
}

# Global font, matching the project's other figures.
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
mpl.rcParams["font.size"] = 16


# Data loading and preparation
def _parse_timestamps(series: pd.Series) -> pd.Series:
    """Parse timestamps, tolerating ISO and day-first (DD/MM/YYYY) formats."""
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().mean() > 0.5:  # mostly failed, retry with day-first
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return parsed


def _freq_columns(df: pd.DataFrame) -> list:
    """Return columns whose name parses as a float (i.e. a frequency in MHz)."""
    freqs = []
    for col in df.columns:
        try:
            float(col)
        except (TypeError, ValueError):
            continue
        freqs.append(col)
    # Sort by numeric frequency value.
    return sorted(freqs, key=lambda c: float(c))


def _rss(df: pd.DataFrame, cols: list) -> pd.Series:
    """Root-sum-square combine of E-field columns (V/m)."""
    val = (df[cols] ** 2).sum(axis=1, min_count=1).pow(0.5)
    return val.where(val != 0, np.nan)


def load_dataset(csv_path: Path):
    """Load a processed EMF CSV.

    Returns (df, freq_cols, tech_cols, op_cols) where df is indexed by the
    timestamp and freq columns are floats (V/m). Technology/operator
    aggregation columns are reused if present, else computed via RSS.
    """
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    if TIMESTAMP_COL not in df.columns:
        # Some CSVs use 'date' instead of 'Timestamp'.
        alt = next((c for c in ("Timestamp", "date", "Date") if c in df.columns), None)
        if alt is None:
            raise ValueError(f"No timestamp column found in {csv_path.name}")
        df = df.rename(columns={alt: TIMESTAMP_COL})

    df[TIMESTAMP_COL] = _parse_timestamps(df[TIMESTAMP_COL])
    df = df.dropna(subset=[TIMESTAMP_COL]).set_index(TIMESTAMP_COL).sort_index()

    # Coerce frequency columns to numeric (V/m).
    freq_cols = _freq_columns(df)
    for c in freq_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Technology aggregations: reuse EMF_<tech> if present, else compute.
    tech_cols = []
    for tech in TECH_ORDER:
        col = f"EMF_{tech}"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            present = [c for c in freq_cols if float(c) in TECH_BANDS.get(tech, [])]
            if not present:
                print(f"  No frequencies for {tech}; skipping")
                continue
            df[col] = _rss(df, present)
        tech_cols.append(col)

    # Operator aggregations: reuse EMF_<op> if present, else compute.
    op_cols = []
    for op in OPERATOR_ORDER:
        col = f"EMF_{op}"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            present = [c for c in freq_cols if float(c) in OPERATOR_BANDS.get(op, [])]
            if not present:
                print(f"  No frequencies for {op}; skipping")
                continue
            df[col] = _rss(df, present)
        op_cols.append(col)

    print(f"  {len(df):,} rows | {len(freq_cols)} freqs | "
          f"{len(tech_cols)} tech | {len(op_cols)} operator columns")
    return df, freq_cols, tech_cols, op_cols


def slice_period(df: pd.DataFrame, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    return df[(df.index >= start) & (df.index <= end)]


def auto_full_period(df: pd.DataFrame) -> dict:
    """Full period spanning the dataset's own first and last timestamp."""
    return {"start": df.index.min().to_pydatetime(),
            "end":   df.index.max().to_pydatetime()}


def auto_zoom_window(df: pd.DataFrame, days: int = ZOOM_DAYS) -> dict:
    """Center a `days`-long zoom on the dataset's temporal midpoint.

    The window is snapped to the nearest dense window only when the midpoint
    falls in a data gap, so the zoom panel stays populated for every dataset.
    """
    idx = df.index
    mid = pd.Timestamp(idx.min() + (idx.max() - idx.min()) / 2).normalize()
    span = pd.Timedelta(days=days)
    mid_start = mid - pd.Timedelta(days=days // 2)

    daily = pd.Series(1, index=idx).resample("1D").sum()
    roll = daily.rolling(f"{days}D").sum()
    best = roll.max() if roll.notna().any() else 0
    centered = ((idx >= mid_start) & (idx < mid_start + span)).sum()

    if best and centered < 0.6 * best:
        # Midpoint sits in a gap: snap to the densest window nearest the middle.
        dense_ends = roll[roll >= 0.95 * best].index
        end_day = min(dense_ends, key=lambda d: abs(d - (mid_start + span)))
        start = pd.Timestamp(end_day).normalize() - pd.Timedelta(days=days - 1)
    else:
        start = mid_start

    return {"start": start.to_pydatetime(),
            "end":   (start + span).to_pydatetime()}


def _drop_outliers(df: pd.DataFrame, cols: list, n_std) -> pd.DataFrame:
    if n_std is None:
        return df
    mask = pd.Series(True, index=df.index)
    for col in cols:
        if col in df:
            m, s = df[col].mean(), df[col].std()
            mask &= df[col].between(m - n_std * s, m + n_std * s)
    return df[mask]


def _clip_extreme_freq_spikes(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """NaN out isolated corrupted per-frequency samples (artifact spikes).

    Threshold = FREQ_CLIP_FACTOR x the FREQ_CLIP_QUANTILE of all plotted
    frequency values. Returns a copy; only affects the frequency panels.
    """
    if FREQ_CLIP_QUANTILE is None or not cols:
        return df
    values = df[cols].to_numpy(dtype=float)
    if not np.isfinite(values).any():
        return df
    threshold = np.nanquantile(values, FREQ_CLIP_QUANTILE) * FREQ_CLIP_FACTOR
    out = df.copy()
    out[cols] = out[cols].where(out[cols] <= threshold)
    return out


# Plotting helpers
def _format_time_axis(ax, span: str):
    """Month ticks for the full period, 2-day ticks for the zoom."""
    if span == "full":
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(0)


def plot_discrete(ax, df, cols, order, span, draw_zoom_markers):
    """Plot a small set of named series (operators or technologies)."""
    dfc = _drop_outliers(df, cols, REMOVE_OUTLIERS_STD)
    style = STYLE[span]
    for name in order:
        col = f"EMF_{name}"
        if col not in dfc.columns:
            continue
        ax.plot(dfc.index, dfc[col], color=SERIES_COLORS.get(name),
                label=name, **style)

    if draw_zoom_markers:
        for x in (ZOOM_WINDOW["start"], ZOOM_WINDOW["end"]):
            ax.axvline(x, color=ZOOM_MARKER_COLOR, ls="--", lw=2.5, alpha=0.9)

    ax.set_xlabel("Timestamp", fontsize=16, fontweight="bold")
    ax.set_ylabel("EMF (V/m)", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0),
              ncol=len(order), frameon=False, fontsize=14,
              prop={"weight": "bold"})
    _format_time_axis(ax, span)


def plot_frequencies(ax, df, freq_cols, span, fig=None):
    """Plot every individual frequency line, coloured by frequency."""
    dfc = _drop_outliers(df, freq_cols, REMOVE_OUTLIERS_STD)
    dfc = _clip_extreme_freq_spikes(dfc, freq_cols)
    style = FREQ_STYLE[span]
    freqs = [float(c) for c in freq_cols]
    fmin, fmax = min(freqs), max(freqs)
    norm = Normalize(vmin=fmin, vmax=fmax)
    cmap = mpl.colormaps[FREQ_CMAP]

    for c in freq_cols:
        ax.plot(dfc.index, dfc[c], color=cmap(norm(float(c))), **style)

    if span == "full":
        for x in (ZOOM_WINDOW["start"], ZOOM_WINDOW["end"]):
            ax.axvline(x, color=ZOOM_MARKER_COLOR, ls="--", lw=2.5, alpha=0.9)

    ax.set_xlabel("Timestamp", fontsize=16, fontweight="bold")
    ax.set_ylabel("EMF (V/m)", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.grid(True, alpha=0.3)
    _format_time_axis(ax, span)

    if FREQ_LEGEND_MODE == "legend":
        ax.legend([str(c) for c in freq_cols], loc="upper center",
                  bbox_to_anchor=(0.5, 1.0), ncol=8, frameon=False, fontsize=7)
    else:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        target = fig if fig is not None else ax.figure
        cb = target.colorbar(sm, ax=ax, pad=0.01, fraction=0.025)
        cb.set_label("Frequency (MHz)", fontsize=13, fontweight="bold")


# Figure builders.

# Panel spec: (breakdown, span). Mirrors Fig. 4 (operator/tech, full+zoom)
# plus an all-frequencies pair.
PANELS = [
    ("operator", "full"), ("operator", "zoom"),
    ("technology", "full"), ("technology", "zoom"),
    ("frequency", "full"), ("frequency", "zoom"),
]

# Subset for the operators + technologies figure (no frequency panels).
OPERATOR_TECH_PANELS = [
    ("operator", "full"), ("operator", "zoom"),
    ("technology", "full"), ("technology", "zoom"),
]

# Output file name (the view) for each standalone panel. The dataset and the
# figure type are conveyed by the containing folder, not the file name.
PANEL_FILENAMES = {
    ("operator", "full"):    "Operators (full period)",
    ("operator", "zoom"):    "Operators (2 week zoom)",
    ("technology", "full"):  "Technologies (full period)",
    ("technology", "zoom"):  "Technologies (2 week zoom)",
    ("frequency", "full"):   "All frequencies (full period)",
    ("frequency", "zoom"):   "All frequencies (2 week zoom)",
}


def _panel_data(df_full, df_zoom, span):
    return df_full if span == "full" else df_zoom


def _draw_panel(ax, breakdown, span, data, freq_cols, tech_cols, op_cols, fig=None):
    if breakdown == "operator":
        plot_discrete(ax, data, op_cols, OPERATOR_ORDER, span,
                      draw_zoom_markers=(span == "full"))
    elif breakdown == "technology":
        plot_discrete(ax, data, tech_cols, TECH_ORDER, span,
                      draw_zoom_markers=(span == "full"))
    else:  # frequency
        plot_frequencies(ax, data, freq_cols, span, fig=fig)


def build_stacked_figure(df_full, df_zoom, freq_cols, tech_cols, op_cols,
                         out_dir, panels, stem):
    """Stacked figure with one row per (breakdown, span) in `panels`, no suptitle."""
    titles = {
        ("operator", "full"):    "Operators (full period)",
        ("operator", "zoom"):    "Operators (2-week zoom)",
        ("technology", "full"):  "Technologies (full period)",
        ("technology", "zoom"):  "Technologies (2-week zoom)",
        ("frequency", "full"):   "All frequencies (full period)",
        ("frequency", "zoom"):   "All frequencies (2-week zoom)",
    }
    fig, axes = plt.subplots(len(panels), 1, figsize=(15, 3.7 * len(panels)))
    if len(panels) == 1:
        axes = [axes]
    for ax, (breakdown, span) in zip(axes, panels):
        data = _panel_data(df_full, df_zoom, span)
        _draw_panel(ax, breakdown, span, data, freq_cols, tech_cols, op_cols, fig=fig)
        ax.set_title(titles[(breakdown, span)], fontsize=15, fontweight="bold")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)

    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {png.name} / {pdf.name}")


def build_separate_figures(df_full, df_zoom, freq_cols, tech_cols, op_cols, out_dir):
    """One standalone figure per (breakdown, span)."""
    print("Building separate per-panel figures...")
    for breakdown, span in PANELS:
        data = _panel_data(df_full, df_zoom, span)
        fig, ax = plt.subplots(figsize=(16, 6))
        _draw_panel(ax, breakdown, span, data, freq_cols, tech_cols, op_cols, fig=fig)
        fig.tight_layout()

        stem = PANEL_FILENAMES[(breakdown, span)]
        fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.pdf", format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {stem}.png / .pdf")


def plot_dataset(ds: dict):
    """Build all Figure-4 style plots for a single dataset entry."""
    global DATASET_LABEL, FULL_PERIOD, ZOOM_WINDOW

    csv = ds["csv"]
    if not csv.exists():
        print(f"Dataset not found: {csv.resolve()}")
        return

    DATASET_LABEL = ds["label"]
    print(f"\n=== {DATASET_LABEL} ===")
    df, freq_cols, tech_cols, op_cols = load_dataset(csv)
    if df.empty:
        print("  no rows after timestamp parsing; skipping")
        return

    FULL_PERIOD = ds.get("full_period") or auto_full_period(df)
    ZOOM_WINDOW = ds.get("zoom_window") or auto_zoom_window(df)

    df_full = slice_period(df, FULL_PERIOD["start"], FULL_PERIOD["end"])
    df_zoom = slice_period(df, ZOOM_WINDOW["start"], ZOOM_WINDOW["end"])
    print(f"  full period : {FULL_PERIOD['start']:%Y-%m-%d} to "
          f"{FULL_PERIOD['end']:%Y-%m-%d}  ({len(df_full):,} pts)")
    print(f"  zoom window : {ZOOM_WINDOW['start']:%Y-%m-%d} to "
          f"{ZOOM_WINDOW['end']:%Y-%m-%d}  ({len(df_zoom):,} pts)")

    if df_full.empty and df_zoom.empty:
        print("  no data in the derived periods; skipping")
        return

    out_dir = PROJECT_ROOT / "src" / "result" / "EMF levels" / ds["folder"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  output: {out_dir}")

    build_stacked_figure(df_full, df_zoom, freq_cols, tech_cols, op_cols, out_dir,
                         PANELS, "Combined (operators, technologies, frequencies)")
    build_stacked_figure(df_full, df_zoom, freq_cols, tech_cols, op_cols, out_dir,
                         OPERATOR_TECH_PANELS, "Combined (operators, technologies)")
    build_separate_figures(df_full, df_zoom, freq_cols, tech_cols, op_cols, out_dir)


def main():
    print("EMFusion Figure-4 style EMF time-trend plots")
    for ds in DATASETS:
        plot_dataset(ds)
    print("\nDone.")


if __name__ == "__main__":
    main()
