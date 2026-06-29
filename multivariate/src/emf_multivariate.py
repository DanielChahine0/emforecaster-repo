#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
emf_multivariate.py - command line tool for the multivariate EMF analysis.

Sub-commands:
  all              run emf-levels then heatmaps
  emf-levels       EMF level (V/m) time-trend plots
  heatmaps         correlation heatmaps (frequency / technology / operator)
  normalize        rebuild the UH/ED processed CSVs in place

Data is read from data/multivariate/processed/ at the repo root.
Figures are written under multivariate/src/result/.
Run `python emf_multivariate.py <command> --help` for a command's options.
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# ======================================================================
# Shared configuration and helpers
# ======================================================================

# Paths anchored to this file, so commands work from any directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "multivariate" / "processed"  # input CSVs
RESULT_DIR = PROJECT_ROOT / "src" / "result"                       # output figures

# Timestamp column name in the CSVs.
TIMESTAMP_COL = "Timestamp"
TIMESTAMP_CANDIDATES = ("Timestamp", "date", "Date")

# The datasets. Each entry: csv path, label (shown in titles), output folder,
# and optional full_period / zoom_window for emf-levels (None = derive them from
# the data's own date range).
DATASETS = [
    {
        "csv": PROCESSED_DIR / "EMF_2023_university_5_with_PTV.csv",
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
    {"csv": PROCESSED_DIR / "UH2022.csv", "label": "UH_2022",
     "folder": "UH2022", "full_period": None, "zoom_window": None},
    {"csv": PROCESSED_DIR / "ED2022.csv", "label": "ED_2022",
     "folder": "ED2022", "full_period": None, "zoom_window": None},
    {"csv": PROCESSED_DIR / "UH2024.csv", "label": "UH_2024",
     "folder": "UH2024", "full_period": None, "zoom_window": None},
    {"csv": PROCESSED_DIR / "ED2024.csv", "label": "ED_2024",
     "folder": "ED2024", "full_period": None, "zoom_window": None},
]

# Plot order for technologies and operators.
TECH_ORDER = ["2G", "3G", "4G", "5G"]
OPERATOR_ORDER = ["W3", "Iliad", "TIM", "VF"]

# Which frequencies (MHz) belong to each technology and each operator. Used to
# build the EMF_* aggregations when a dataset does not already have them.
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

# Spike clipping. A per-frequency value above FREQ_CLIP_FACTOR x the
# FREQ_CLIP_QUANTILE of its data is treated as a bad sample and dropped (NaN).
# Set FREQ_CLIP_QUANTILE = None to turn it off. Aggregations are never clipped.
FREQ_CLIP_QUANTILE = 0.999
FREQ_CLIP_FACTOR = 3.0


def set_serif_font(size=None):
    """Use a serif font (Times New Roman) for all figures."""
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    if size is not None:
        mpl.rcParams["font.size"] = size


def parse_timestamps(series: pd.Series) -> pd.Series:
    """Parse timestamps, retrying day-first if the normal parse mostly fails."""
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().mean() > 0.5:  # mostly failed, retry day-first
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return parsed


def frequency_columns(df: pd.DataFrame) -> list:
    """Return the frequency columns (numeric MHz headers, 600-4000), sorted."""
    cols = []
    for col in df.columns:
        try:
            val = float(col)
        except (TypeError, ValueError):
            continue
        if 600.0 <= val <= 4000.0:
            cols.append(col)
    return sorted(cols, key=lambda c: float(c))


def rss(df: pd.DataFrame, cols: list) -> pd.Series:
    """Combine E-field columns by root-sum-square (V/m); a 0 result becomes NaN."""
    val = (df[cols] ** 2).sum(axis=1, min_count=1).pow(0.5)
    return val.where(val != 0, np.nan)


def select_datasets(labels):
    """Pick the datasets matching `labels` (None = all), keeping order."""
    if not labels:
        return list(DATASETS)
    wanted = set(labels)
    selected = [ds for ds in DATASETS if ds["label"] in wanted]
    missing = wanted - {ds["label"] for ds in DATASETS}
    if missing:
        known = ", ".join(ds["label"] for ds in DATASETS)
        raise SystemExit(f"Unknown dataset label(s): {', '.join(sorted(missing))}\n"
                         f"Known labels: {known}")
    return selected


# ======================================================================
# Command: normalize
# ======================================================================
#
# Rewrite the UH/ED processed CSVs in place into the standard schema:
#   - strip a BOM and rename the timestamp column to 'Timestamp'
#   - drop empty / unnamed columns
#   - snap off-canonical 2022 frequency labels to the standard band centres
#   - add the EMF_2G/3G/4G/5G, EMF_Iliad/TIM/VF/W3 and EMF_Total columns
# Safe to re-run: it recomputes from the frequency columns each time.

# Files rewritten in place (the 2023 reference file is left alone).
NORM_FILES = ["UH2022.csv", "ED2022.csv", "UH2024.csv", "ED2024.csv"]

# Off-canonical 2022 labels -> standard band centres (MHz).
NORM_SNAP = {
    2137.0: 2137.5, 2162.0: 2162.5, 2647.0: 2647.5, 2662.0: 2662.5,
    3547.0: 3547.5, 3447.0: 3449.0, 3468.0: 3469.0,
}

# Technology by frequency range (MHz), covering all 2022/2023/2024 channels.
NORM_TECH_RANGES = {
    "2G": [(930.0, 965.0)],
    "3G": [(2110.0, 2175.0)],
    "4G": [(760.0, 825.0), (1825.0, 1880.0), (2575.0, 2690.0)],
    "5G": [(3400.0, 3800.0)],
}

# Column output order for normalize.
NORM_TECH_ORDER = ["2G", "3G", "4G", "5G"]
NORM_OPERATOR_ORDER = ["Iliad", "TIM", "VF", "W3"]


def norm_freq_value(col):
    """Snapped MHz value of a frequency column, or None if it is not a frequency."""
    try:
        val = float(str(col).strip())
    except (TypeError, ValueError):
        return None
    if not (600.0 <= val <= 4000.0):
        return None
    return NORM_SNAP.get(val, val)


def norm_rss(df: pd.DataFrame, cols: list) -> pd.Series:
    """Combine E-field columns by root-sum-square (V/m); a 0 result becomes NaN."""
    val = (df[cols] ** 2).sum(axis=1).pow(0.5)
    return val.where(val != 0, np.nan)


def norm_tech_of(mhz: float):
    for tech, ranges in NORM_TECH_RANGES.items():
        if any(lo <= mhz <= hi for lo, hi in ranges):
            return tech
    return None


def normalize_file(path: Path):
    # utf-8-sig drops a leading BOM if present.
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    ts_col = next((c for c in TIMESTAMP_CANDIDATES if c in df.columns), None)
    if ts_col is None:
        raise ValueError(f"{path.name}: no timestamp column "
                         f"(looked for {TIMESTAMP_CANDIDATES})")
    ts = parse_timestamps(df[ts_col])

    # Build a clean frame keyed by timestamp.
    out = pd.DataFrame({"Timestamp": ts})
    n_snapped = 0
    for col in df.columns:
        if col == ts_col:
            continue
        mhz = norm_freq_value(col)
        if mhz is None:
            continue  # skip EMF_* and empty columns
        if mhz != float(str(col).strip()):
            n_snapped += 1
        label = str(mhz)
        series = pd.to_numeric(df[col], errors="coerce")
        if label in out.columns:
            # two columns snapped to the same centre: combine them
            out[label] = np.sqrt(out[label].fillna(0.0) ** 2
                                 + series.fillna(0.0) ** 2)
        else:
            out[label] = series

    out = out.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    freq_labels = sorted((c for c in out.columns if c != "Timestamp"),
                         key=lambda c: float(c))
    freq_mhz = {c: float(c) for c in freq_labels}

    # technology aggregations (by frequency range)
    for tech in NORM_TECH_ORDER:
        cols = [c for c in freq_labels if norm_tech_of(freq_mhz[c]) == tech]
        if cols:
            out[f"EMF_{tech}"] = norm_rss(out, cols)

    # operator aggregations (exact channels)
    for op in NORM_OPERATOR_ORDER:
        wanted = set(OPERATOR_BANDS[op])
        cols = [c for c in freq_labels if freq_mhz[c] in wanted]
        if cols:
            out[f"EMF_{op}"] = norm_rss(out, cols)

    # total across all frequencies
    out["EMF_Total"] = norm_rss(out, freq_labels)

    agg_order = ([f"EMF_{t}" for t in NORM_TECH_ORDER]
                 + [f"EMF_{o}" for o in NORM_OPERATOR_ORDER] + ["EMF_Total"])
    agg_order = [c for c in agg_order if c in out.columns]
    out = out[["Timestamp"] + freq_labels + agg_order]

    out["Timestamp"] = out["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False)

    return out, freq_labels, agg_order, n_snapped, ts


def run_normalize(files=None):
    files = files or NORM_FILES
    print("Normalizing UH/ED datasets in place")
    for name in files:
        path = PROCESSED_DIR / name
        if not path.exists():
            print(f"  missing: {path}")
            continue
        out, freq_labels, agg_order, n_snapped, ts = normalize_file(path)
        valid = ts.dropna()
        span = (f"{valid.min():%Y-%m-%d} to {valid.max():%Y-%m-%d}"
                if not valid.empty else "no valid timestamps")
        print(f"  {name}: {len(out):,} rows | {len(freq_labels)} freqs "
              f"({n_snapped} snapped) | {span}")
        print(f"      added: {', '.join(agg_order)}")
    print("Done.")


# ======================================================================
# Command: emf-levels
# ======================================================================
#
# EMF level (V/m) time-trend plots. For each dataset it plots, over the full
# period and a ~2-week zoom:
#   1. operators (W3, Iliad, TIM, VF)
#   2. technologies (2G, 3G, 4G, 5G)
#   3. every individual frequency, coloured by frequency
# It writes one combined figure, one operators+technologies figure, and six
# single-panel figures into result/EMF levels/<folder>/.

# Length of the auto zoom window, in days.
ZOOM_DAYS = 14

# Drop points outside +/- N std before plotting. Off by default.
REMOVE_OUTLIERS_STD = None

# All-frequencies panels: "colorbar" (colour by frequency) or "legend".
FREQ_LEGEND_MODE = "colorbar"

# Colormap for the all-frequencies panels.
FREQ_CMAP = "turbo"

# Line width / opacity per panel type.
STYLE = {
    "full": dict(lw=0.4, alpha=0.9),   # full period (dense)
    "zoom": dict(lw=0.9, alpha=0.9),   # zoom (~2 weeks)
}
FREQ_STYLE = {
    "full": dict(lw=0.35, alpha=0.85),
    "zoom": dict(lw=0.7, alpha=0.85),
}

# Fixed colour per operator and technology.
SERIES_COLORS = {
    "W3": "#1f77b4", "Iliad": "#ff7f0e", "TIM": "#2ca02c", "VF": "#d62728",
    "2G": "#1f77b4", "3G": "#ff7f0e", "4G": "#2ca02c", "5G": "#d62728",
}

# Dashed lines marking the zoom window on full-period panels.
ZOOM_MARKER_COLOR = "#1f3b5c"

# How tall the dashed markers are (fraction of the axes height).
ZOOM_MARKER_YMAX = 0.75

# Set per dataset in plot_dataset(); read when drawing zoom markers.
DATASET_LABEL = DATASETS[0]["label"]
FULL_PERIOD = DATASETS[0]["full_period"]
ZOOM_WINDOW = DATASETS[0]["zoom_window"]


def load_dataset(csv_path: Path):
    """Load a processed CSV, indexed by timestamp.

    Returns (df, freq_cols, tech_cols, op_cols). EMF_* columns are reused if
    present, otherwise built from the frequency columns.
    """
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    if TIMESTAMP_COL not in df.columns:
        alt = next((c for c in TIMESTAMP_CANDIDATES if c in df.columns), None)
        if alt is None:
            raise ValueError(f"No timestamp column found in {csv_path.name}")
        df = df.rename(columns={alt: TIMESTAMP_COL})

    df[TIMESTAMP_COL] = parse_timestamps(df[TIMESTAMP_COL])
    df = df.dropna(subset=[TIMESTAMP_COL]).set_index(TIMESTAMP_COL).sort_index()

    # frequency columns to numeric (V/m)
    freq_cols = frequency_columns(df)
    for c in freq_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # technologies: reuse EMF_<tech> if present, else compute
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
            df[col] = rss(df, present)
        tech_cols.append(col)

    # operators: reuse EMF_<op> if present, else compute
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
            df[col] = rss(df, present)
        op_cols.append(col)

    print(f"  {len(df):,} rows | {len(freq_cols)} freqs | "
          f"{len(tech_cols)} tech | {len(op_cols)} operator columns")
    return df, freq_cols, tech_cols, op_cols


def slice_period(df: pd.DataFrame, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    return df[(df.index >= start) & (df.index <= end)]


def auto_full_period(df: pd.DataFrame) -> dict:
    """Full period from the data's first to last timestamp."""
    return {"start": df.index.min().to_pydatetime(),
            "end":   df.index.max().to_pydatetime()}


def auto_zoom_window(df: pd.DataFrame, days: int = None) -> dict:
    """A `days`-long zoom centred on the middle of the data.

    If the middle falls in a gap, snap to the densest nearby window so the
    panel is not empty.
    """
    days = ZOOM_DAYS if days is None else days
    idx = df.index
    mid = pd.Timestamp(idx.min() + (idx.max() - idx.min()) / 2).normalize()
    span = pd.Timedelta(days=days)
    mid_start = mid - pd.Timedelta(days=days // 2)

    daily = pd.Series(1, index=idx).resample("1D").sum()
    roll = daily.rolling(f"{days}D").sum()
    best = roll.max() if roll.notna().any() else 0
    centered = ((idx >= mid_start) & (idx < mid_start + span)).sum()

    if best and centered < 0.6 * best:
        # middle is in a gap: snap to the densest nearby window
        dense_ends = roll[roll >= 0.95 * best].index
        end_day = min(dense_ends, key=lambda d: abs(d - (mid_start + span)))
        start = pd.Timestamp(end_day).normalize() - pd.Timedelta(days=days - 1)
    else:
        start = mid_start

    return {"start": start.to_pydatetime(),
            "end":   (start + span).to_pydatetime()}


def drop_outliers(df: pd.DataFrame, cols: list, n_std) -> pd.DataFrame:
    if n_std is None:
        return df
    mask = pd.Series(True, index=df.index)
    for col in cols:
        if col in df:
            m, s = df[col].mean(), df[col].std()
            mask &= df[col].between(m - n_std * s, m + n_std * s)
    return df[mask]


def clip_extreme_freq_spikes(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Drop per-frequency spikes using one global threshold. Returns a copy."""
    if FREQ_CLIP_QUANTILE is None or not cols:
        return df
    values = df[cols].to_numpy(dtype=float)
    if not np.isfinite(values).any():
        return df
    threshold = np.nanquantile(values, FREQ_CLIP_QUANTILE) * FREQ_CLIP_FACTOR
    out = df.copy()
    out[cols] = out[cols].where(out[cols] <= threshold)
    return out


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
    """Plot a few named series (operators or technologies)."""
    dfc = drop_outliers(df, cols, REMOVE_OUTLIERS_STD)
    style = STYLE[span]
    for name in order:
        col = f"EMF_{name}"
        if col not in dfc.columns:
            continue
        ax.plot(dfc.index, dfc[col], color=SERIES_COLORS.get(name),
                label=name, **style)

    if draw_zoom_markers:
        for x in (ZOOM_WINDOW["start"], ZOOM_WINDOW["end"]):
            ax.axvline(x, color=ZOOM_MARKER_COLOR, ls="--", lw=2.5, alpha=0.9,
                       ymax=ZOOM_MARKER_YMAX)

    ax.set_xlabel("Timestamp", fontsize=16, fontweight="bold")
    ax.set_ylabel("EMF (V/m)", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0),
              ncol=len(order), frameon=False, fontsize=14,
              prop={"weight": "bold"})
    _format_time_axis(ax, span)


def plot_frequencies(ax, df, freq_cols, span, fig=None):
    """Plot every frequency as a line, coloured by frequency."""
    dfc = drop_outliers(df, freq_cols, REMOVE_OUTLIERS_STD)
    dfc = clip_extreme_freq_spikes(dfc, freq_cols)
    style = FREQ_STYLE[span]
    freqs = [float(c) for c in freq_cols]
    fmin, fmax = min(freqs), max(freqs)
    norm = Normalize(vmin=fmin, vmax=fmax)
    cmap = mpl.colormaps[FREQ_CMAP]

    for c in freq_cols:
        ax.plot(dfc.index, dfc[c], color=cmap(norm(float(c))), **style)

    if span == "full":
        for x in (ZOOM_WINDOW["start"], ZOOM_WINDOW["end"]):
            ax.axvline(x, color=ZOOM_MARKER_COLOR, ls="--", lw=2.5, alpha=0.9,
                       ymax=ZOOM_MARKER_YMAX)

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


# The six panels: (breakdown, span).
PANELS = [
    ("operator", "full"), ("operator", "zoom"),
    ("technology", "full"), ("technology", "zoom"),
    ("frequency", "full"), ("frequency", "zoom"),
]

# Panels for the operators + technologies figure (no frequencies).
OPERATOR_TECH_PANELS = [
    ("operator", "full"), ("operator", "zoom"),
    ("technology", "full"), ("technology", "zoom"),
]

# Output file name for each single panel.
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
    """One stacked figure with a row per panel."""
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
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {png.name}")


def build_separate_figures(df_full, df_zoom, freq_cols, tech_cols, op_cols, out_dir):
    """One figure per panel."""
    print("Building separate per-panel figures...")
    for breakdown, span in PANELS:
        data = _panel_data(df_full, df_zoom, span)
        fig, ax = plt.subplots(figsize=(16, 6))
        _draw_panel(ax, breakdown, span, data, freq_cols, tech_cols, op_cols, fig=fig)
        fig.tight_layout()

        stem = PANEL_FILENAMES[(breakdown, span)]
        fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {stem}.png")


def plot_dataset(ds: dict):
    """Make all plots for one dataset."""
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

    out_dir = RESULT_DIR / "EMF levels" / ds["folder"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  output: {out_dir}")

    build_stacked_figure(df_full, df_zoom, freq_cols, tech_cols, op_cols, out_dir,
                         PANELS, "Combined (operators, technologies, frequencies)")
    build_stacked_figure(df_full, df_zoom, freq_cols, tech_cols, op_cols, out_dir,
                         OPERATOR_TECH_PANELS, "Combined (operators, technologies)")
    build_separate_figures(df_full, df_zoom, freq_cols, tech_cols, op_cols, out_dir)


def run_emf_levels(datasets=None):
    set_serif_font(size=16)
    datasets = datasets if datasets is not None else DATASETS
    print("EMF level time-trend plots")
    for ds in datasets:
        plot_dataset(ds)
    print("\nDone.")


# ======================================================================
# Command: heatmaps
# ======================================================================
#
# Correlation heatmaps per dataset:
#   1. frequency bands
#   2. technologies (2G/3G/4G/5G)
#   3. operators (Iliad/TIM/VF/W3)
# Writes a combined trio plus the three single heatmaps into
# result/Heatmaps/<folder>/, and an all-datasets grid when 2+ datasets are run.


def clip_frequency_artifacts(df):
    """Drop per-frequency spikes (per-column threshold).

    Returns (df, n_clipped, affected_cols).
    """
    freq_cols = frequency_columns(df)
    if FREQ_CLIP_QUANTILE is None or not freq_cols:
        return df, 0, []
    thresh = df[freq_cols].quantile(FREQ_CLIP_QUANTILE) * FREQ_CLIP_FACTOR
    mask = df[freq_cols] > thresh
    counts = mask.sum()
    affected = [str(c) for c in counts.index[counts > 0]]
    df = df.copy()
    df[freq_cols] = df[freq_cols].mask(mask)
    return df, int(counts.sum()), affected


def create_single_heatmap(title, data, output_folder, filename, trim_prefix=None):
    """Save one correlation heatmap."""
    if data.empty or len(data.columns) < 2:
        print(f"   Skipping {title}: insufficient data ({len(data.columns)} columns)")
        return

    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='viridis', vmin=0, vmax=1)

    # tick labels (trim the EMF_ prefix if asked)
    xticklabels = list(corr.columns)
    yticklabels = list(corr.index)
    if trim_prefix:
        xticklabels = [str(c).replace(trim_prefix, "") for c in xticklabels]
        yticklabels = [str(c).replace(trim_prefix, "") for c in yticklabels]
        tick_fontsize = 15
        label_fontsize = 14
    else:
        # font size by how many labels
        if len(xticklabels) > 20:
            tick_fontsize = 15
            label_fontsize = 15
        elif len(xticklabels) > 15:
            tick_fontsize = 15
            label_fontsize = 15
        else:
            tick_fontsize = 15
            label_fontsize = 15

    ax.set_xticks(np.arange(corr.shape[1]))
    ax.set_yticks(np.arange(corr.shape[0]))
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=label_fontsize)
    ax.set_yticklabels(yticklabels, fontsize=label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # write the value in each cell (white on viridis)
    if len(corr.columns) <= 10:
        fontsize = 9 if trim_prefix else 6
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                ax.text(j, i, f"{corr.iloc[i,j]:.2f}",
                        ha="center", va="center", fontsize=fontsize,
                        color='white', fontweight='bold')

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('Correlation', fontsize=13, fontweight='bold')
    plt.tight_layout()

    output_path_png = output_folder / f"{filename}.png"
    fig.savefig(output_path_png, dpi=300, bbox_inches="tight")
    print(f"   Saved: {output_path_png.name}")
    plt.close(fig)


def generate_dataset_heatmaps(df, dataset_name, dataset_folder):
    """Make the four heatmaps for one dataset."""
    dataset_folder.mkdir(parents=True, exist_ok=True)

    print(f"   Generating heatmaps for {dataset_name}...")

    # combined trio
    create_dataset_trio_heatmap(df, dataset_name, dataset_folder)

    # frequencies
    freq_cols = frequency_columns(df)
    if len(freq_cols) >= 2:
        freq_data = df[freq_cols].ffill().bfill()
        if not freq_data.empty and freq_data.notna().sum().sum() > 0:
            create_single_heatmap(f"{dataset_name} - Frequency Bands", freq_data,
                                  dataset_folder, "Frequency bands")
        else:
            print(f"   No valid frequency data for {dataset_name}")
    else:
        print(f"   Insufficient frequency columns for {dataset_name} ({len(freq_cols)} found)")

    # technologies
    tech_cols = [col for col in df.columns if isinstance(col, str) and col.startswith('EMF_')
                 and any(tech in col for tech in ['2G', '3G', '4G', '5G'])]
    if len(tech_cols) >= 2:
        tech_data = df[tech_cols].ffill().bfill()
        if not tech_data.empty and tech_data.notna().sum().sum() > 0:
            create_single_heatmap(f"{dataset_name} - Technology Bands", tech_data,
                                  dataset_folder, "Technology bands", "EMF_")
        else:
            print(f"   No valid technology data for {dataset_name}")
    else:
        print(f"   Insufficient technology columns for {dataset_name} ({len(tech_cols)} found)")

    # operators
    operator_cols = [col for col in df.columns if isinstance(col, str) and col.startswith('EMF_')
                     and any(op in col for op in ['Iliad', 'TIM', 'VF', 'W3'])]
    if len(operator_cols) >= 2:
        operator_data = df[operator_cols].ffill().bfill()
        if not operator_data.empty and operator_data.notna().sum().sum() > 0:
            create_single_heatmap(f"{dataset_name} - Operators", operator_data,
                                  dataset_folder, "Operators", "EMF_")
        else:
            print(f"   No valid operator data for {dataset_name}")
    else:
        print(f"   Insufficient operator columns for {dataset_name} ({len(operator_cols)} found)")


def create_dataset_trio_heatmap(df, dataset_name, dataset_folder):
    """One figure with frequency, technology and operator heatmaps side by side."""
    dataset_folder.mkdir(parents=True, exist_ok=True)

    freq_cols = frequency_columns(df)
    tech_cols = [col for col in df.columns if isinstance(col, str) and col.startswith('EMF_')
                 and any(tech in col for tech in ['2G', '3G', '4G', '5G'])]
    operator_cols = [col for col in df.columns if isinstance(col, str) and col.startswith('EMF_')
                     and any(op in col for op in ['Iliad', 'TIM', 'VF', 'W3'])]

    if len(freq_cols) < 2 or len(tech_cols) < 2 or len(operator_cols) < 2:
        print(f"   Skipping trio heatmap for {dataset_name}: insufficient data")
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    plot_data = [
        (freq_cols, "(a) Frequency Bands", None, axes[0], False),
        (tech_cols, "(b) Technology Bands", "EMF_", axes[1], True),
        (operator_cols, "(c) Operator Bands", "EMF_", axes[2], True)
    ]

    correlation_matrices = []

    for cols_to_use, subplot_title, trim_prefix, ax, is_trimmed in plot_data:
        subset = df[cols_to_use].ffill().bfill()
        if not subset.empty and subset.notna().sum().sum() > 0:
            corr = subset.corr()
            correlation_matrices.append(corr)

            labels = list(corr.columns)
            if trim_prefix:
                labels = [str(c).replace(trim_prefix, "") for c in labels]

            im = ax.imshow(corr, cmap='viridis', vmin=0, vmax=1)

            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))

            # font size by how many labels
            if len(labels) > 15:
                label_fontsize = 11
                tick_fontsize = 10
                text_fontsize = 8
            elif len(labels) > 8:
                label_fontsize = 12
                tick_fontsize = 11
                text_fontsize = 9
            else:
                label_fontsize = 13
                tick_fontsize = 12
                text_fontsize = 12

            ax.set_xticklabels(labels, rotation=90, fontsize=label_fontsize)
            ax.set_yticklabels(labels, fontsize=label_fontsize)
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

            # write the value in each cell
            if len(labels) <= 10:
                text_size = 12 if is_trimmed else 6
                for i in range(corr.shape[0]):
                    for j in range(corr.shape[1]):
                        ax.text(j, i, f"{corr.iloc[i,j]:.2f}",
                               ha="center", va="center", fontsize=text_size,
                               color='black', fontweight='bold')

            ax.set_xlabel(subplot_title, fontsize=13, fontweight='bold', labelpad=10)
        else:
            ax.text(0.5, 0.5, "No data\navailable", ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            correlation_matrices.append(None)
            ax.set_xlabel(subplot_title, fontsize=13, fontweight='bold', labelpad=10)

    # shared colorbar
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Correlation', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    plt.subplots_adjust(left=0.05, right=0.90, top=0.95, bottom=0.15, wspace=0.2)

    name = "Combined (frequency, technology, operators)"
    fig.savefig(dataset_folder / f"{name}.png", dpi=300, bbox_inches="tight")
    print(f"   Saved: {name}.png")
    plt.close(fig)


def create_all_datasets_comparison_heatmap(datasets_info, output_root):
    """One grid comparing all datasets."""
    output_root.mkdir(parents=True, exist_ok=True)

    if len(datasets_info) < 2:
        print("   Need at least 2 datasets for comparison heatmap")
        return

    n_datasets = len(datasets_info)
    fig, axes = plt.subplots(n_datasets, 3, figsize=(18, 6*n_datasets))
    if n_datasets == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('EMF Data Correlation Analysis: All Datasets Comparison',
                 fontsize=20, fontweight='bold', y=0.98)

    for row, (df, dataset_name, _folder) in enumerate(datasets_info):
        freq_cols = frequency_columns(df)
        tech_cols = [col for col in df.columns if isinstance(col, str) and col.startswith('EMF_')
                     and any(tech in col for tech in ['2G', '3G', '4G', '5G'])]
        operator_cols = [col for col in df.columns if isinstance(col, str) and col.startswith('EMF_')
                         and any(op in col for op in ['Iliad', 'TIM', 'VF', 'W3'])]

        plot_data = [
            (freq_cols, "Frequency Bands", None),
            (tech_cols, "Technology Bands", "EMF_"),
            (operator_cols, "Operators", "EMF_")
        ]

        for col, (cols_to_use, comp_name, trim_prefix) in enumerate(plot_data):
            ax = axes[row, col]

            if len(cols_to_use) >= 2:
                subset = df[cols_to_use].ffill().bfill()
                if not subset.empty and subset.notna().sum().sum() > 0:
                    corr = subset.corr()

                    im = ax.imshow(corr, cmap='viridis', vmin=0, vmax=1)

                    labels = list(corr.columns)
                    if trim_prefix:
                        labels = [str(c).replace(trim_prefix, "") for c in labels]

                    ax.set_xticks(np.arange(len(labels)))
                    ax.set_yticks(np.arange(len(labels)))

                    # font size
                    if len(labels) > 15:
                        label_fontsize = 15
                        tick_fontsize = 15
                        text_fontsize = 15
                    elif len(labels) > 8:
                        label_fontsize = 15
                        tick_fontsize = 15
                        text_fontsize = 15
                    else:
                        label_fontsize = 15
                        tick_fontsize = 15
                        text_fontsize = 15

                    ax.set_xticklabels(labels, rotation=90, fontsize=label_fontsize)
                    ax.set_yticklabels(labels, fontsize=label_fontsize)
                    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=tick_fontsize)

                    if len(labels) <= 10:
                        for i in range(corr.shape[0]):
                            for j in range(corr.shape[1]):
                                ax.text(j, i, f"{corr.iloc[i,j]:.2f}",
                                       ha="center", va="center", fontsize=text_fontsize,
                                       color='white', fontweight='bold')

                    ax.set_title(f"{dataset_name}\n{comp_name}", fontsize=13, fontweight='bold', pad=10)
                else:
                    ax.text(0.5, 0.5, "No data\navailable", ha='center', va='center',
                           transform=ax.transAxes, fontsize=10)
                    ax.set_title(f"{dataset_name}\n{comp_name}", fontsize=10, fontweight='bold')
                    ax.set_xticks([])
                    ax.set_yticks([])
            else:
                ax.text(0.5, 0.5, f"Insufficient\ndata\n({len(cols_to_use)} cols)",
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f"{dataset_name}\n{comp_name}", fontsize=10, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])

    plt.tight_layout()

    name = "All datasets comparison"
    fig.savefig(output_root / f"{name}.png", dpi=300, bbox_inches="tight")
    print(f"   Saved: {name}.png")
    plt.close(fig)


def generate_all_heatmaps(datasets_info, output_root):
    """Make every heatmap for the given datasets.

    datasets_info: list of (df, name, folder). output_root: the Heatmaps folder.
    """
    print("Generating correlation heatmaps...")
    for df, dataset_name, folder in datasets_info:
        generate_dataset_heatmaps(df, dataset_name, output_root / folder)
    create_all_datasets_comparison_heatmap(datasets_info, output_root)
    print(f"All heatmaps saved in: {output_root}")


def run_heatmaps(datasets=None):
    set_serif_font()
    datasets = datasets if datasets is not None else DATASETS
    print("Heatmap generation")

    datasets_info = []
    for ds in datasets:
        csv_path, label, folder = ds["csv"], ds["label"], ds["folder"]
        if not csv_path.exists():
            print(f"Dataset not found: {csv_path.resolve()}")
            continue
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"Loaded {label}: {len(df):,} rows x {len(df.columns)} cols")
        df, n_clipped, clipped_cols = clip_frequency_artifacts(df)
        if n_clipped:
            print(f"  clipped {n_clipped} per-frequency spike(s) "
                  f"(> {FREQ_CLIP_FACTOR}x the {FREQ_CLIP_QUANTILE:g} quantile) "
                  f"in MHz: {', '.join(clipped_cols)}")
        datasets_info.append((df, label, folder))

    if not datasets_info:
        print("No datasets loaded; nothing to do.")
        return

    out_root = RESULT_DIR / "Heatmaps"
    out_root.mkdir(parents=True, exist_ok=True)

    generate_all_heatmaps(datasets_info, out_root)
    print("\nDone. Files saved in:", out_root.resolve())


# ======================================================================
# Command: all
# ======================================================================


def run_all(datasets=None):
    print("Step 1 of 2: EMF levels")
    run_emf_levels(datasets)

    print("\nStep 2 of 2: Heatmaps")
    run_heatmaps(datasets)

    print("\nAll done.")


# ======================================================================
# Command: window-analysis
# ======================================================================
#
# Read a raw fileCP .txt (dBm), keep the longest continuous period, convert dBm
# to E-field (V/m) with antenna-factor calibration, add the EMF_* aggregations,
# and plot the 1/3/7/14-day windows around the middle of that period. Needs the
# .txt and the antenna-factor .xlsx (pass with --input / --antenna-factors).

# Default input file and its continuous period.
WIN_TARGET_DATASET = "fileCP_2024_UNIVERSITY_HOSPITAL.txt"
WIN_CONTINUOUS_PERIOD = {
    "start": dt.datetime(2023, 7, 26, 16, 49, 2),
    "end": dt.datetime(2024, 3, 11, 5, 8, 28),
    "duration": "228 days, 12 hours"
}

# Middle of the period.
WIN_CENTER = dt.datetime(2023, 11, 17, 12, 0, 0)

# Window sizes (days).
WIN_SIZES = {"1day": 1, "3days": 3, "7days": 7, "14days": 14}

# Default antenna-factor file.
WIN_AF_FILE = "9018-18835.xlsx"

# Tag used in output file names.
WIN_LABEL = "2024_UNIVERSITY_HOSPITAL"


def win_parse_raw(path: Path) -> pd.DataFrame:
    """Parse the raw .txt into a frame keyed by timestamp; drop all-zero rows."""
    print(f"Parsing: {path.name}")

    txt = path.read_text(encoding="utf-8", errors="ignore")
    rows = []
    all_zero_count = 0

    for block in txt.split("Inizio;")[1:]:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 4:
            continue

        try:
            dt.datetime.strptime(lines[0], "%d/%m/%Y %H:%M:%S")
        except ValueError:
            continue

        freqs = [
            float(m.group())
            for tok in lines[1].split(";")
            for m in [re.search(r"\d+(?:\.\d+)?", tok)]
            if m
        ]
        n = len(freqs)

        for line in lines[3:]:
            tok = [t for t in line.split(";") if t]
            if len(tok) < 2 * n:
                continue

            try:
                row_ts = dt.datetime.strptime(tok[0], "%d/%m/%Y %H:%M:%S")
            except ValueError:
                continue

            row = {"Timestamp": row_ts}
            valid_values = []
            all_zero = True

            for i, f in enumerate(freqs):
                try:
                    v = float(tok[2 * i + 1])
                    if v == 0:
                        v = math.nan
                    else:
                        valid_values.append(v)
                        all_zero = False
                except ValueError:
                    v = math.nan
                row[f] = v

            # keep rows with at least one non-zero value
            if not all_zero and valid_values:
                rows.append(row)
            else:
                all_zero_count += 1

    df = pd.DataFrame(rows).set_index("Timestamp").sort_index()

    print(f"   Parsed {len(df):,} rows with {len(df.columns)} frequencies")
    print(f"   Filtered out {all_zero_count:,} all-zero rows")
    return df


def win_filter_to_continuous_period(df: pd.DataFrame, period: dict) -> pd.DataFrame:
    """Keep only rows inside the continuous period."""
    start_time = period["start"]
    end_time = period["end"]

    print(f"   Filtering to continuous period: {start_time:%Y-%m-%d %H:%M:%S} "
          f"to {end_time:%Y-%m-%d %H:%M:%S}")
    print(f"   Period duration: {period['duration']}")

    filtered_df = df[(df.index >= start_time) & (df.index <= end_time)].copy()

    original_rows = len(df)
    filtered_rows = len(filtered_df)
    pct = (filtered_rows / original_rows * 100) if original_rows else 0.0
    print(f"   Filtered: {original_rows:,} -> {filtered_rows:,} rows ({pct:.1f}% retained)")

    return filtered_df


def win_load_antenna_factors(af_path: Path):
    """Load the antenna-factor table, or (None, None) if unavailable."""
    if not af_path.exists():
        print(f"   Antenna factor file not found: {af_path}")
        return None, None
    try:
        af_df = pd.read_excel(af_path, usecols=["Freq (MHz)", "AF(dBm-1)"]).dropna()
        af_freq = af_df["Freq (MHz)"].to_numpy()
        af_val = af_df["AF(dBm-1)"].to_numpy()
        print(f"   Loaded antenna factors: {len(af_freq)} frequency points")
        return af_freq, af_val
    except Exception as e:
        print(f"   Error loading antenna factors: {str(e)}")
        return None, None


def win_get_antenna_factor(freq_mhz: float, af_freq, af_val) -> float:
    """Nearest antenna factor (dB) for a frequency; 20.0 if no table."""
    if af_freq is None or af_val is None:
        return 20.0  # fallback
    idx = np.abs(af_freq - freq_mhz).argmin()
    return float(af_val[idx])


def win_dBm_to_E_field(dBm: float, freq_mhz: float, af_freq, af_val) -> float:
    """Convert dBm to E-field (V/m)."""
    if math.isnan(dBm):
        return math.nan

    C = 299_792_458.0  # speed of light
    Z0 = 120 * math.pi  # free-space impedance

    P_r = 10 ** ((dBm - 30) / 10)  # received power, W

    lam = C / (freq_mhz * 1e6)  # wavelength, m
    alpha_db = win_get_antenna_factor(freq_mhz, af_freq, af_val)  # dB m^-1
    G_f = (9.73 / (lam * 10 ** (alpha_db / 20))) ** 2  # linear antenna gain

    A_e = G_f * lam ** 2 / (4 * math.pi)  # effective aperture, m^2

    return math.sqrt(P_r * Z0 / A_e)  # E-field, V/m


def win_convert_to_emf(rss_df: pd.DataFrame, af_freq, af_val) -> pd.DataFrame:
    """Convert every frequency column from dBm to E-field (V/m)."""
    print("   Converting dBm to E-field...")

    emf_df = rss_df.copy()
    freq_cols = [col for col in rss_df.columns if isinstance(col, (int, float))]

    for freq in freq_cols:
        emf_df[freq] = rss_df[freq].apply(
            lambda dBm: win_dBm_to_E_field(dBm, freq, af_freq, af_val)
        )

    print(f"   EMF conversion complete for {len(freq_cols)} frequencies")
    return emf_df


def win_add_aggregated_columns(emf_df: pd.DataFrame) -> pd.DataFrame:
    """Add the EMF_<tech>, EMF_<operator> and EMF_Total columns."""
    print("   Adding aggregated EMF columns...")

    for tech, freqs in TECH_BANDS.items():
        present_freqs = [f for f in freqs if f in emf_df.columns]
        if present_freqs:
            emf_val = (emf_df[present_freqs] ** 2).sum(axis=1).pow(0.5)
            emf_val = emf_val.where(emf_val != 0, np.nan)
            emf_df[f"EMF_{tech}"] = emf_val
        else:
            print(f"   No frequencies found for {tech}")

    for op, freqs in OPERATOR_BANDS.items():
        present_freqs = [f for f in freqs if f in emf_df.columns]
        if present_freqs:
            emf_val = (emf_df[present_freqs] ** 2).sum(axis=1).pow(0.5)
            emf_val = emf_val.where(emf_val != 0, np.nan)
            emf_df[f"EMF_{op}"] = emf_val
        else:
            print(f"   No frequencies found for {op}")

    freq_cols = [c for c in emf_df.columns if isinstance(c, (int, float))]
    if freq_cols:
        emf_val = (emf_df[freq_cols] ** 2).sum(axis=1).pow(0.5)
        emf_val = emf_val.where(emf_val != 0, np.nan)
        emf_df["EMF_Total"] = emf_val

    tech_cols = [col for col in emf_df.columns if isinstance(col, str) and col.startswith('EMF_')
                 and any(t in col for t in TECH_BANDS.keys())]
    op_cols = [col for col in emf_df.columns if isinstance(col, str) and col.startswith('EMF_')
               and any(o in col for o in OPERATOR_BANDS.keys())]
    total_cols = [col for col in emf_df.columns if col == 'EMF_Total']

    print(f"   Added {len(tech_cols)} technology + {len(op_cols)} operator "
          f"+ {len(total_cols)} total EMF columns")
    return emf_df


def win_get_window_timerange(window_name: str, center_date: dt.datetime):
    """Start and end of a window, snapped to whole days."""
    days = WIN_SIZES[window_name]
    half_window = dt.timedelta(days=days / 2)

    # align to midnight
    start_time = (center_date - half_window).replace(hour=0, minute=0, second=0, microsecond=0)
    end_time = (center_date + half_window).replace(hour=23, minute=59, second=59, microsecond=999999)

    return start_time, end_time


def win_filter_to_window(df: pd.DataFrame, window_name: str, center_date: dt.datetime) -> pd.DataFrame:
    """Keep only rows inside one window."""
    start_time, end_time = win_get_window_timerange(window_name, center_date)

    print(f"   Filtering to {window_name} window: {start_time:%Y-%m-%d %H:%M:%S} "
          f"to {end_time:%Y-%m-%d %H:%M:%S}")

    windowed_df = df[(df.index >= start_time) & (df.index <= end_time)].copy()

    print(f"   Window filtered: {len(df):,} -> {len(windowed_df):,} rows")
    return windowed_df


def win_remove_outliers_for_plotting(df, cols, n_std=3):
    """Drop rows where a column is outside n_std standard deviations."""
    mask = pd.Series([True] * len(df), index=df.index)
    for col in cols:
        if col in df:
            mean = df[col].mean()
            std = df[col].std()
            mask &= (df[col] >= mean - n_std * std) & (df[col] <= mean + n_std * std)
    return df[mask]


def win_plot_window_analysis(emf_df: pd.DataFrame, output_folder: Path, center_date: dt.datetime):
    """Plot technology and operator bands for each window."""
    print("Generating window analysis plots...")

    plots_folder = output_folder / "plots"
    plots_folder.mkdir(exist_ok=True)

    TECH_COLS = ["EMF_2G", "EMF_3G", "EMF_4G", "EMF_5G"]
    OP_COLS = ["EMF_W3", "EMF_Iliad", "EMF_TIM", "EMF_VF"]

    available_tech_cols = [col for col in TECH_COLS if col in emf_df.columns]
    available_op_cols = [col for col in OP_COLS if col in emf_df.columns]

    if not available_tech_cols and not available_op_cols:
        print("   No EMF aggregation columns found for plotting")
        return

    print(f"   Available technology columns: {available_tech_cols}")
    print(f"   Available operator columns: {available_op_cols}")

    for window_name in WIN_SIZES.keys():
        print(f"\nProcessing {window_name} window...")

        windowed_df = win_filter_to_window(emf_df, window_name, center_date)
        if windowed_df.empty:
            print(f"   No data in {window_name} window")
            continue

        df_plot = windowed_df.reset_index()

        # technology bands
        if available_tech_cols:
            plt.figure(figsize=(16, 8))
            df_clean = win_remove_outliers_for_plotting(df_plot, available_tech_cols)

            for col in available_tech_cols:
                legend_label = col.replace("EMF_", "")
                plt.plot(df_clean["Timestamp"], df_clean[col], lw=0.8, alpha=0.8, label=legend_label)

            plt.xlabel("Timestamp", fontsize=20, fontweight='bold')
            plt.ylabel("EMF (V/m)", fontsize=20, fontweight='bold')
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00),
                       ncol=len(available_tech_cols), frameon=False, fontsize=16,
                       title_fontsize=18, prop={'weight': 'bold'})
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            tech_png = plots_folder / f"emf_technology_{WIN_LABEL}_{window_name}_window.png"
            tech_pdf = plots_folder / f"emf_technology_{WIN_LABEL}_{window_name}_window.pdf"
            plt.savefig(tech_png, dpi=300, bbox_inches='tight')
            plt.savefig(tech_pdf, dpi=300, format='pdf', bbox_inches='tight')
            print(f"   Saved technology plot: {tech_png.name}")
            plt.close()

        # operator bands
        if available_op_cols:
            plt.figure(figsize=(16, 8))
            df_clean = win_remove_outliers_for_plotting(df_plot, available_op_cols)

            for col in available_op_cols:
                legend_label = col.replace("EMF_", "")
                plt.plot(df_clean["Timestamp"], df_clean[col], lw=0.8, alpha=0.8, label=legend_label)

            plt.xlabel("Timestamp", fontsize=20, fontweight='bold')
            plt.ylabel("EMF (V/m)", fontsize=20, fontweight='bold')
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00),
                       ncol=len(available_op_cols), frameon=False, fontsize=16,
                       title_fontsize=18, prop={'weight': 'bold'})
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            op_png = plots_folder / f"emf_operator_{WIN_LABEL}_{window_name}_window.png"
            op_pdf = plots_folder / f"emf_operator_{WIN_LABEL}_{window_name}_window.pdf"
            plt.savefig(op_png, dpi=300, bbox_inches='tight')
            plt.savefig(op_pdf, dpi=300, format='pdf', bbox_inches='tight')
            print(f"   Saved operator plot: {op_png.name}")
            plt.close()


def win_plot_all_windows_summary(emf_df: pd.DataFrame, output_folder: Path, center_date: dt.datetime):
    """One figure with all windows (technology and operator columns)."""
    print("Generating all windows summary plot...")

    plots_folder = output_folder / "plots"
    plots_folder.mkdir(exist_ok=True)

    TECH_COLS = ["EMF_2G", "EMF_3G", "EMF_4G", "EMF_5G"]
    OP_COLS = ["EMF_W3", "EMF_Iliad", "EMF_TIM", "EMF_VF"]

    available_tech_cols = [col for col in TECH_COLS if col in emf_df.columns]
    available_op_cols = [col for col in OP_COLS if col in emf_df.columns]

    if not available_tech_cols and not available_op_cols:
        print("   No EMF aggregation columns found for summary plotting")
        return

    fig, axes = plt.subplots(4, 2, figsize=(24, 20))

    fig.suptitle('EMF Window Analysis - 2024 UNIVERSITY HOSPITAL\n'
                 'Multiple Time Windows from Continuous Period',
                 fontsize=24, fontweight='bold', y=0.98)

    window_names = list(WIN_SIZES.keys())

    for row, window_name in enumerate(window_names):
        windowed_df = win_filter_to_window(emf_df, window_name, center_date)

        if windowed_df.empty:
            for col in [0, 1]:
                axes[row, col].text(0.5, 0.5, f"No data in {window_name} window",
                                  ha='center', va='center', transform=axes[row, col].transAxes, fontsize=14)
                axes[row, col].set_title(f"{window_name.upper()} Window", fontsize=16, fontweight='bold')
            continue

        df_plot = windowed_df.reset_index()
        start_time, end_time = win_get_window_timerange(window_name, center_date)

        # technology (left)
        ax_tech = axes[row, 0]
        if available_tech_cols:
            df_clean = win_remove_outliers_for_plotting(df_plot, available_tech_cols)

            for col in available_tech_cols:
                legend_label = col.replace("EMF_", "")
                ax_tech.plot(df_clean["Timestamp"], df_clean[col], lw=0.6, alpha=0.8, label=legend_label)

            ax_tech.set_xlabel("Timestamp", fontsize=14, fontweight='bold')
            ax_tech.set_ylabel("EMF (V/m)", fontsize=14, fontweight='bold')
            ax_tech.tick_params(axis='both', which='major', labelsize=12)
            ax_tech.legend(loc='upper right', fontsize=10, frameon=True, prop={'weight': 'bold'})
            ax_tech.set_title(f"Technology - {window_name.upper()} "
                              f"({start_time:%m-%d} to {end_time:%m-%d})", fontsize=14, fontweight='bold')
            ax_tech.grid(True, alpha=0.3)
        else:
            ax_tech.text(0.5, 0.5, "No Technology\nData Available",
                        ha='center', va='center', transform=ax_tech.transAxes, fontsize=12)
            ax_tech.set_title(f"Technology - {window_name.upper()}", fontsize=14, fontweight='bold')

        # operator (right)
        ax_op = axes[row, 1]
        if available_op_cols:
            df_clean = win_remove_outliers_for_plotting(df_plot, available_op_cols)

            for col in available_op_cols:
                legend_label = col.replace("EMF_", "")
                ax_op.plot(df_clean["Timestamp"], df_clean[col], lw=0.6, alpha=0.8, label=legend_label)

            ax_op.set_xlabel("Timestamp", fontsize=14, fontweight='bold')
            ax_op.set_ylabel("EMF (V/m)", fontsize=14, fontweight='bold')
            ax_op.tick_params(axis='both', which='major', labelsize=12)
            ax_op.legend(loc='upper right', fontsize=10, frameon=True, prop={'weight': 'bold'})
            ax_op.set_title(f"Operator - {window_name.upper()} "
                            f"({start_time:%m-%d} to {end_time:%m-%d})", fontsize=14, fontweight='bold')
            ax_op.grid(True, alpha=0.3)
        else:
            ax_op.text(0.5, 0.5, "No Operator\nData Available",
                      ha='center', va='center', transform=ax_op.transAxes, fontsize=12)
            ax_op.set_title(f"Operator - {window_name.upper()}", fontsize=14, fontweight='bold')

    plt.tight_layout()

    summary_png = plots_folder / f"emf_{WIN_LABEL}_all_windows_summary.png"
    summary_pdf = plots_folder / f"emf_{WIN_LABEL}_all_windows_summary.pdf"

    fig.savefig(summary_png, dpi=300, bbox_inches='tight')
    fig.savefig(summary_pdf, dpi=300, format='pdf', bbox_inches='tight')
    print(f"   Saved all windows summary plot: {summary_png.name}")
    print(f"   Saved all windows summary plot: {summary_pdf.name}")
    plt.close(fig)


def run_window_analysis(input_path=None, antenna_factors=None, center_date=None,
                        period=None, timestamp=None):
    """Run the window analysis on a raw .txt.

    input_path / antenna_factors default to the constants above; output goes to
    a timestamped folder under result/.
    """
    set_serif_font(size=20)

    file_path = Path(input_path) if input_path else Path(WIN_TARGET_DATASET)
    af_path = Path(antenna_factors) if antenna_factors else Path(WIN_AF_FILE)
    center_date = center_date or WIN_CENTER
    period = period or WIN_CONTINUOUS_PERIOD
    timestamp = timestamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("FileCP Window Analysis - 2024 UNIVERSITY HOSPITAL")
    print("=" * 70)
    print(f"Target dataset: {file_path}")
    print(f"Continuous period: {period['start']:%Y-%m-%d %H:%M:%S} "
          f"to {period['end']:%Y-%m-%d %H:%M:%S}")
    print(f"Window center: {center_date:%Y-%m-%d %H:%M:%S}")

    print("\nWindow Analysis:")
    for window_name, days in WIN_SIZES.items():
        start_time, end_time = win_get_window_timerange(window_name, center_date)
        print(f"   - {window_name}: {start_time:%Y-%m-%d} to {end_time:%Y-%m-%d} ({days} days)")

    # output under result/
    result_folder = RESULT_DIR
    result_folder.mkdir(parents=True, exist_ok=True)
    timestamped_folder = result_folder / f"window_analysis_{timestamp}"
    timestamped_folder.mkdir(exist_ok=True)
    print(f"\nOutput folder: {timestamped_folder}")

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    print("\n" + "=" * 70)

    try:
        df = win_parse_raw(file_path)
        if df.empty:
            print(f"No data extracted from: {file_path}")
            return

        df = win_filter_to_continuous_period(df, period)
        if df.empty:
            print("No data in continuous period")
            return

        print("\nLoading antenna factor calibration...")
        af_freq, af_val = win_load_antenna_factors(af_path)

        emf_df = win_convert_to_emf(df, af_freq, af_val)
        emf_df = win_add_aggregated_columns(emf_df)

        freq_cols = [col for col in emf_df.columns if isinstance(col, (int, float))]
        emf_agg_cols = [col for col in emf_df.columns if isinstance(col, str) and col.startswith('EMF_')]

        print(f"   Dataset ready: {len(freq_cols)} frequencies + {len(emf_agg_cols)} EMF aggregations")
        print(f"   Total data points in continuous period: {len(emf_df):,}")

        win_plot_window_analysis(emf_df, timestamped_folder, center_date)
        win_plot_all_windows_summary(emf_df, timestamped_folder, center_date)

        # summary report
        summary_file = timestamped_folder / "window_analysis_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("FileCP Window Analysis Summary - 2024 UNIVERSITY HOSPITAL\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {dt.datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"Output folder: {timestamped_folder}\n\n")

            f.write(f"TARGET DATASET: {file_path}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Continuous period: {period['start']:%Y-%m-%d %H:%M:%S} "
                    f"to {period['end']:%Y-%m-%d %H:%M:%S}\n")
            f.write(f"Duration: {period['duration']}\n")
            f.write(f"Window center: {center_date:%Y-%m-%d %H:%M:%S}\n")
            f.write(f"Total data points: {len(emf_df):,}\n\n")

            f.write("WINDOW ANALYSIS:\n")
            f.write("-" * 60 + "\n")
            for window_name, days in WIN_SIZES.items():
                start_time, end_time = win_get_window_timerange(window_name, center_date)
                windowed_df = win_filter_to_window(emf_df, window_name, center_date)
                f.write(f"{window_name}: {start_time:%Y-%m-%d} to {end_time:%Y-%m-%d} ({days} days)\n")
                f.write(f"  Data points: {len(windowed_df):,}\n")
                f.write(f"  Files generated:\n")
                f.write(f"    - emf_technology_{WIN_LABEL}_{window_name}_window.png/.pdf\n")
                f.write(f"    - emf_operator_{WIN_LABEL}_{window_name}_window.png/.pdf\n\n")

            f.write("SUMMARY VISUALIZATIONS:\n")
            f.write("-" * 60 + "\n")
            f.write(f"- emf_{WIN_LABEL}_all_windows_summary.png/.pdf (4 windows x 2 plot types)\n")

        print("\n" + "=" * 70)
        print("WINDOW ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"Successfully processed: {file_path}")
        print(f"Output folder: {timestamped_folder}")
        print(f"Plots folder: {timestamped_folder}/plots")
        print(f"Summary report: {summary_file.name}")

        print("\nGenerated window plots:")
        for window_name in WIN_SIZES.keys():
            print(f"   - {window_name} window:")
            print(f"     - emf_technology_{WIN_LABEL}_{window_name}_window.png/.pdf")
            print(f"     - emf_operator_{WIN_LABEL}_{window_name}_window.png/.pdf")

        print("\nSummary visualization:")
        print(f"   - emf_{WIN_LABEL}_all_windows_summary.png/.pdf (4 windows x 2 plot types)")

        print(f"\nWindow analysis complete. Files saved in: {timestamped_folder.absolute()}")

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()


# ======================================================================
# Command line interface
# ======================================================================


def _add_clip_options(p):
    p.add_argument("--freq-clip-quantile", type=float, default=None,
                   help=f"Quantile for per-frequency spike clipping "
                        f"(default {FREQ_CLIP_QUANTILE}).")
    p.add_argument("--freq-clip-factor", type=float, default=None,
                   help=f"Factor x quantile above which a sample is an artifact "
                        f"(default {FREQ_CLIP_FACTOR}).")
    p.add_argument("--no-freq-clip", action="store_true",
                   help="Disable per-frequency spike clipping entirely.")


def _apply_clip_options(args):
    global FREQ_CLIP_QUANTILE, FREQ_CLIP_FACTOR
    if getattr(args, "no_freq_clip", False):
        FREQ_CLIP_QUANTILE = None
    elif getattr(args, "freq_clip_quantile", None) is not None:
        FREQ_CLIP_QUANTILE = args.freq_clip_quantile
    if getattr(args, "freq_clip_factor", None) is not None:
        FREQ_CLIP_FACTOR = args.freq_clip_factor


def build_parser():
    labels = [ds["label"] for ds in DATASETS]

    parser = argparse.ArgumentParser(
        prog="emf_multivariate.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Unified entry point for the multivariate EMF analysis. "
                    "Pick a sub-command; use `<command> --help` for its options.",
        epilog="Examples:\n"
               "  python emf_multivariate.py all\n"
               "  python emf_multivariate.py emf-levels --datasets UH_2022 ED_2022\n"
               "  python emf_multivariate.py heatmaps --no-freq-clip\n"
               "  python emf_multivariate.py normalize\n",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    # all
    p_all = sub.add_parser("all", help="Run emf-levels then heatmaps.",
                           formatter_class=argparse.RawDescriptionHelpFormatter)
    p_all.add_argument("--datasets", nargs="+", metavar="LABEL", choices=labels,
                       help=f"Subset of datasets to process (default: all). "
                            f"Choices: {', '.join(labels)}.")
    _add_clip_options(p_all)
    p_all.add_argument("--zoom-days", type=int, default=None,
                       help=f"Length of the auto-derived zoom window in days "
                            f"(default {ZOOM_DAYS}).")
    p_all.add_argument("--remove-outliers-std", type=float, default=None,
                       help="Drop points outside +/- N std before plotting "
                            "(default: off).")
    p_all.add_argument("--freq-legend-mode", choices=["colorbar", "legend"], default=None,
                       help=f"How to render the all-frequencies panels "
                            f"(default {FREQ_LEGEND_MODE}).")
    p_all.add_argument("--freq-cmap", default=None,
                       help=f"Colormap for the all-frequencies panels "
                            f"(default {FREQ_CMAP}).")

    # emf-levels
    p_emf = sub.add_parser("emf-levels",
                           help="EMF level time-trend plots.",
                           formatter_class=argparse.RawDescriptionHelpFormatter)
    p_emf.add_argument("--datasets", nargs="+", metavar="LABEL", choices=labels,
                       help=f"Subset of datasets to process (default: all). "
                            f"Choices: {', '.join(labels)}.")
    p_emf.add_argument("--zoom-days", type=int, default=None,
                       help=f"Length of the auto-derived zoom window in days "
                            f"(default {ZOOM_DAYS}).")
    p_emf.add_argument("--remove-outliers-std", type=float, default=None,
                       help="Drop points outside +/- N std before plotting "
                            "(default: off).")
    p_emf.add_argument("--freq-legend-mode", choices=["colorbar", "legend"], default=None,
                       help=f"How to render the all-frequencies panels "
                            f"(default {FREQ_LEGEND_MODE}).")
    p_emf.add_argument("--freq-cmap", default=None,
                       help=f"Colormap for the all-frequencies panels "
                            f"(default {FREQ_CMAP}).")
    _add_clip_options(p_emf)

    # heatmaps
    p_hm = sub.add_parser("heatmaps", help="Correlation heatmaps per dataset.",
                          formatter_class=argparse.RawDescriptionHelpFormatter)
    p_hm.add_argument("--datasets", nargs="+", metavar="LABEL", choices=labels,
                      help=f"Subset of datasets to process (default: all). "
                           f"Choices: {', '.join(labels)}.")
    _add_clip_options(p_hm)

    # normalize
    p_norm = sub.add_parser("normalize",
                            help="Rebuild UH/ED processed CSVs in place.",
                            formatter_class=argparse.RawDescriptionHelpFormatter)
    p_norm.add_argument("--files", nargs="+", metavar="NAME", default=None,
                        help=f"CSV file names under data/multivariate/processed/ "
                             f"to normalize (default: {', '.join(NORM_FILES)}).")

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    global ZOOM_DAYS, REMOVE_OUTLIERS_STD, FREQ_LEGEND_MODE, FREQ_CMAP

    if args.command in ("emf-levels", "all"):
        if getattr(args, "zoom_days", None) is not None:
            ZOOM_DAYS = args.zoom_days
        if getattr(args, "remove_outliers_std", None) is not None:
            REMOVE_OUTLIERS_STD = args.remove_outliers_std
        if getattr(args, "freq_legend_mode", None) is not None:
            FREQ_LEGEND_MODE = args.freq_legend_mode
        if getattr(args, "freq_cmap", None) is not None:
            FREQ_CMAP = args.freq_cmap

    if args.command in ("emf-levels", "heatmaps", "all"):
        _apply_clip_options(args)
        datasets = select_datasets(getattr(args, "datasets", None))

    if args.command == "all":
        run_all(datasets)
    elif args.command == "emf-levels":
        run_emf_levels(datasets)
    elif args.command == "heatmaps":
        run_heatmaps(datasets)
    elif args.command == "normalize":
        run_normalize(args.files)


if __name__ == "__main__":
    main()
