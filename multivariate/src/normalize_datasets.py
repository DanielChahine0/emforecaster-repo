#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Normalize the UH/ED processed CSVs into the same schema as
EMF_2023_university_5_with_PTV.csv, so they work unchanged with
plot_emf_figure4.py and run_heatmaps.py.

For each file this:
  - strips a UTF-8 BOM and renames the timestamp column to 'Timestamp'
  - drops empty / unnamed trailing columns
  - snaps off-canonical 2022 frequency labels to the project's band centres
    (2137 -> 2137.5, 2162 -> 2162.5, 2647 -> 2647.5, 2662 -> 2662.5,
     3547 -> 3547.5, 3447 -> 3449, 3468 -> 3469)
  - appends EMF_2G/3G/4G/5G, EMF_Iliad/TIM/VF/W3 and EMF_Total using the same
    RSS method as parse_continuous_periods_window.add_aggregated_columns.
    Technology is assigned by frequency range, so the extra 2022 channels 3568
    and 3610 still count as 5G; operators use the canonical channel map.

Files are rewritten in place. They are tracked in git, so the originals can be
restored with `git checkout` if needed. The script is idempotent: re-running it
recomputes from the frequency columns rather than stacking new columns.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "datasets" / "processed"

# Files normalized in place (the reference 2023 file is left untouched).
FILES = ["UH2022.csv", "ED2022.csv", "UH2024.csv", "ED2024.csv"]

# Off-canonical 2022 labels mapped to the project's band centres (MHz).
SNAP = {
    2137.0: 2137.5, 2162.0: 2162.5, 2647.0: 2647.5, 2662.0: 2662.5,
    3547.0: 3547.5, 3447.0: 3449.0, 3468.0: 3469.0,
}

# Technology assigned by frequency range (MHz). Non-overlapping; covers every
# channel in the 2022/2023/2024 sets, including the extra 2022 5G channels.
TECH_RANGES = {
    "2G": [(930.0, 965.0)],
    "3G": [(2110.0, 2175.0)],
    "4G": [(760.0, 825.0), (1825.0, 1880.0), (2575.0, 2690.0)],
    "5G": [(3400.0, 3800.0)],
}

# Operator channel map (exact centres), from
# parse_continuous_periods_window.add_aggregated_columns.
OPERATOR_BANDS = {
    "Iliad": [763.0, 1835.0, 2150.0, 2635.0, 3630.0],
    "TIM":   [773.0, 806.0, 935.0, 2137.5, 2662.5, 3760.0, 3469.0],
    "VF":    [783.0, 816.0, 945.0, 1870.0, 2162.5, 2647.5, 3680.0],
    "W3":    [796.0, 955.0, 1850.0, 2120.0, 2585.0, 2680.0, 3610.0],
}

TECH_ORDER = ["2G", "3G", "4G", "5G"]
OPERATOR_ORDER = ["Iliad", "TIM", "VF", "W3"]
TIMESTAMP_CANDIDATES = ("Timestamp", "date", "Date")


def parse_timestamps(series: pd.Series) -> pd.Series:
    """Parse timestamps, tolerating ISO, day-first and US month-first formats."""
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().mean() > 0.5:  # mostly failed, retry day-first
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return parsed


def freq_value(col):
    """Return the (snapped) numeric MHz value of a frequency column, or None."""
    try:
        val = float(str(col).strip())
    except (TypeError, ValueError):
        return None
    if not (600.0 <= val <= 4000.0):
        return None
    return SNAP.get(val, val)


def rss(df: pd.DataFrame, cols: list) -> pd.Series:
    """Root-sum-square combine of E-field columns (V/m); exact 0 -> NaN."""
    val = (df[cols] ** 2).sum(axis=1).pow(0.5)
    return val.where(val != 0, np.nan)


def tech_of(mhz: float):
    for tech, ranges in TECH_RANGES.items():
        if any(lo <= mhz <= hi for lo, hi in ranges):
            return tech
    return None


def normalize(path: Path):
    # utf-8-sig transparently strips a leading BOM if one is present.
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    ts_col = next((c for c in TIMESTAMP_CANDIDATES if c in df.columns), None)
    if ts_col is None:
        raise ValueError(f"{path.name}: no timestamp column "
                         f"(looked for {TIMESTAMP_CANDIDATES})")
    ts = parse_timestamps(df[ts_col])

    # Clean frame indexed by timestamp, with canonical float labels.
    out = pd.DataFrame({"Timestamp": ts})
    n_snapped = 0
    for col in df.columns:
        if col == ts_col:
            continue
        mhz = freq_value(col)
        if mhz is None:
            continue  # drops EMF_* (on re-run), Unnamed/empty trailing columns
        if mhz != float(str(col).strip()):
            n_snapped += 1
        label = str(mhz)
        series = pd.to_numeric(df[col], errors="coerce")
        if label in out.columns:
            # Two raw columns snapped to the same centre: RSS-combine them.
            out[label] = np.sqrt(out[label].fillna(0.0) ** 2
                                 + series.fillna(0.0) ** 2)
        else:
            out[label] = series

    out = out.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    freq_labels = sorted((c for c in out.columns if c != "Timestamp"),
                         key=lambda c: float(c))
    freq_mhz = {c: float(c) for c in freq_labels}

    # Technology aggregations (assigned by frequency range).
    for tech in TECH_ORDER:
        cols = [c for c in freq_labels if tech_of(freq_mhz[c]) == tech]
        if cols:
            out[f"EMF_{tech}"] = rss(out, cols)

    # Operator aggregations (exact channel map).
    for op in OPERATOR_ORDER:
        wanted = set(OPERATOR_BANDS[op])
        cols = [c for c in freq_labels if freq_mhz[c] in wanted]
        if cols:
            out[f"EMF_{op}"] = rss(out, cols)

    # Total across all frequencies.
    out["EMF_Total"] = rss(out, freq_labels)

    agg_order = ([f"EMF_{t}" for t in TECH_ORDER]
                 + [f"EMF_{o}" for o in OPERATOR_ORDER] + ["EMF_Total"])
    agg_order = [c for c in agg_order if c in out.columns]
    out = out[["Timestamp"] + freq_labels + agg_order]

    out["Timestamp"] = out["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False)

    return out, freq_labels, agg_order, n_snapped, ts


def main():
    print("Normalizing UH/ED datasets in place")
    for name in FILES:
        path = PROCESSED / name
        if not path.exists():
            print(f"  missing: {path}")
            continue
        out, freq_labels, agg_order, n_snapped, ts = normalize(path)
        valid = ts.dropna()
        span = (f"{valid.min():%Y-%m-%d} to {valid.max():%Y-%m-%d}"
                if not valid.empty else "no valid timestamps")
        print(f"  {name}: {len(out):,} rows | {len(freq_labels)} freqs "
              f"({n_snapped} snapped) | {span}")
        print(f"      added: {', '.join(agg_order)}")
    print("Done.")


if __name__ == "__main__":
    main()
