#!/usr/bin/env python3
"""Build the two pooled VoIP datasets from the raw per-codec traces.

    python data/ML_DATASETS/build_voip.py            # writes the two CSVs
    python data/ML_DATASETS/build_voip.py --check    # report only, write nothing

Input: 16 whitespace-free CSV traces under data/ML_DATASETS/{Mobile,Fixed}/,
one per (access type, codec), each a header row `Time,MOS,BW,RTT,JIT,DJB,SNR`
followed by rows on a contiguous 1..N step grid.

Output: data/multivariate/processed/VOIP_MOBILE.csv and VOIP_FIXED.csv, one
file per access type, holding that type's eight traces stacked in a fixed
codec order with a `Trace` column naming the codec. The loader
(utils/dataloading.py) splits each trace on its own before pooling, so the
stacking order here only has to be stable, not meaningful.

Why pool by access type rather than per trace or all sixteen together: a single
trace is 161-820 rows, which at the benchmark window leaves a validation split
too small to calibrate a conformal interval on, while pooling all sixteen mixes
mobile and fixed access -- the contrast most worth keeping separate. See
.personal/PLAN.md §5.3.

Cleaning, deliberately minimal (PLAN.md decision 7):
  * JIT above JIT_MAX is not a measurement -- mob_g729 carries values up to
    9.9e11 against a plausible range of tens to hundreds -- so those cells are
    nulled and linearly interpolated WITHIN their own trace.
  * RTT outliers (up to 4450) are kept: a multi-second round trip on a bad
    mobile link is a real event, and it is exactly what a forecaster should see.
  * The leading RTT = 0 rows (no measurement yet at call setup) are kept.
Every edit is counted and printed, and the CSVs are committed, so the cleaning
is an auditable artifact rather than something that happens at load time.
"""

import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT_DIR = os.path.join(REPO, "data", "multivariate", "processed")

# Fixed codec order, so a rebuild reproduces the same file byte for byte.
CODECS = ["g711", "g722", "g729", "gsm", "mpeg16", "opus", "spx8000", "spx16000"]
ACCESS = {"Mobile": ("mob", "VOIP_MOBILE"), "Fixed": ("fix", "VOIP_FIXED")}

CHANNELS = ["MOS", "BW", "RTT", "JIT", "DJB", "SNR"]

# Jitter ceiling. The largest jitter anywhere in the corpus that is not part of
# the mob_g729 blow-up is 706, so this sits three orders of magnitude above the
# real data and cannot clip a plausible reading.
JIT_MAX = 1e6


def load_trace(access, codec):
    """One raw trace as a frame of CHANNELS, with its own cleaning report."""
    prefix, _ = ACCESS[access]
    path = os.path.join(HERE, access, f"{prefix}_{codec}.txt")
    df = pd.read_csv(path)

    expected = ["Time"] + CHANNELS
    if list(df.columns) != expected:
        raise ValueError(f"{path}: columns {list(df.columns)} != {expected}")
    # The loader's per-trace split assumes rows are in time order on a
    # contiguous grid; a gap would silently become a discontinuity inside a
    # window rather than between traces, where the boundary mask handles it.
    if not (df["Time"].to_numpy() == np.arange(1, len(df) + 1)).all():
        raise ValueError(f"{path}: Time is not a contiguous 1..N grid")

    df = df[CHANNELS].apply(pd.to_numeric, errors="coerce")
    report = {"rows": len(df), "nulled_jit": 0, "still_nan": 0, "max_jit_dropped": 0.0}

    absurd = df["JIT"] > JIT_MAX
    if absurd.any():
        report["nulled_jit"] = int(absurd.sum())
        report["max_jit_dropped"] = float(df.loc[absurd, "JIT"].max())
        df.loc[absurd, "JIT"] = np.nan

    # Interpolation is per trace by construction: this runs before any stacking,
    # so a repaired value is never drawn from a different call.
    df = df.interpolate(method="linear", limit_direction="both")
    report["still_nan"] = int(df.isna().sum().sum())
    return df, report


def build(access, check=False):
    prefix, name = ACCESS[access]
    frames, reports = [], {}
    for codec in CODECS:
        df, report = load_trace(access, codec)
        reports[f"{prefix}_{codec}"] = report
        df.insert(0, "Trace", codec)
        # The trace's own step index, kept so a row in the pooled file can be
        # traced back to its source. The loader drops it with Trace.
        df.insert(0, "Timestamp", np.arange(1, len(df) + 1))
        frames.append(df)

    pooled = pd.concat(frames, ignore_index=True)
    if pooled.isna().any().any():
        raise ValueError(f"{name}: NaNs survived cleaning")

    print(f"\n{name}  ({len(pooled)} rows, {len(CODECS)} traces, "
          f"{len(CHANNELS)} channels)")
    for trace, r in reports.items():
        note = ""
        if r["nulled_jit"]:
            note = (f"  <- nulled {r['nulled_jit']} JIT cell(s), "
                    f"max {r['max_jit_dropped']:.3g}, interpolated")
        print(f"    {trace:<16} {r['rows']:>4} rows{note}")

    if not check:
        os.makedirs(OUT_DIR, exist_ok=True)
        out = os.path.join(OUT_DIR, f"{name}.csv")
        pooled.to_csv(out, index=False)
        print(f"    wrote {os.path.relpath(out, REPO)}")
    return pooled


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report what would be built, write nothing")
    args = ap.parse_args()
    for access in ACCESS:
        build(access, check=args.check)
    print()


if __name__ == "__main__":
    main()
