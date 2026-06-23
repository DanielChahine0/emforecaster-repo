#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Driver for heatmap_generator.py.

heatmap_generator.py is a function library with no entry point, so this small
runner loads a processed dataset and calls generate_all_heatmaps(...). Conventions
(project root anchoring, timestamped result/ folder) mirror plot_emf_figure4.py.

Outputs (into src/result/heatmaps_<dataset>_<timestamp>/heatmaps/):
  - heatmap_trio_<dataset>.png/.pdf            (Frequency | Technology | Operators)
  - heatmap_<dataset>_-_frequency_bands.png/.pdf
  - heatmap_<dataset>_-_technology_bands.png/.pdf
  - heatmap_<dataset>_-_operators.png/.pdf
  - heatmap_all_datasets_comparison.*          (only when >= 2 datasets are listed)
"""

from pathlib import Path

import pandas as pd

from heatmap_generator import generate_all_heatmaps, get_actual_frequency_columns

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# (csv path, label, output folder) tuples. The label is shown in the figure
# titles; the folder is the sub-folder under result/Heatmaps/. With more than one
# dataset a comparison grid across all of them is also produced.
DATASETS = [
    (PROJECT_ROOT / "datasets/processed/EMF_2023_university_5_with_PTV.csv",
     "EMF_2023_university", "EMF2023"),
    (PROJECT_ROOT / "datasets/processed/UH2022.csv", "UH_2022", "UH2022"),
    (PROJECT_ROOT / "datasets/processed/ED2022.csv", "ED_2022", "ED2022"),
    (PROJECT_ROOT / "datasets/processed/UH2024.csv", "UH_2024", "UH2024"),
    (PROJECT_ROOT / "datasets/processed/ED2024.csv", "ED_2024", "ED2024"),
]

# Per-frequency extreme-spike removal, mirroring plot_emf_figure4.py. Some raw
# per-frequency columns contain isolated corrupted samples (e.g. a single
# ~119 V/m reading at 1835 MHz) that would otherwise dominate the Pearson
# correlation and flip the sign of an entire row/column. Any value above
# FREQ_CLIP_FACTOR x the FREQ_CLIP_QUANTILE of a frequency's own data is treated
# as an artifact and set to NaN before heatmap_generator's ffill/bfill + corr.
# The EMF_* aggregations are left untouched, so the technology/operator heatmaps
# are unaffected. Set FREQ_CLIP_QUANTILE = None to disable.
FREQ_CLIP_QUANTILE = 0.999
FREQ_CLIP_FACTOR = 3.0


def clip_frequency_artifacts(df):
    """Mask per-frequency spikes to NaN. Returns (df, n_clipped, affected_cols)."""
    freq_cols = get_actual_frequency_columns(df)
    if FREQ_CLIP_QUANTILE is None or not freq_cols:
        return df, 0, []
    thresh = df[freq_cols].quantile(FREQ_CLIP_QUANTILE) * FREQ_CLIP_FACTOR
    mask = df[freq_cols] > thresh
    counts = mask.sum()
    affected = [str(c) for c in counts.index[counts > 0]]
    df = df.copy()
    df[freq_cols] = df[freq_cols].mask(mask)
    return df, int(counts.sum()), affected


def main():
    print("Heatmap generation")

    datasets_info = []
    for csv_path, label, folder in DATASETS:
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

    out_root = PROJECT_ROOT / "src" / "result" / "Heatmaps"
    out_root.mkdir(parents=True, exist_ok=True)

    generate_all_heatmaps(datasets_info, out_root)
    print("\nDone. Files saved in:", out_root.resolve())


if __name__ == "__main__":
    main()
