#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run EMF level generation and heatmap generation together.

This is a thin driver that calls the two existing entry points in order:
  1. plot_emf_figure4.main()  writes into result/EMF levels/<dataset>/
  2. run_heatmaps.main()      writes into result/Heatmaps/<dataset>/

Each underlying script keeps its own DATASETS list and configuration, so edit
those files to change which datasets are processed or how they are styled.

The dataset normalization step (normalize_datasets.py) is intentionally left
out; run it separately when the processed CSVs need to be rebuilt.
"""

import plot_emf_figure4
import run_heatmaps


def main():
    print("Step 1 of 2: EMF levels")
    plot_emf_figure4.main()

    print("\nStep 2 of 2: Heatmaps")
    run_heatmaps.main()

    print("\nAll done.")


if __name__ == "__main__":
    main()
