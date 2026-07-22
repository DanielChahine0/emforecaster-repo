"""Outage-fill masking: flat-filled long gaps must never be trained on or scored.

The 2023 university file contains a ~10k-step whole-station outage that
_repair_missing_values flat-fills; it occupies ~64% of the chronological test
split, so without masking the benchmark mostly scores constant-prediction.
These tests pin the three layers of the fix: gap detection, window exclusion in
ForecastingDataset, and the end-to-end loader plumbing.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from forecasting.multivariate.utils.dataloading import (
    _long_gap_cells,
    _repair_missing_values,
    load_forecasting,
)
from forecasting.multivariate.utils.datasets import ForecastingDataset


def _frame_with_gap(n=100, gap_start=40, gap_len=20):
    df = pd.DataFrame({"a": np.arange(n, dtype=float), "b": np.arange(n, dtype=float)})
    df.iloc[gap_start : gap_start + gap_len, :] = np.nan
    return df


def test_long_gap_cells_flags_only_runs_longer_than_gap():
    df = _frame_with_gap(gap_start=40, gap_len=20)
    df.iloc[5, 0] = np.nan  # short gap: repairable, not synthetic

    cells = _long_gap_cells(df, max_interp_gap=10)
    assert cells.iloc[40:60].all().all()
    assert not cells.iloc[5, 0]
    assert cells.to_numpy().sum() == 20 * 2

    # A run exactly at the threshold is still repairable (strictly longer flags).
    cells = _long_gap_cells(df, max_interp_gap=20)
    assert not cells.iloc[40:60].any().any()


def test_repair_interpolates_short_and_flat_fills_long_gaps():
    df = _frame_with_gap(gap_start=40, gap_len=20)
    df.iloc[5, 0] = np.nan

    repaired = _repair_missing_values(df, max_interp_gap=10)
    assert not repaired.isna().any().any()
    # Short gap: linear interpolation between neighbors 4 and 6.
    assert repaired.iloc[5, 0] == pytest.approx(5.0)
    # Long gap: flat forward fill of the last real reading (39), no ramp.
    assert (repaired.iloc[40:60, 0] == 39.0).all()


def test_forecasting_dataset_without_flags_keeps_legacy_window_count():
    data = np.random.rand(3, 50)
    ds = ForecastingDataset(data, seq_len=8, pred_len=4)
    assert len(ds) == 50 - 8 - 4

    ds_masked = ForecastingDataset(data, 8, 4, synthetic=np.zeros(50, dtype=bool))
    assert len(ds_masked) == len(ds)
    x0, y0 = ds[0]
    x0m, y0m = ds_masked[0]
    assert torch.equal(x0, x0m) and torch.equal(y0, y0m)


def test_forecasting_dataset_excludes_windows_overlapping_flags():
    data = np.random.rand(2, 50)
    synthetic = np.zeros(50, dtype=bool)
    synthetic[20:25] = True
    seq_len, pred_len = 8, 4
    window = seq_len + pred_len

    ds = ForecastingDataset(data, seq_len, pred_len, synthetic=synthetic)
    # Valid starts: [i, i + 12) must miss [20, 25) -> i <= 8 or i >= 25.
    expected = [i for i in range(50 - window) if i + window <= 20 or i >= 25]
    assert list(ds.valid_starts) == expected
    assert len(ds) == len(expected)

    # Every emitted window carries only clean steps.
    clean = data[:, ~synthetic]
    for i in range(len(ds)):
        x, y = ds[i]
        assert not np.isin(
            x.numpy(), data[:, synthetic].astype(np.float32)
        ).any()


def test_forecasting_dataset_rejects_misaligned_flags():
    with pytest.raises(ValueError):
        ForecastingDataset(np.random.rand(2, 50), 8, 4, synthetic=np.zeros(49, bool))


@pytest.mark.parametrize("dataset", ["ED2022", "UH2022"])
def test_clean_datasets_have_no_synthetic_rows(dataset):
    (train, train_syn), (val, val_syn), (test, test_syn) = load_forecasting(
        dataset_name=dataset, seq_len=192, pred_len=96, return_synthetic=True
    )
    for syn in (train_syn, val_syn, test_syn):
        assert syn is not None and not syn.any()
    # No flags set -> window counts identical to the unmasked behavior.
    ds = ForecastingDataset(test, 192, 96, synthetic=test_syn)
    assert len(ds) == test.shape[1] - 192 - 96


def test_emf2023_test_split_excludes_the_outage():
    (train, train_syn), (val, val_syn), (test, test_syn) = load_forecasting(
        dataset_name="EMF_2023_university_5_with_PTV",
        seq_len=192,
        pred_len=96,
        return_synthetic=True,
    )
    # The whole-station outage dominates the test split.
    assert test_syn.mean() > 0.5
    # Train and val only have short (repaired) dropouts.
    assert not train_syn.any() and not val_syn.any()

    ds = ForecastingDataset(test, 192, 96, synthetic=test_syn)
    window = 192 + 96
    assert 0 < len(ds) < test.shape[1] - window
    # No valid window may touch a flat-filled step.
    starts = ds.valid_starts
    flagged = np.flatnonzero(test_syn)
    for s in (starts[0], starts[len(starts) // 2], starts[-1]):
        assert not test_syn[s : s + window].any()
    # Exhaustive: windowed cumulative count of flags is zero at every kept start.
    cumsum = np.concatenate(([0], np.cumsum(test_syn)))
    assert (cumsum[starts + window] - cumsum[starts] == 0).all()
