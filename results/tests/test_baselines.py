"""The lookback-mean baseline behind runs.csv's skill column.

The production code computes the trivial forecast's error from running sums;
here it is checked against the plain per-window loop, on a dataset whose
windows are masked (the VoIP trace joins), and pinned on ED2022 where the
Transformer's collapsed seeds were measured against it.
"""

import numpy as np
import pytest

from emfres.baselines import lookback_mean_mse, skill


def _direct(dataset, seq_len, pred_len, train_split, val_split):
    from forecasting.multivariate.utils.dataloading import load_forecasting
    from forecasting.multivariate.utils.datasets import ForecastingDataset

    _, _, (test, syn) = load_forecasting(
        dataset_name=dataset, seq_len=seq_len, pred_len=pred_len,
        train_split=train_split, val_split=val_split, return_synthetic=True)
    test = np.asarray(test, dtype=np.float64)
    windows = ForecastingDataset(test, seq_len, pred_len, synthetic=syn)
    starts = windows.valid_starts if windows.valid_starts is not None else range(len(windows))
    se = np.zeros(test.shape[0])
    n = 0
    for s in starts:
        x = test[:, s:s + seq_len]
        y = test[:, s + seq_len:s + seq_len + pred_len]
        se += ((y - x.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
        n += pred_len
    return se / n


def test_running_sums_match_the_per_window_loop_on_masked_windows():
    fast = lookback_mean_mse("voip_mobile", 24, 6, 0.6, 0.2)
    slow = _direct("VOIP_MOBILE", 24, 6, 0.6, 0.2)
    assert fast.shape == (6,)
    np.testing.assert_allclose(fast, slow, rtol=1e-9)


def test_ed2022_benchmark_value_is_pinned():
    # The value the Transformer's collapsed seeds (1.0073, 1.0074) sit on.
    assert lookback_mean_mse("ed2022", 336, 96, 0.7, 0.1).mean() == pytest.approx(1.0075, abs=5e-5)


def test_datasets_without_a_joint_loader_have_no_baseline():
    assert lookback_mean_mse("det", 336, 96, 0.7, 0.1) is None


def test_skill_is_one_minus_the_error_ratio():
    assert skill(0.5, 1.0) == pytest.approx(0.5)
    assert skill(1.2, 1.0) == pytest.approx(-0.2)
    assert skill(None, 1.0) is None
    assert skill(0.5, 0.0) is None
