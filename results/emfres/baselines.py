"""The lookback-mean forecast: the error a model scores by learning nothing.

PatchTST, TSMixer, EMForecaster, the Transformer and LSTM normalize each input
window (RevIN) before forecasting, so a network whose output never leaves zero
in that normalized space forecasts, once denormalized, the mean of its own
lookback window. That is what a collapsed run of one of them looks like from
the outside. DLinear has no RevIN in any paradigm (its job config's
revin/revout flags do nothing), and neither do ARIMA and Persistence, so for
those three the lookback mean is a reference point, not where a collapse
lands. On the EMF benchmark the Transformer does exactly this at seeds 69422
and 69423 on every dataset, and its test MSE matches the lookback-mean
forecast's to within 0.05% (ED2022: 1.0073 against 1.0075) while its one
trained seed is 50% better.

``skill`` is 1 - mse / lookback_mse, measured on the same windows: positive
when a model beats the trivial forecast, zero when it matches it, negative when
it is worse (ARIMA is, by 1.2% on ED2022, and matches it on UH2022).

What skill does NOT give is an automatic collapse detector, and none is
derived from it here. Zero skill has several causes that look alike in one
number: a run that collapsed, a frequency band that no model can forecast
better than its own mean, and a weak model on a hard target. Checked across all
~1,900 scored runs, band-level skill forms a continuum through zero with no gap
to put a threshold in, and even a seed-level rule ("some seeds learned, some
did not") flags VoIP cells whose seeds are merely small and spread out. The
Transformer's collapse is identified by the conjunction of evidence instead --
a flat training loss, skill of 0.000, and two failed seeds whose test MSEs
agree to within 2e-4 -- and is documented rather than flagged. A training-loss
rule fails too: per-channel DLinear barely moves its loss because it *starts*
at a moving-average forecast with a tiny learning rate.

``train_mean_mse`` is the second forecast that learns nothing: the training
split's mean, which is 0 in the scaled space every model forecasts in (the
scalers are fit on the training split). A model without RevIN that collapses
lands there instead: the single-series Transformer, which has no RevIN in
its code, sits at 1.1619 on ptv2 at L336/O96 -- the training-mean MSE to four
decimals -- at three of its five seeds.

Computed from the data with the training package's own loader, so the split,
the train-only scaler, the outage and trace-join masks and the window starts
are exactly the ones the runs were scored on. The single-series tree
(det, ptv, ptv2) is read with its own package's loader,
forecasting/univariate, the one its runs trained on.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np

# frames dataset slug -> the name load_forecasting takes, for the joint-channel
# corpora (forecasting/multivariate).
LOADER_NAME = {
    "ed2022": "ED2022",
    "uh2022": "UH2022",
    "emf_2023_university_5_with_ptv": "EMF_2023_university_5_with_PTV",
    "ed2024": "ED2024",
    "uh2024": "UH2024",
    "voip_mobile": "VOIP_MOBILE",
    "voip_fixed": "VOIP_FIXED",
}

# The single-series tree's slugs -> the name forecasting/univariate's
# load_forecasting takes. One broadband series each, scored as one channel.
SERIES_LOADER_NAME = {"det": "rf_emf_det", "ptv": "rf_emf_ptv", "ptv2": "rf_emf_ptv2"}

_CACHE: dict = {}

_REPO = Path(__file__).resolve().parents[2]
_NAMES: dict = {}


def band_names(dataset: str) -> Optional[list]:
    """The dataset's channel labels in the loader's channel order, or None.

    The header of data/multivariate/processed/<name>.csv without the columns
    load_forecasting drops: Timestamp, the EMF_* aggregates and the broadband
    PTV on the EMF corpus (labels are the band frequency in MHz), Timestamp
    and Trace on VoIP (the signal names). Only the header is read. Channel i
    of a joint run's sl_test/mse_per_channel and the per-channel run ch<i> are
    both column i of this list (tests/test_frames.py holds it to the loader's
    channel count).
    """
    name = LOADER_NAME.get(str(dataset).lower())
    if name is None:
        return None
    if name not in _NAMES:
        path = _REPO / "data" / "multivariate" / "processed" / f"{name}.csv"
        try:
            with open(path, newline="") as f:
                header = next(csv.reader(f))
        except (OSError, StopIteration):
            _NAMES[name] = None
        else:
            if name.startswith("VOIP_"):
                keep = [c for c in header if c not in {"Timestamp", "Trace"}]
            else:
                keep = [c for c in header
                        if c != "Timestamp" and not c.startswith("EMF_") and c != "PTV"]
            _NAMES[name] = keep
    return _NAMES[name]


def lookback_mean_mse(dataset: str, seq_len: int, pred_len: int,
                      train_split: float, val_split: float) -> Optional[np.ndarray]:
    """Per-channel test MSE of forecasting every window's lookback mean.

    Returns an array with one entry per channel, in the loader's channel order
    (a per-channel run's ``channel`` indexes it; a joint run's MSE is its mean),
    one entry for a single series, or None for a dataset no loader knows.
    """
    found = _trivial(dataset, seq_len, pred_len, train_split, val_split)
    return None if found is None else found[0]


def train_mean_mse(dataset: str, seq_len: int, pred_len: int,
                   train_split: float, val_split: float) -> Optional[np.ndarray]:
    """Per-channel test MSE of forecasting the training split's mean (0 when scaled).

    Same windows, channel order and None as lookback_mean_mse.
    """
    found = _trivial(dataset, seq_len, pred_len, train_split, val_split)
    return None if found is None else found[1]


def _series(dataset: str, seq_len: int, pred_len: int, train_split: float,
            val_split: float) -> tuple:
    """(lookback, train mean) MSE of a single series, as one-channel arrays."""
    from forecasting.univariate.utils.dataloading import load_forecasting

    _, _, (x, y) = load_forecasting(
        dataset_name=SERIES_LOADER_NAME[dataset], seq_len=seq_len, pred_len=pred_len,
        train_split=train_split, val_split=val_split, univariate=True, scale=True)
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)    # (windows, seq_len)
    y = np.asarray(y, dtype=np.float64).reshape(len(y), -1)    # (windows, pred_len)
    lookback = ((y - x.mean(axis=1, keepdims=True)) ** 2).mean()
    return np.array([lookback]), np.array([(y * y).mean()])


def _trivial(dataset: str, seq_len: int, pred_len: int, train_split: float,
             val_split: float) -> Optional[tuple]:
    """(lookback-mean, train-mean) per-channel test MSE, cached; None if unknown."""
    dataset = str(dataset).lower()
    if dataset not in LOADER_NAME and dataset not in SERIES_LOADER_NAME:
        return None
    name = LOADER_NAME.get(dataset) or SERIES_LOADER_NAME[dataset]
    key = (name, seq_len, pred_len, round(float(train_split), 6), round(float(val_split), 6))
    if key in _CACHE:
        return _CACHE[key]
    if dataset in SERIES_LOADER_NAME:
        _CACHE[key] = _series(dataset, seq_len, pred_len, train_split, val_split)
        return _CACHE[key]
    from forecasting.multivariate.utils.dataloading import load_forecasting
    from forecasting.multivariate.utils.datasets import ForecastingDataset

    _, _, (test, synthetic) = load_forecasting(
        dataset_name=name, seq_len=seq_len, pred_len=pred_len,
        train_split=train_split, val_split=val_split, return_synthetic=True)
    test = np.asarray(test, dtype=np.float64)             # (channels, time)
    windows = ForecastingDataset(test, seq_len, pred_len, synthetic=synthetic)
    starts = (np.asarray(windows.valid_starts) if windows.valid_starts is not None
              else np.arange(len(windows)))

    # Squared error of the lookback-mean forecast over every window, from
    # running sums: for target y and forecast m, sum (y - m)^2 over the
    # horizon is sum y^2 - 2 m sum y + O m^2.
    zero = np.zeros((test.shape[0], 1))
    c1 = np.concatenate([zero, np.cumsum(test, axis=1)], axis=1)
    c2 = np.concatenate([zero, np.cumsum(test * test, axis=1)], axis=1)
    mean = (c1[:, starts + seq_len] - c1[:, starts]) / seq_len
    sy = c1[:, starts + seq_len + pred_len] - c1[:, starts + seq_len]
    syy = c2[:, starts + seq_len + pred_len] - c2[:, starts + seq_len]
    se = syy - 2.0 * mean * sy + pred_len * mean * mean
    n = len(starts) * pred_len
    _CACHE[key] = (se.sum(axis=1) / n, syy.sum(axis=1) / n)
    return _CACHE[key]


def skill(mse: Optional[float], baseline: Optional[float]) -> Optional[float]:
    """1 - mse / baseline, or None when either is missing."""
    if mse is None or not baseline:
        return None
    return 1.0 - mse / baseline
