"""Per-band ARIMA wrapped as a zero-parameter nn.Module.

ARIMA is univariate, so a "multivariate" cell here is C independent ARIMAs
stacked back into the (B, C, pred_len) layout every other model in the grid
emits. Wrapping it as an nn.Module rather than writing a standalone runner is
what buys the rest of the pipeline for free, and identically rather than
approximately: the same window enumeration and outage masking
(utils/datasets.py::ForecastingDataset), the same train/val/test boundaries and
train-only StandardScaler, the same split-conformal calibration on the val split
(exp/exp.py::test), the same log.json keys, the same logs/ layout, and therefore
the same grid resume marker.

Protocol. One fit per (dataset, band) on the scaled TRAIN split, then
``ARIMAResults.apply(window)`` per evaluation window. ``apply`` defaults to
refit=False, so it keeps the fitted parameters and only runs a Kalman pass over
the new window with the filter re-initialized: the model sees exactly seq_len
points of context, which is the same information the deep models get. Do not
replace this with one continuous pass over the test series -- that would give
ARIMA unbounded context and void the comparison.

The train split carries no seq_len term (utils/dataloading.py, train_idx =
[0, num_train]), so the fit is identical across all four grid windows and across
both the joint and per-band grids.

Two defects in the repo's reference script (analysis/general/arima.py) are
deliberately not carried over. It calls ``train.ravel()``, which flattens a
(C, T) array into one series with every band concatenated end to end -- harmless
only because it is pointed at a single-channel file. And it swallows failures
with a bare ``except: continue``, which would silently shrink the evaluated
window set and desynchronize ARIMA's support from every other model's, biasing
both the point metrics and the conformal scores. Here a failed window raises, so
the grid records a loud [FAIL] cell.

Cost, measured on this repo's data (statsmodels 0.14.6): ~2.4 ms per
apply+forecast at pred_len 96 and ~3.6 ms at 336, and ~0.5-2.2 s per fit. If a
sweep ever needs to be faster, the two lossless levers are caching the fits
across cells and parallelising the band loop; striding the evaluation windows is
NOT one, because it would make ARIMA's support differ from every other model's.
"""

import warnings

import numpy as np
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA as _ARIMA

from forecasting.multivariate.utils.dataloading import load_forecasting


class ARIMAForecaster(nn.Module):
    """Classical ARIMA baseline with the (B, C, L) -> (B, C, pred_len) interface.

    Args:
        dataset (str): Dataset name, used to reload the train split for fitting.
        seq_len (int): Lookback length. Only used to reload the split; the model
            forecasts from whatever context length it is handed.
        pred_len (int): Forecast horizon.
        order (tuple): statsmodels (p, d, q).
        trend (str): statsmodels trend term.
        train_split, val_split, scale: must match the run's data config so the
            fit sees the same scaled train series the loader produced.
        target_channel (int | None): Fit and forecast this band only (the
            per-channel grid). None fits every band (the joint grid).
    """

    def __init__(
        self,
        dataset,
        seq_len,
        pred_len,
        order=(1, 0, 1),
        trend="c",
        train_split=0.7,
        val_split=0.1,
        scale=True,
        target_channel=None,
    ):
        super().__init__()
        self.dataset = dataset
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.order = tuple(order)
        self.trend = trend
        self.train_split = train_split
        self.val_split = val_split
        self.scale = scale
        self.target_channel = target_channel
        self._fitted = None
        self._bands = None

    def _train_series(self):
        """The scaled train split as (num_bands, T), with outage fill as NaN.

        Loaded with univariate=False even in the per-band grid: the split
        boundaries and the StandardScaler are applied before load_forecasting
        branches on `univariate`, and the univariate branch windows
        ``train_data.T[i]``, so band i is the same series either way. Reading the
        contiguous array directly avoids trying to reconstruct one from the
        pre-windowed (x, y) tensors that branch returns.

        Steps flat-filled across a long sensor outage are set to NaN rather than
        fitted on. statsmodels' Kalman filter treats NaN as missing, so the fit
        skips them instead of learning a constant. The evaluation windows never
        contain them at all -- ForecastingDataset drops any window that overlaps
        one -- so this only affects the fit.
        """
        (train, synthetic), _, _ = load_forecasting(
            dataset_name=self.dataset,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            window_stride=1,
            scale=self.scale,
            train_split=self.train_split,
            val_split=self.val_split,
            univariate=False,
            return_synthetic=True,
        )
        series = np.asarray(train, dtype=np.float64)
        if synthetic is not None:
            series = series.copy()
            series[:, np.asarray(synthetic, dtype=bool)] = np.nan
        return series

    def _fit_bands(self):
        series = self._train_series()
        self._bands = (
            [self.target_channel]
            if self.target_channel is not None
            else list(range(series.shape[0]))
        )
        with warnings.catch_warnings():
            # statsmodels warns per fit about the missing date index and about
            # non-stationary starting parameters. Scoped to the fit rather than
            # set globally (the reference script's warnings.simplefilter at
            # import time silences the whole process).
            warnings.simplefilter("ignore")
            self._fitted = [
                _ARIMA(series[c], order=self.order, trend=self.trend).fit()
                for c in self._bands
            ]

    def fit_info(self):
        """Per-band fit provenance. A fit-once model writes no checkpoint, so this
        is the only record of what was actually estimated."""
        if self._fitted is None:
            return None
        return {
            "order": list(self.order),
            "trend": self.trend,
            "bands": [
                {
                    "channel": int(band),
                    "params": [float(v) for v in res.params],
                    "aic": float(res.aic),
                    "llf": float(res.llf),
                    "converged": bool(
                        getattr(res, "mle_retvals", {}).get("converged", True)
                    ),
                }
                for band, res in zip(self._bands, self._fitted)
            ],
        }

    @torch.no_grad()
    def forward(self, x):
        """(B, C, L) -> (B, C, pred_len).

        The output layout is load-bearing and fails silently if wrong:
        Experiment.coverage transposes a 3-D forecast to (n, pred_len, C), so
        emitting (B, pred_len, C) would not raise -- it would take the conformal
        quantiles along the wrong axis and divide alpha by C instead of pred_len.
        """
        if self._fitted is None:
            self._fit_bands()

        ctx = x.detach().cpu().to(torch.float64).numpy()
        batch, channels, _ = ctx.shape
        if channels != len(self._fitted):
            raise ValueError(
                f"ARIMA fitted {len(self._fitted)} band(s) but the batch has "
                f"{channels} channel(s); target_channel/num_channels disagree "
                f"with the loader."
            )

        out = np.empty((batch, channels, self.pred_len), dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for c, res in enumerate(self._fitted):
                for b in range(batch):
                    # No bare except: a failed window must fail the cell loudly
                    # rather than quietly shrink the evaluated support.
                    out[b, c] = res.apply(ctx[b, c]).forecast(steps=self.pred_len)
        return torch.from_numpy(out).to(dtype=x.dtype, device=x.device)
