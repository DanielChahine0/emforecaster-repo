"""Single-series ARIMA wrapped as a zero-parameter nn.Module.

The single-series counterpart of forecasting/multivariate/models/arima.py, ported
on 2026-08-03 so the univariate_series grid carries the same seven models as the
other two. Same protocol, same statsmodels calls; the differences are all
consequences of the package it lives in:

- This package's loader has no outage masking (`return_synthetic` and the
  `_long_gap_cells` machinery are multivariate-only additions). Only EMF_2023 has
  the long flat-filled sensor gaps those exist for, and it is not one of this
  grid's datasets, so nothing is dropped by leaving them out.
- The loader passes `clip`/`thresh`/`average_italy` from the data config, and any
  of the three changes the series before it is split and scaled. They are
  threaded through here rather than assumed off, because a fit on a different
  series than the windows were cut from would be wrong in a way no shape check
  would catch.
- The dataset yields 2-D (B, L) inputs (utils/datasets.py::
  UnivariateForecastingDataset keeps no channel axis), unlike the multivariate
  (B, C, L). forward mirrors whichever rank it is handed.

Wrapping ARIMA as an nn.Module rather than writing a standalone runner is what
buys the rest of the pipeline for free, and identically rather than
approximately: the same window enumeration, the same train/val/test boundaries
and train-only StandardScaler, the same split-conformal calibration on the val
split (exp/exp.py::test), the same log.json keys, the same logs/ layout, and
therefore the same grid resume marker.

Protocol. One fit per series on the scaled TRAIN split, then
``ARIMAResults.apply(window)`` per evaluation window. ``apply`` defaults to
refit=False, so it keeps the fitted parameters and only runs a Kalman pass over
the new window with the filter re-initialized: the model sees exactly seq_len
points of context, which is the same information the deep models get. Do not
replace this with one continuous pass over the test series -- that would give
ARIMA unbounded context and void the comparison.

The train split carries no seq_len term (utils/dataloading.py, train_idx =
[0, num_train]), so the fit is identical across all four grid windows.

Two defects in the repo's reference script (analysis/general/arima.py) are
deliberately not carried over. It calls ``train.ravel()``, which flattens a
(C, T) array into one series with every channel concatenated end to end --
harmless only because it is pointed at a single-channel file. And it swallows
failures with a bare ``except: continue``, which would silently shrink the
evaluated window set and desynchronize ARIMA's support from every other model's,
biasing both the point metrics and the conformal scores. Here a failed window
raises, so the grid records a loud [FAIL] cell.
"""

import warnings

import numpy as np
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA as _ARIMA

from forecasting.univariate.utils.dataloading import load_forecasting


class ARIMAForecaster(nn.Module):
    """Classical ARIMA baseline with the (B, [C,] L) -> (B, [C,] pred_len) interface.

    Args:
        dataset (str): Dataset name, used to reload the train split for fitting.
        seq_len (int): Lookback length. Only used to reload the split; the model
            forecasts from whatever context length it is handed.
        pred_len (int): Forecast horizon.
        order (tuple): statsmodels (p, d, q).
        trend (str): statsmodels trend term.
        train_split, val_split, scale, clip, thresh, average_italy: must match the
            run's data config so the fit sees the same scaled train series the
            loader produced.
        target_channel (int | None): Fit and forecast this channel only. None
            fits every channel the file exposes (one, for this grid's datasets).
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
        clip=False,
        thresh=1.0,
        average_italy=False,
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
        self.clip = clip
        self.thresh = thresh
        self.average_italy = average_italy
        self.target_channel = target_channel
        self._fitted = None
        self._channels = None

    def _train_series(self):
        """The scaled train split as (num_channels, T).

        Loaded with univariate=False even though the grid runs univariate=True:
        the split boundaries and the StandardScaler are applied before
        load_forecasting branches on `univariate`, and the univariate branch
        windows ``train_data.T[i]``, so channel i is the same series either way.
        Reading the contiguous array directly avoids trying to reconstruct one
        from the pre-windowed (x, y) tensors that branch returns.
        """
        train, _, _ = load_forecasting(
            dataset_name=self.dataset,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            window_stride=1,
            scale=self.scale,
            clip=self.clip,
            thresh=self.thresh,
            train_split=self.train_split,
            val_split=self.val_split,
            univariate=False,
            average_italy=self.average_italy,
        )
        return np.asarray(train, dtype=np.float64)

    def _fit_channels(self):
        series = self._train_series()
        self._channels = (
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
                for c in self._channels
            ]

    def fit_info(self):
        """Per-channel fit provenance. A fit-once model writes no checkpoint, so
        this is the only record of what was actually estimated."""
        if self._fitted is None:
            return None
        return {
            "order": list(self.order),
            "trend": self.trend,
            "channels": [
                {
                    "channel": int(channel),
                    "params": [float(v) for v in res.params],
                    "aic": float(res.aic),
                    "llf": float(res.llf),
                    "converged": bool(
                        getattr(res, "mle_retvals", {}).get("converged", True)
                    ),
                }
                for channel, res in zip(self._channels, self._fitted)
            ],
        }

    @torch.no_grad()
    def forward(self, x):
        """(B, L) -> (B, pred_len), or (B, C, L) -> (B, C, pred_len).

        The rank is mirrored rather than normalized to 3-D on the way out. This
        package's prepare_output_and_target squeezes the forecast before the
        criterion sees it, and its UnivariateForecastingDataset keeps targets at
        (B, pred_len); returning (B, 1, pred_len) against a (B, pred_len) target
        would not raise, it would broadcast the loss to (B, B, pred_len).
        """
        if self._fitted is None:
            self._fit_channels()

        squeeze_back = x.dim() == 2
        ctx = x.detach().cpu().to(torch.float64).numpy()
        if squeeze_back:
            ctx = ctx[:, None, :]
        batch, channels, _ = ctx.shape
        if channels != len(self._fitted):
            raise ValueError(
                f"ARIMA fitted {len(self._fitted)} channel(s) but the batch has "
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
        if squeeze_back:
            out = out[:, 0, :]
        return torch.from_numpy(out).to(dtype=x.dtype, device=x.device)
