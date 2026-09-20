"""Persistence baseline: repeat the last observed value across the horizon.

The reference every other forecast has to beat. It exists for the VoIP grids,
whose channels are step-like -- MOS repeats its previous value in 90-99% of
steps, and RTT / JIT / DJB / SNR in 80-92% -- so a model can post a low error by
learning nothing but "carry the last reading forward". Without this column an
MSE of 0.4 on those datasets is uninterpretable.

Zero parameters and no gradient, so it runs under exp.no_train like ARIMA: the
epoch loop and the checkpoint round-trip are skipped, while the split-conformal
calibration on val and the test evaluation run normally and write the same
sl_test/* keys (and the same grid resume marker) as every other cell. Its
conformal intervals are honest for the same reason ARIMA's are -- the predictor
is fixed by construction, so calibration and test residuals come from the same
model.

The forecast is made in the loader's standardized space, exactly like every
other model's, so the metrics are directly comparable.
"""

import torch.nn as nn


class Persistence(nn.Module):
    """Last-value forecast with the (B, C, L) -> (B, C, pred_len) interface.

    Args:
        pred_len (int): Forecast horizon.
    """

    def __init__(self, pred_len):
        super(Persistence, self).__init__()
        self.pred_len = pred_len

    def forward(self, x):
        # Rank-agnostic on purpose: the joint grid hands this (B, C, seq_len)
        # and the per-band grid (B, 1, seq_len), and both must come back with
        # the horizon on the last axis. expand is a view, so contiguous() is
        # what makes the result safe to write into downstream.
        last = x[..., -1:]
        return last.expand(*x.shape[:-1], self.pred_len).contiguous()
