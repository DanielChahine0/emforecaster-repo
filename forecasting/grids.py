"""Canonical experiment grids for the conformal-prediction point-forecasting sweep.

Each model entry is (display_label, on-disk model_id, job_name). The on-disk
model_id can differ from the label (the Transformer's is RF_EMF_Transformer) and
drives the run directory, so it must match the model's args.yaml exactly.

Three grids share the same WINDOWS so their cells line up one-to-one:

- "multivariate"      One model per cell, forecasting all 27/29 bands jointly.
- "univariate"        The per-band counterpart of the multivariate sweep: one
                      independent model PER BAND, over the same datasets, models
                      and windows, averaged back into one number per cell. Driven
                      by forecasting/perchannel.py, not forecasting/grid.py.
- "univariate_series" The original single-series sweep over the DET/PTV
                      monitoring CSVs. Unchanged; it was renamed from
                      "univariate" when that name was repurposed for the per-band
                      grid above.
"""

import csv
import os
import re

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WINDOWS = [(192, 96), (192, 336), (336, 96), (336, 336)]

MV_MODELS = [
    ("DLinear", "DLinear", "rf_emf_mv/dlinear"),
    ("PatchTST", "PatchTST", "rf_emf_mv/patchtst"),
    ("TSMixer", "TSMixer", "rf_emf_mv/tsmixer"),
    ("Transformer", "RF_EMF_Transformer", "rf_emf_mv/transformer"),
    ("EMForecaster", "EMForecaster", "rf_emf_mv/emforecaster"),
]
MV_DATASETS = ["ED2022", "UH2022", "EMF_2023_university_5_with_PTV"]

SERIES_MODELS = [
    ("DLinear", "DLinear", "rf_emf/dlinear"),
    ("PatchTST", "PatchTST", "rf_emf/patchtst"),
    ("TSMixer", "TSMixer", "rf_emf/tsmixer"),
    ("Transformer", "RF_EMF_Transformer", "rf_emf/baselines/transformer"),
    ("EMForecaster", "EMForecaster", "rf_emf/emforecaster"),
]
SERIES_DATASETS = ["rf_emf_det", "rf_emf_ptv", "rf_emf_ptv2"]

GRIDS = {
    "multivariate": {
        "models": MV_MODELS, "datasets": MV_DATASETS, "windows": WINDOWS, "alpha": 0.1,
    },
    # Deliberately the same models/datasets/windows as the multivariate grid: a
    # univariate cell is only meaningful if it is comparable to the multivariate
    # cell beside it. per_channel is what makes forecasting/perchannel.py fan a
    # cell out into one run per band instead of training it once.
    "univariate": {
        "models": MV_MODELS, "datasets": MV_DATASETS, "windows": WINDOWS, "alpha": 0.1,
        "per_channel": True,
    },
    "univariate_series": {
        "models": SERIES_MODELS, "datasets": SERIES_DATASETS, "windows": WINDOWS,
        "alpha": 0.1,
    },
}


# One window token: "336x96", "336_96", or the logs/ dir form "L336_O96".
_WINDOW_RE = re.compile(r"^L?(\d+)[x_]O?(\d+)$", re.IGNORECASE)


def parse_window_spec(spec):
    """Parse a --window value into [(seq_len, pred_len), ...].

    Accepts one or more comma-separated windows, each as 'LxO' (e.g. '336x96')
    or the logs/ directory form 'L336_O96'. Raises ValueError on a malformed
    token so a typo fails loudly instead of silently selecting nothing.
    """
    windows = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        match = _WINDOW_RE.match(token)
        if not match:
            raise ValueError(
                f"bad --window value {token!r}; use 'LxO' like '336x96' "
                f"or 'L336_O96'"
            )
        windows.append((int(match.group(1)), int(match.group(2))))
    if not windows:
        raise ValueError("empty --window value")
    return windows


def select_windows(spec, available):
    """The windows from `available` selected by `spec`, in grid order.

    `spec` None/empty -> all of `available` unchanged. Otherwise every requested
    window must exist in the grid (else ValueError), duplicates collapse, and the
    result keeps the grid's own ordering so runs and resume paths stay stable
    regardless of how the flag was typed.
    """
    if not spec:
        return list(available)
    requested = set(parse_window_spec(spec))
    missing = [w for w in requested if w not in set(available)]
    if missing:
        pretty = ", ".join(f"L{s}_O{p}" for s, p in sorted(missing))
        have = ", ".join(f"L{s}_O{p}" for s, p in available)
        raise ValueError(
            f"requested window(s) {pretty} not in grid windows ({have})"
        )
    return [w for w in available if w in requested]


def _is_band(column):
    """Whether a processed-CSV column becomes a model channel.

    Mirrors the column filter in multivariate/utils/dataloading.py: drop the
    timestamp, the EMF_* operator/technology/total aggregates (root-sum-squares of
    the bands, which would leak the target), and the broadband PTV column. Kept in
    step by hand rather than imported -- pulling the loader in here would drag
    torch into grid startup.
    """
    return not (column == "Timestamp" or column.startswith("EMF_") or column == "PTV")


def dataset_num_channels(dataset, root_dir=None):
    """Number of frequency-band channels the loader exposes for `dataset`.

    Read from the CSV header instead of hardcoded so the 27-vs-29 band split
    across the Italy datasets (and any file added later) needs no code change.
    This has to agree with the loader's filter or the per-band grid would run the
    wrong number of channels.
    """
    root = root_dir or _REPO_ROOT
    path = os.path.join(root, "data", "multivariate", "processed", f"{dataset}.csv")
    with open(path, newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f))
    return sum(1 for column in header if _is_band(column))
