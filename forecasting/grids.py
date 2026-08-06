"""Canonical experiment grids for the conformal-prediction point-forecasting sweep.

Each model entry is (display_label, on-disk model_id, job_name). The on-disk
model_id can differ from the label (the Transformer's is RF_EMF_Transformer) and
drives the run directory, so it must match the model's args.yaml exactly.

Four grids share the same WINDOWS so their cells line up one-to-one:

- "multivariate"      One model per cell, forecasting all 27/29 bands jointly.
- "univariate"        The per-band counterpart of the multivariate sweep: one
                      independent model PER BAND, over the same datasets, models
                      and windows, averaged back into one number per cell. Driven
                      by forecasting/perchannel.py, not forecasting/grid.py.
- "univariate_series" The original single-series sweep over the DET/PTV
                      monitoring CSVs. It was renamed from "univariate" when that
                      name was repurposed for the per-band grid above.
- "cluster"           The multivariate sweep run through CCM (Channel Clustering
                      Module, NeurIPS 2024), whose learned soft channel clustering
                      replaces the model's ordinary head. Adds a k dimension
                      (k_values) to every cell; four models, since CCM ships
                      wrappers for exactly four. See CLUSTER_MODELS below and
                      forecasting/multivariate/models/ccm.py.

The first three carry the same seven models since 2026-08-03 -- EMForecaster, DLinear,
PatchTST, TSMixer, Transformer, LSTM, ARIMA -- so a model's column exists in
every table. The two per-band/joint grids share one model list outright
(UNIVARIATE_MODELS is MV_MODELS); univariate_series keeps a separate list only
because it runs a different package with different job configs, not because it is
a different set of models.
"""

import csv
import os
import re

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WINDOWS = [(192, 96), (192, 336), (336, 96), (336, 336)]

# The grid's canonical seed. Every run currently on disk sits under it, so it has
# to stay first in any multi-seed list: a --seeds sweep then resumes the work
# already done and only trains the seeds it adds. Extra seeds continue the
# consecutive block the job configs already use (seed_list: [69421, 69422, ...]).
SEED = 69421

# Job names must start with "rf_emf_mv": run_grid resolves the config in-process
# and ignores the prefix, but the per-band driver shells out to main.py, which
# picks forecasting.multivariate only on job_name.startswith("rf_emf_mv").
#
# PatchTST was dropped from the sweep on 2026-07-24 when LSTM and ARIMA were
# added, and put back on 2026-08-03 so all three grids carry the same seven
# models. Its 12 pre-drop cells are NOT on disk any more -- commit e84faec4
# deleted logs/multivariate/PatchTST/ along with the other three windows of every
# model -- so these cells are unrun, not resumable. `git checkout e84faec4^ --
# logs/multivariate` restores the old tree if the pre-drop numbers are wanted.
MV_MODELS = [
    ("DLinear", "DLinear", "rf_emf_mv/dlinear"),
    ("PatchTST", "PatchTST", "rf_emf_mv/patchtst"),
    ("TSMixer", "TSMixer", "rf_emf_mv/tsmixer"),
    ("Transformer", "RF_EMF_Transformer", "rf_emf_mv/transformer"),
    ("EMForecaster", "EMForecaster", "rf_emf_mv/emforecaster"),
    # models/recurrent.py::RecurrentModel with backbone_id "LSTM" -- NOT
    # models/rf_emf_baselines.py::LSTM (model_id RF_EMF_LSTM), which builds
    # nn.LSTM(seq_len, d_model) with batch_first left False and so runs its
    # recurrence over the DataLoader batch axis. See the job config header.
    ("LSTM", "LSTM", "rf_emf_mv/lstm"),
    # Classical fit-once baseline: one statsmodels ARIMA per band, no gradient
    # loop (exp.no_train). See forecasting/multivariate/models/arima.py.
    ("ARIMA", "ARIMA", "rf_emf_mv/arima"),
]
MV_DATASETS = ["ED2022", "UH2022", "EMF_2023_university_5_with_PTV"]

# The per-band grid runs exactly the joint grid's models, one band at a time, off
# the very same rf_emf_mv/* job configs. Aliased rather than copied so the two can
# never drift: a model added to one is added to both, and the cells stay
# comparable one-to-one.
UNIVARIATE_MODELS = MV_MODELS

# The CCM cluster grid (forecasting/multivariate/models/ccm.py). CCM assigns
# channels to k clusters with a learned, soft, sinkhorn-normalized assignment and
# mixes k parallel per-cluster layers by it, rather than partitioning channels up
# front. The clustering is threaded through the backbone, not bolted onto the
# output, so a model is only in this grid if CCM implements it.
#
# That is exactly these four: DLinear, PatchTST, TSMixer and TimesNet are the
# wrappers in TimeSeriesCCM-main/models/. EMForecaster, Transformer, LSTM and
# ARIMA have no upstream counterpart -- clustering them would mean an adaptation
# of our own rather than the paper's method, so they are out. CCMForecaster
# raises on any model_id outside this set instead of silently building something
# else.
#
# Two consequences worth knowing when reading the results:
#   - This list is NOT a subset of MV_MODELS. TimesNet has a job config
#     (rf_emf_mv/timesnet) and runs fine, but it is not in the multivariate grid,
#     so its cluster cells have no same-window multivariate baseline to sit
#     beside. Add TimesNet to MV_MODELS if that comparison is wanted.
#   - The set changed on 2026-08-06, when CCM replaced the KMeans-partition head.
#     PatchTST and TSMixer were excluded under the old head and are back;
#     EMForecaster and LSTM were included and are gone.
CLUSTER_MODELS = [
    ("DLinear", "DLinear", "rf_emf_mv/dlinear"),
    ("PatchTST", "PatchTST", "rf_emf_mv/patchtst"),
    ("TSMixer", "TSMixer", "rf_emf_mv/tsmixer"),
    ("TimesNet", "TimesNet", "rf_emf_mv/timesnet"),
]

# The single-series counterpart list. Job names here are "rf_emf/*", NOT
# "rf_emf_mv/*": this grid is the only one that runs the forecasting.univariate
# package (grid.py::SETUP_PACKAGE), which resolves its configs under
# forecasting/univariate/jobs/exp/ and uses that package's Experiment. There is no
# per-model package selection, so an rf_emf_mv/* job name here would not resolve.
#
# LSTM and ARIMA joined on 2026-08-03, matching the seven models the other two
# grids carry. Both needed a backport into the univariate package first (the
# "LSTM" model_id alias, models/arima.py, the exp.no_train path); see the headers
# of forecasting/univariate/jobs/exp/rf_emf/{lstm,arima}/args.yaml.
SERIES_MODELS = [
    ("DLinear", "DLinear", "rf_emf/dlinear"),
    ("PatchTST", "PatchTST", "rf_emf/patchtst"),
    ("TSMixer", "TSMixer", "rf_emf/tsmixer"),
    ("Transformer", "RF_EMF_Transformer", "rf_emf/baselines/transformer"),
    ("EMForecaster", "EMForecaster", "rf_emf/emforecaster"),
    # RecurrentModel(backbone_id="LSTM") under the "LSTM" alias -- the same model
    # the other two grids run, NOT the rf_emf/baselines/lstm config beside it
    # (model_id RF_EMF_LSTM), which is the frozen batch-axis baseline that
    # produced the published "LSTM (1997)" column. Do not merge the two columns.
    ("LSTM", "LSTM", "rf_emf/lstm"),
    ("ARIMA", "ARIMA", "rf_emf/arima"),
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
        "models": UNIVARIATE_MODELS, "datasets": MV_DATASETS, "windows": WINDOWS,
        "alpha": 0.1, "per_channel": True,
    },
    "univariate_series": {
        "models": SERIES_MODELS, "datasets": SERIES_DATASETS, "windows": WINDOWS,
        "alpha": 0.1,
    },
    # CCM channel clustering. Same datasets/windows as "multivariate", plus a
    # k_values sweep dimension. k=1 (every channel in one cluster) and
    # k=num_channels (one cluster per channel) are deliberately outside the
    # sweep: they are the degenerate ends of the soft assignment rather than
    # interesting cluster counts, and --k reaches them for a one-off sanity
    # check without editing this list.
    "cluster": {
        "models": CLUSTER_MODELS, "datasets": MV_DATASETS, "windows": WINDOWS,
        "alpha": 0.1, "k_values": [2, 3, 5, 9],
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


def select_seeds(spec):
    """The seeds every cell of a grid is trained on, for a --seeds value.

    `spec` None -> just [SEED], i.e. exactly what the grids did before --seeds
    existed. An integer count N -> the first N seeds counting up from SEED, so
    N=3 is [69421, 69422, 69423]. A comma-separated list of integers is taken
    literally instead, for re-running one specific seed.

    Raises ValueError on a non-positive count or a malformed token, so a typo is
    a clean error rather than a sweep that silently trains nothing.
    """
    if spec is None or spec == "":
        return [SEED]
    text = str(spec).strip()
    if "," in text:
        seeds = []
        for token in text.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                seeds.append(int(token))
            except ValueError:
                raise ValueError(
                    f"bad --seeds value {token!r}; use a count like '3' or an "
                    f"explicit list like '69421,69423'"
                )
        if not seeds:
            raise ValueError("empty --seeds value")
        # Duplicates would train the same run twice; the second pass would skip
        # on the resume check, but the plan and the counts would still lie.
        return list(dict.fromkeys(seeds))
    try:
        count = int(text)
    except ValueError:
        raise ValueError(
            f"bad --seeds value {text!r}; use a count like '3' or an explicit "
            f"list like '69421,69423'"
        )
    if count < 1:
        raise ValueError(f"--seeds must be at least 1, got {count}")
    return [SEED + i for i in range(count)]


def select_models(spec, available):
    """The models from `available` selected by `spec`, in grid order.

    `spec` None/empty -> all of them. Otherwise a comma-separated list of either
    display labels or on-disk model_ids, matched case-insensitively, so
    '--model dlinear', '--model DLinear' and '--model RF_EMF_Transformer' all
    work. Matching both names matters because the two differ for the Transformer
    (label "Transformer", model_id "RF_EMF_Transformer"), and the model_id is
    what appears in the logs/ path the user is usually reading from.

    Raises ValueError on a name not in this grid, so a typo (or a model that
    belongs to a different grid) is a clean error rather than a silently empty
    sweep.
    """
    if not spec:
        return list(available)
    requested = [t.strip().lower() for t in str(spec).split(",") if t.strip()]
    if not requested:
        raise ValueError("empty --model value")
    selected, missing = [], []
    for name in dict.fromkeys(requested):
        hit = [m for m in available if name in (m[0].lower(), m[1].lower())]
        if hit:
            selected.extend(hit)
        else:
            missing.append(name)
    if missing:
        have = ", ".join(sorted({m[1] for m in available}))
        raise ValueError(
            f"model(s) {', '.join(missing)} not in this grid (has: {have})"
        )
    # Grid order, deduped: two spellings of one model must not plan it twice.
    return [m for m in available if m in set(selected)]


def select_datasets(spec, available):
    """The datasets from `available` selected by `spec`, in grid order.

    `spec` None/empty -> all of them. Otherwise a comma-separated list matched
    case-insensitively against both the dataset name and its logs/ folder slug,
    so '--dataset ED2022', '--dataset ed2022' and '--dataset emf2023' (the slug
    for EMF_2023_university_5_with_PTV) all work. The slug form is accepted
    because it is what the logs/ tree shows, and typing the full name of the
    emf2023 dataset is otherwise error-prone.
    """
    if not spec:
        return list(available)
    requested = [t.strip().lower() for t in str(spec).split(",") if t.strip()]
    if not requested:
        raise ValueError("empty --dataset value")
    from forecasting.run_path import dataset_slug

    selected, missing = [], []
    for name in dict.fromkeys(requested):
        hit = [d for d in available
               if name in (d.lower(), dataset_slug(d).lower())]
        if hit:
            selected.extend(hit)
        else:
            missing.append(name)
    if missing:
        have = ", ".join(f"{d} ({dataset_slug(d)})" for d in available)
        raise ValueError(
            f"dataset(s) {', '.join(missing)} not in this grid (has: {have})"
        )
    return [d for d in available if d in set(selected)]


def select_k_values(spec, available):
    """The cluster counts a cluster grid sweeps, for a --k value.

    `spec` None/empty -> `available` unchanged (None for a grid with no k
    dimension, which is what tells run_grid not to add k to its cells at all).
    Otherwise a single count ('5') or a comma-separated list ('1,2') selects
    exactly those, deduped and run in ascending order.

    Unlike select_windows, a requested k need NOT be one of the grid's own
    k_values: the endpoints k=1 and k=num_channels are deliberately outside the
    sweep (see GRIDS["cluster"]) precisely so they can be run as one-offs, and
    '--k 1' has to reach them without editing this file. run_grid logs any such
    off-grid value so it is visible in the run's own output.

    Raises ValueError on a non-positive or malformed k, and on --k against a grid
    that has no k dimension, so a typo is a clean error rather than a sweep that
    silently trains the wrong thing.
    """
    if spec is None or spec == "":
        return list(available) if available else None
    if not available:
        raise ValueError(
            "--k only applies to a grid with a k dimension (currently just "
            "'cluster'); this grid has none"
        )
    ks = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            k = int(token)
        except ValueError:
            raise ValueError(
                f"bad --k value {token!r}; use a count like '5' or a list "
                f"like '1,2'"
            )
        if k < 1:
            raise ValueError(f"--k must be at least 1, got {k}")
        ks.append(k)
    if not ks:
        raise ValueError("empty --k value")
    # Duplicates would plan the same cell twice; the second pass would skip on the
    # resume check, but the plan and the [n/total] counts would still lie.
    return sorted(dict.fromkeys(ks))


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
