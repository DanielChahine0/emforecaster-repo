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
- "cluster"           The multivariate sweep with CCM's learned soft channel
                      clustering (Channel Clustering Module, NeurIPS 2024)
                      threaded through this repo's own models: the multivariate
                      cell's exact network with a linear map it applies to each
                      channel on its own swapped for k cluster-wise copies. Adds
                      a k dimension (k_values) to every cell. See CLUSTER_MODELS
                      below and forecasting/multivariate/models/ccm.py.

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

# The CCM cluster grids (forecasting/multivariate/models/ccm.py), EMF and VoIP.
# CCM assigns channels to k clusters with a learned, soft, sinkhorn-normalized
# assignment and mixes k parallel copies of a layer by it, rather than
# partitioning channels up front. The clustering modules are upstream's own
# code, imported verbatim from the vendored TimeSeriesCCM-main/ tree
# (utils/ccm_upstream.py), threaded through THIS REPO'S OWN models: a cluster
# cell is the multivariate cell's exact network, same job config and init, with
# one linear map it applies to each channel on its own swapped for upstream's
# Cluster_wise_linear. Which map, per model:
#
#   DLinear       the seasonal and trend linears (the whole model)
#   PatchTST      the flatten head; the attention encoder is untouched
#   TSMixer       the temporal MLP linears in every ResBlock, and the head
#   Transformer   the fc2 read-out to the horizon; the encoder and convs are
#                 untouched
#   EMForecaster  the Linear read-out (base_model) from the flattened patch
#                 embedding; the patch embedding is untouched
#   LSTM          the linear head on the averaged LSTM state; the patch
#                 projection and the LSTM are untouched
#
# The first three are where upstream's own wrappers put Cluster_wise_linear.
# Upstream has no version of the last three; they follow its PatchTST
# precedent of clustering the one read-out every channel shares.
#
# These are MV_MODELS' six gradient-trained models, so every cluster cell has
# same-window multivariate and per-channel baselines beside it, and both
# cluster grids carry all six. That includes the Transformer on VoIP, whose
# multivariate cells never beat the lookback mean there: every model is run,
# and one that does not train is reported as such rather than dropped.
#
# ARIMA, and Persistence on VoIP, are out, and get_model raises on them rather
# than build something else under their name: neither has a gradient path for
# the assignment to be learned through, and both forecast each band on its
# own, which is already k = num_channels.
#
# History, for reading old results: until 2026-08-25 this grid ran the paper's
# whole released networks (DLinearC, PatchTSTC, ...) -- a different
# implementation from the multivariate cells (different init, RevIN,
# decomposition kernel). Runs made by those wrappers are NOT comparable to runs
# of the native implementation that replaced them and must not share a logs
# tree with them; they sit under logs/_archive/, and logs/cluster/ holds only
# runs of the native (post-2026-08-25) implementation. TimesNet was
# in that version of the grid (upstream ships a wrapper for it); our TimesNet
# embeds all channels jointly from the first layer, has nothing per-channel to
# route, and never had an MV baseline, so it left with the wrappers. Before
# 2026-08-06 the grid was a static KMeans-partition head; those config fields
# are gone too.
CLUSTER_MODELS = [
    ("DLinear", "DLinear", "rf_emf_mv/dlinear"),
    ("PatchTST", "PatchTST", "rf_emf_mv/patchtst"),
    ("TSMixer", "TSMixer", "rf_emf_mv/tsmixer"),
    ("Transformer", "RF_EMF_Transformer", "rf_emf_mv/transformer"),
    ("EMForecaster", "EMForecaster", "rf_emf_mv/emforecaster"),
    # The LSTM-backboned RecurrentModel, as in MV_MODELS; its cluster build
    # refuses any other backbone.
    ("LSTM", "LSTM", "rf_emf_mv/lstm"),
]

# The joint grids' models the cluster grids leave out, as above. A cluster grid
# asked for one by name raises get_model's reason (select_models), not only
# "not in this grid", which reads like a typo.
CLUSTER_REFUSED = ("ARIMA", "Persistence")

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

# ---------------------------------------------------------------------------
# VoIP call-quality grids (data/ML_DATASETS -> data/multivariate/processed/
# VOIP_{MOBILE,FIXED}.csv, built by data/ML_DATASETS/build_voip.py).
#
# Six heterogeneous QoS/QoE channels per dataset (MOS, BW, RTT, JIT, DJB, SNR)
# against the EMF files' 27-29 near-homogeneous frequency bands, which is what
# makes them worth running through the same three paradigms: a multivariate
# model can exploit the QoS -> MOS coupling that a per-channel one cannot see,
# and a cluster count of 2-3 has an obvious hypothesis to find.
#
# The seven EMF models plus a persistence baseline. Persistence is not optional
# decoration here: four of the six channels repeat their previous value in
# 80-99% of steps, so without it a low MSE cannot be distinguished from "carried
# the last reading forward". It is deliberately absent from the EMF grids, whose
# published tables it would change.
VOIP_MODELS = MV_MODELS + [
    ("Persistence", "Persistence", "rf_emf_mv/persistence"),
]
VOIP_DATASETS = ["VOIP_MOBILE", "VOIP_FIXED"]

# The traces are 161-820 rows, so the EMF benchmark window (336 + 96) does not
# fit in any of them. L48/O12 keeps the EMF grid's 4:1 lookback-to-horizon ratio
# and is the benchmark; L24/O6 is the secondary window. L96/O24 is deliberately
# absent: only four of the sixteen traces admit a test window at that size.
VOIP_WINDOWS = [(48, 12), (24, 6)]

# Applied to every cell of the VoIP grids, on top of the job config, by
# forecasting/grid.py and forecasting/perchannel.py.
#
# The split is the config default 0.6/0.2/0.2 rather than the EMF jobs'
# 0.7/0.1/0.2: a 10% validation split of a 400-row trace leaves too few windows
# to calibrate a conformal interval on, and validation is what both early
# stopping and split conformal read.
#
# Batch 64 rather than 512 because a pooled VoIP train split is ~1,900 windows:
# at 512 that is four optimizer steps per epoch and the OneCycle schedule never
# develops. Applied uniformly to every model and paradigm, so it changes what
# the numbers mean but not what they can be compared against.
VOIP_OVERRIDES = {
    "sl.batch_size": 64,
    "data.train_split": 0.6,
    "data.val_split": 0.2,
}

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
    # The VoIP counterparts of the three grids above, sharing their drivers,
    # their log trees and their conformal settings; only the datasets, windows
    # and the VOIP_OVERRIDES differ. Runs land in the same logs/<setup>/ trees
    # under their own dataset slugs (voip_mobile / voip_fixed), so nothing has
    # to learn a new layout and the EMF and VoIP cells stay one `dataset`
    # column apart.
    "voip": {
        "models": VOIP_MODELS, "datasets": VOIP_DATASETS, "windows": VOIP_WINDOWS,
        "alpha": 0.1, "overrides": VOIP_OVERRIDES,
    },
    "voip_perchannel": {
        "models": VOIP_MODELS, "datasets": VOIP_DATASETS, "windows": VOIP_WINDOWS,
        "alpha": 0.1, "per_channel": True, "overrides": VOIP_OVERRIDES,
    },
    # k runs from the controlled baseline, k = 1 (see the EMF cluster grid
    # below), to num_channels: at k = 6 there is one cluster-wise copy per
    # signal, though the soft assignment need not give each signal its own,
    # and the rest of the network stays shared across the six.
    "voip_cluster": {
        "models": CLUSTER_MODELS, "datasets": VOIP_DATASETS,
        "windows": VOIP_WINDOWS, "alpha": 0.1, "k_values": [1, 2, 3, 6],
        "overrides": VOIP_OVERRIDES, "refused": CLUSTER_REFUSED,
    },
    # CCM channel clustering: the multivariate grid's datasets and windows, plus
    # a k dimension. k = 1 is the controlled baseline and part of the sweep: a
    # single cluster holding every channel reduces each cluster-wise layer to
    # the multivariate cell's one shared map, so k > 1 against k = 1 isolates
    # what the clustering itself does. The other end, k = num_channels (27, or
    # 29 on emf2023: one copy per band), is not swept; --k reaches it, or any
    # other count, as a one-off without editing this list. Like the
    # multivariate grid's, a bare sweep covers all four WINDOWS; the L336_O96
    # benchmark alone is --window 336x96.
    "cluster": {
        "models": CLUSTER_MODELS, "datasets": MV_DATASETS, "windows": WINDOWS,
        "alpha": 0.1, "k_values": [1, 2, 3, 5, 9], "refused": CLUSTER_REFUSED,
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

    `spec` None/empty -> all of `available` unchanged. Duplicates collapse, and
    requested windows that are in the grid keep the grid's own ordering so runs
    and resume paths stay stable regardless of how the flag was typed.

    Like select_k_values, a requested window need NOT be one of the grid's own:
    a custom lookback ('--window 720x96') runs as a one-off without editing this
    file. Off-grid windows sort ascending after the grid ones, and run_grid logs
    them so a one-off cell is never mistaken for part of the canonical sweep.
    parse_window_spec still rejects malformed tokens, so a typo that doesn't
    parse as LxO fails loudly rather than training the wrong cell.
    """
    if not spec:
        return list(available)
    requested = set(parse_window_spec(spec))
    on_grid = [w for w in available if w in requested]
    off_grid = sorted(requested - set(available))
    return on_grid + off_grid


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


def select_models(spec, available, refused=()):
    """The models from `available` selected by `spec`, in grid order.

    `spec` None/empty -> all of them. Otherwise a comma-separated list of either
    display labels or on-disk model_ids, matched case-insensitively, so
    '--model dlinear', '--model DLinear' and '--model RF_EMF_Transformer' all
    work. Matching both names matters because the two differ for the Transformer
    (label "Transformer", model_id "RF_EMF_Transformer"), and the model_id is
    what appears in the logs/ path the user is usually reading from.

    Raises ValueError on a name not in this grid, so a typo (or a model that
    belongs to a different grid) is a clean error rather than a silently empty
    sweep. `refused` is a cluster grid's CLUSTER_REFUSED: a model_id in it gets
    get_model's reason for leaving the model out instead.
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
        why = [m for m in refused if m.lower() in missing]
        if why:
            # Imported here: ccm pulls in torch, which grid startup avoids.
            from forecasting.multivariate.models.ccm import no_adaptation_error
            raise no_adaptation_error(why[0])
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

    Like select_windows, a requested k need NOT be one of the grid's own
    k_values: k=num_channels on the EMF grid (one cluster per band) is left out
    of the sweep, and '--k 27' has to reach it, or any other count, as a
    one-off without editing this file. run_grid logs any such off-grid value so
    it is visible in the run's own output.

    Raises ValueError on a non-positive or malformed k, and on --k against a grid
    that has no k dimension, so a typo is a clean error rather than a sweep that
    silently trains the wrong thing.
    """
    if spec is None or spec == "":
        return list(available) if available else None
    if not available:
        raise ValueError(
            "--k only applies to a grid with a k dimension ('cluster' or "
            "'voip_cluster'); this grid has none"
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
    the bands, which would leak the target), the broadband PTV column, and the
    pooled VoIP files' Trace label. Kept in step by hand rather than imported --
    pulling the loader in here would drag torch into grid startup.
    """
    return not (column == "Timestamp" or column.startswith("EMF_")
                or column == "PTV" or column == "Trace")


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
