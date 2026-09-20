"""The pooled VoIP datasets: per-trace splits and the cross-trace window mask.

The property everything else rests on is that no training or evaluation window
may contain rows from two different calls. It is enforced by one flagged row per
join (see pooled_trace_splits), which is easy to get subtly wrong -- an off-by-one
would leak a discontinuity into the middle of a window and quietly corrupt every
cell -- so it is tested directly, on the real trace lengths as well as on
synthetic ones.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from forecasting.grids import dataset_num_channels
from forecasting.multivariate.utils.dataloading import (
    _trace_split_bounds,
    create_windows,
    load_forecasting,
    pooled_trace_splits,
)
from forecasting.multivariate.utils.datasets import ForecastingDataset

VOIP = ["VOIP_MOBILE", "VOIP_FIXED"]
CHANNELS = ["MOS", "BW", "RTT", "JIT", "DJB", "SNR"]


def _labelled_frame(lengths):
    """A frame whose every channel value is that row's trace ordinal."""
    rows = []
    for i, n in enumerate(lengths):
        block = pd.DataFrame({c: np.full(n, float(i)) for c in CHANNELS})
        block.insert(0, "Trace", f"t{i}")
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def _windows_are_single_trace(data, syn, seq_len, pred_len):
    """True if every window the dataset emits carries one trace ordinal."""
    ds = ForecastingDataset(data, seq_len, pred_len, synthetic=syn)
    assert len(ds) > 0
    for i in range(len(ds)):
        x, y = ds[i]
        if len(torch.unique(torch.cat([x.flatten(), y.flatten()]))) != 1:
            return False
    return True


def test_trace_split_bounds_mirror_the_contiguous_split():
    # Same arithmetic the single-series path uses, including the seq_len
    # back-extension on val and test.
    (tr, va, te) = _trace_split_bounds(100, seq_len=10, train_split=0.6, val_split=0.2)
    assert tr == (0, 60)
    assert va == (50, 80)
    assert te == (70, 100)


def test_trace_split_bounds_clamp_short_traces():
    # A trace shorter than one lookback must not produce a negative start.
    (tr, va, te) = _trace_split_bounds(20, seq_len=48, train_split=0.6, val_split=0.2)
    assert tr[0] == 0 and va[0] == 0 and te[0] == 0


def test_every_trace_reaches_every_split():
    df = _labelled_frame([200, 300, 250])
    splits = pooled_trace_splits(df, CHANNELS, seq_len=24, train_split=0.6, val_split=0.2)
    for data, _ in splits:
        assert set(np.unique(data)) == {0.0, 1.0, 2.0}


def test_one_join_flag_per_boundary_and_none_at_the_end():
    df = _labelled_frame([200, 300, 250])
    for data, syn in pooled_trace_splits(df, CHANNELS, seq_len=24,
                                         train_split=0.6, val_split=0.2):
        # Three traces -> two joins, and the final row joins nothing.
        assert syn.sum() == 2
        assert not syn[-1]
        # A flag sits on the last row of a trace, so the row after it (if any)
        # belongs to the next trace.
        for i in np.flatnonzero(syn):
            assert data[i, 0] != data[i + 1, 0]


def test_no_window_spans_two_traces_synthetic():
    df = _labelled_frame([200, 300, 250])
    for data, syn in pooled_trace_splits(df, CHANNELS, seq_len=24,
                                         train_split=0.6, val_split=0.2):
        assert _windows_are_single_trace(data.T, syn, 24, 6)


@pytest.mark.parametrize("dataset", VOIP)
def test_no_window_spans_two_traces_on_the_real_lengths(dataset):
    # The real per-codec row counts, with the channels replaced by the trace
    # ordinal so a spanning window is visible as two distinct values.
    import os

    from forecasting.multivariate.utils import dataloading

    root = os.path.dirname(os.path.dirname(os.path.abspath(dataloading.__file__)))
    path = os.path.join(os.path.dirname(os.path.dirname(root)),
                        "data", "multivariate", "processed", f"{dataset}.csv")
    real = pd.read_csv(path)
    lengths = real.groupby("Trace", sort=False).size().tolist()
    df = _labelled_frame(lengths)
    for data, syn in pooled_trace_splits(df, CHANNELS, seq_len=48,
                                         train_split=0.6, val_split=0.2):
        assert _windows_are_single_trace(data.T, syn, 48, 12)


@pytest.mark.parametrize("dataset", VOIP)
def test_loader_shape_scaling_and_channel_count(dataset):
    assert dataset_num_channels(dataset) == 6
    (tr, tr_s), (va, va_s), (te, te_s) = load_forecasting(
        dataset_name=dataset, seq_len=48, pred_len=12,
        train_split=0.6, val_split=0.2, return_synthetic=True)
    for data, syn in ((tr, tr_s), (va, va_s), (te, te_s)):
        assert data.shape[0] == 6                    # (channels, time)
        assert syn is not None and len(syn) == data.shape[1]
        assert syn.sum() == 7                        # eight traces, seven joins
    # The scaler is fit on train only: train is standardized, val/test are not
    # forced to zero mean.
    assert np.abs(tr.mean(axis=1)).max() < 1e-10
    assert np.allclose(tr.std(axis=1), 1.0)
    assert np.abs(va.mean(axis=1)).max() > 1e-6


@pytest.mark.parametrize("dataset", VOIP)
def test_joint_and_per_channel_paths_agree_on_window_count(dataset):
    # A multivariate cell and its per-channel twin must score the same windows.
    splits = load_forecasting(
        dataset_name=dataset, seq_len=48, pred_len=12,
        train_split=0.6, val_split=0.2, return_synthetic=True)
    for data, syn in splits:
        joint = len(ForecastingDataset(data, 48, 12, synthetic=syn))
        flags = create_windows(torch.from_numpy(syn.astype(np.float32)), 48 + 12, 1)
        per_channel = int((flags.amax(dim=1) == 0).sum())
        assert joint == per_channel


def test_processed_csvs_match_the_raw_traces():
    # Guards against a stale rebuild: the committed CSVs must still be what
    # build_voip.py produces from data/ML_DATASETS/.
    import importlib.util
    import os

    from forecasting.multivariate.utils import dataloading

    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(dataloading.__file__)))))
    spec = importlib.util.spec_from_file_location(
        "build_voip", os.path.join(repo, "data", "ML_DATASETS", "build_voip.py"))
    build_voip = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_voip)

    for access, (_, name) in build_voip.ACCESS.items():
        rebuilt = build_voip.build(access, check=True)
        on_disk = pd.read_csv(os.path.join(
            repo, "data", "multivariate", "processed", f"{name}.csv"))
        pd.testing.assert_frame_equal(rebuilt, on_disk, check_dtype=False)


# --------------------------------------------------------------------------
# The VoIP grids: the same three paradigms as the EMF sweep, one corpus apart.
# --------------------------------------------------------------------------

import torch.nn as nn  # noqa: E402

from forecasting import grid, grids  # noqa: E402
from forecasting.multivariate.models.persistence import Persistence  # noqa: E402

VOIP_SETUPS = ["voip", "voip_perchannel", "voip_cluster"]


def test_persistence_repeats_the_last_value():
    model = Persistence(pred_len=4)
    x = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
    out = model(x)
    assert out.shape == (2, 3, 4)
    assert torch.equal(out, x[..., -1:].expand(2, 3, 4))
    # Parameter-free, so it logs num_params 0 and needs no optimizer.
    assert sum(p.numel() for p in model.parameters()) == 0
    # The per-band grid hands it a singleton channel axis.
    assert model(torch.zeros(2, 1, 5)).shape == (2, 1, 4)


def test_persistence_output_is_writable():
    # expand returns a view; a non-contiguous forecast breaks the conformal
    # stack, which writes into its own copies of pred/target.
    out = Persistence(pred_len=3)(torch.zeros(1, 2, 4))
    assert out.is_contiguous()


@pytest.mark.parametrize("setup", VOIP_SETUPS)
def test_voip_grid_model_ids_match_job_configs(setup):
    load_config, _ = grid.setup_modules(setup)
    for label, model_id, job in grids.GRIDS[setup]["models"]:
        args = load_config(grid.args_path(setup, job))
        assert args.exp.model_id == model_id, f"{setup} {job}"


def test_voip_grid_shapes():
    # 8 models x 2 datasets x 2 windows; the cluster grid is the three
    # clusterable backbones x 2 x 2 x 2 k.
    voip = grids.GRIDS["voip"]
    assert len(voip["models"]) == 8
    assert len(voip["models"]) * len(voip["datasets"]) * len(voip["windows"]) == 32
    cl = grids.GRIDS["voip_cluster"]
    assert [m[1] for m in cl["models"]] == ["DLinear", "PatchTST", "TSMixer"]
    assert cl["k_values"] == [2, 3]
    # The degenerate ends stay outside the sweep, reachable with --k.
    assert grids.select_k_values("1,2,3,6", cl["k_values"]) == [1, 2, 3, 6]


def test_voip_perchannel_mirrors_the_joint_grid():
    # Same reason as the EMF pair: a per-band cell is only interpretable beside
    # the joint cell it matches.
    joint, per = grids.GRIDS["voip"], grids.GRIDS["voip_perchannel"]
    assert joint["models"] == per["models"]
    assert joint["datasets"] == per["datasets"]
    assert joint["windows"] == per["windows"]
    assert per["per_channel"] is True and not joint.get("per_channel")


def test_voip_grids_carry_persistence_but_the_emf_grids_do_not():
    for setup in ("voip", "voip_perchannel"):
        assert "Persistence" in {m[1] for m in grids.GRIDS[setup]["models"]}
    for setup in ("multivariate", "univariate", "univariate_series", "cluster"):
        assert "Persistence" not in {m[1] for m in grids.GRIDS[setup]["models"]}


def test_voip_setups_reuse_the_emf_packages_and_log_trees():
    for setup in VOIP_SETUPS:
        assert grid.SETUP_PACKAGE[setup] == "multivariate"
    assert grid.LOG_SETUP["voip"] == "multivariate"
    assert grid.LOG_SETUP["voip_perchannel"] == "univariate (per channel)"
    assert grid.LOG_SETUP["voip_cluster"] == "cluster"


def test_voip_overrides_reach_both_drivers():
    from forecasting import perchannel

    over = grids.GRIDS["voip"]["overrides"]
    assert over == {"sl.batch_size": 64, "data.train_split": 0.6, "data.val_split": 0.2}

    cell = grid.cell_ablation("DLinear", "VOIP_MOBILE", 48, 12, 0.1, grid.SEED,
                              overrides=over)
    band = perchannel.channel_ablation("VOIP_MOBILE", 48, 12, 0, 0.1, overrides=over)
    for ablation in (cell, band):
        assert ablation["global.sl.batch_size"] == 64
        assert ablation["global.data.train_split"] == 0.6
        assert ablation["global.data.val_split"] == 0.2
    # An EMF cell is untouched by this mechanism.
    plain = grid.cell_ablation("DLinear", "ED2022", 336, 96, 0.1, grid.SEED)
    assert "global.sl.batch_size" not in plain


def test_memory_profile_still_wins_over_a_grid_override():
    # The profile is a hardware mitigation; a grid's declared batch size must
    # not be able to reinstate an OOM.
    ablation = grid.cell_ablation("EMForecaster", "ED2022", 336, 96, 0.1, grid.SEED,
                                  overrides={"sl.batch_size": 512})
    assert ablation["global.sl.batch_size"] == 128
    assert ablation["global.sl.accum_steps"] == 4


def test_apply_overrides_rejects_an_unknown_field():
    import types

    args = types.SimpleNamespace(sl=types.SimpleNamespace(batch_size=512))
    grid.apply_overrides(args, {"sl.batch_size": 64})
    assert args.sl.batch_size == 64
    with pytest.raises(ValueError):
        grid.apply_overrides(args, {"sl.nonexistent": 1})
    with pytest.raises(ValueError):
        grid.apply_overrides(args, {"nogroup.field": 1})


@pytest.mark.parametrize("dataset", VOIP)
def test_univariate_path_matches_the_joint_path_window_for_window(dataset):
    # What forecasting/perchannel.py actually runs: the univariate windowing
    # branch with one band pinned. It must emit exactly the windows the joint
    # path scores, or a per-channel cell is not comparable to the multivariate
    # cell beside it.
    joint = [len(ForecastingDataset(d, 48, 12, synthetic=s))
             for d, s in load_forecasting(
                 dataset_name=dataset, seq_len=48, pred_len=12,
                 train_split=0.6, val_split=0.2, return_synthetic=True)]
    for channel in (0, 5):
        splits = load_forecasting(
            dataset_name=dataset, seq_len=48, pred_len=12, univariate=True,
            single_channel=True, target_channel=channel,
            train_split=0.6, val_split=0.2)
        for (x, y), n in zip(splits, joint):
            assert x.shape == (n, 48)
            assert y.shape == (n, 12)
