import json
import os

import numpy as np
import pytest
import torch

from forecasting import grid, grids
from forecasting.multivariate.config.config import Global
from forecasting.multivariate.layers.clustered_head import ClusteredHead
from forecasting.multivariate.utils.clustering import compute_channel_clusters


# ---------------------------------------------------------------------------
# ClusteredHead
# ---------------------------------------------------------------------------


def test_clustered_head_output_shape():
    head = ClusteredHead(cluster_assignment=[0, 0, 1, 1, 1], k_clusters=2,
                         in_features=8, pred_len=4)
    x = torch.randn(3, 5, 8)
    out = head(x)
    assert out.shape == (3, 5, 4)


def test_clustered_head_preserves_channel_order_and_isolates_clusters():
    # channel 0 -> cluster 1 (alone); channels 1, 2 -> cluster 0 (mixed together).
    torch.manual_seed(0)
    head = ClusteredHead(cluster_assignment=[1, 0, 0], k_clusters=2, in_features=4, pred_len=2)
    x = torch.randn(2, 3, 4)
    out = head(x)
    assert out.shape == (2, 3, 2)

    # Channel 0 is alone in its cluster: perturbing channels 1/2 must not change
    # channel 0's output row (no cross-cluster mixing, and the output row order
    # still matches the input channel order after the permute/inverse-permute).
    x_perturbed = x.clone()
    x_perturbed[:, 1:, :] = torch.randn(2, 2, 4)
    out_perturbed = head(x_perturbed)
    assert torch.allclose(out[:, 0, :], out_perturbed[:, 0, :], atol=1e-6)

    # Conversely, perturbing channel 0 must not change channels 1/2's output.
    x_perturbed2 = x.clone()
    x_perturbed2[:, 0, :] = torch.randn(2, 4)
    out_perturbed2 = head(x_perturbed2)
    assert torch.allclose(out[:, 1:, :], out_perturbed2[:, 1:, :], atol=1e-6)


def test_clustered_head_k1_is_single_group():
    head = ClusteredHead(cluster_assignment=[0, 0, 0], k_clusters=1, in_features=4, pred_len=2)
    assert len(head.linears) == 1
    x = torch.randn(1, 3, 4)
    assert head(x).shape == (1, 3, 2)


def test_clustered_head_k_equals_c_gives_one_head_per_channel():
    head = ClusteredHead(cluster_assignment=[0, 1, 2], k_clusters=3, in_features=4, pred_len=2)
    assert len(head.linears) == 3
    x = torch.randn(1, 3, 4)
    assert head(x).shape == (1, 3, 2)


def test_clustered_head_rejects_empty_cluster_assignment():
    with pytest.raises(ValueError):
        ClusteredHead(cluster_assignment=[], k_clusters=1, in_features=4, pred_len=2)


def test_clustered_head_rejects_out_of_range_cluster_id():
    with pytest.raises(ValueError):
        ClusteredHead(cluster_assignment=[0, 1, 5], k_clusters=2, in_features=4, pred_len=2)


def test_clustered_head_rejects_empty_cluster():
    # k_clusters=3 but no channel assigned to cluster 2.
    with pytest.raises(ValueError):
        ClusteredHead(cluster_assignment=[0, 1, 0, 1], k_clusters=3, in_features=4, pred_len=2)


# ---------------------------------------------------------------------------
# compute_channel_clusters
# ---------------------------------------------------------------------------


def test_compute_channel_clusters_k1_is_all_zeros():
    data = np.random.randn(6, 50)
    assert compute_channel_clusters(data, k=1) == [0] * 6


def test_compute_channel_clusters_k_at_least_num_channels_is_identity():
    data = np.random.randn(4, 50)
    assert compute_channel_clusters(data, k=4) == [0, 1, 2, 3]
    assert compute_channel_clusters(data, k=10) == [0, 1, 2, 3]


def test_compute_channel_clusters_deterministic_for_same_seed():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((10, 200))
    labels_a = compute_channel_clusters(data, k=3, seed=69421)
    labels_b = compute_channel_clusters(data, k=3, seed=69421)
    assert labels_a == labels_b


def test_compute_channel_clusters_returns_valid_labels():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((9, 200))
    labels = compute_channel_clusters(data, k=3, seed=69421)
    assert len(labels) == 9
    assert set(labels) <= {0, 1, 2}


# ---------------------------------------------------------------------------
# Config round trip
# ---------------------------------------------------------------------------


def test_hybrid_config_defaults_are_disabled():
    args = Global()
    assert args.hybrid.enabled is False
    assert args.hybrid.k_clusters == 1
    assert args.hybrid.cluster_assignment == []


def test_hybrid_yaml_block_populates_config():
    # Guards the exact silent-drop foot-gun the repo's own `arima:` comment
    # warns about in config.py: forgetting to register a section on Global
    # means pydantic's extra='ignore' drops it with no error.
    data = {"hybrid": {"enabled": True, "k_clusters": 5, "cluster_seed": 123}}
    args = Global(**data)
    assert args.hybrid.enabled is True
    assert args.hybrid.k_clusters == 5
    assert args.hybrid.cluster_seed == 123


# ---------------------------------------------------------------------------
# grids.py / grid.py wiring
# ---------------------------------------------------------------------------


def test_hybrid_models_excludes_arima_and_tsmixer():
    ids = {model_id for _, model_id, _ in grids.HYBRID_MODELS}
    assert "ARIMA" not in ids
    assert "TSMixer" not in ids
    # PatchTST's constructor takes channel_independent and no cluster_assignment/
    # k_clusters, so a k sweep would train the same model once per k.
    assert "PatchTST" not in ids
    assert "DLinear" in ids
    assert "EMForecaster" in ids
    assert "RF_EMF_Transformer" in ids
    assert "LSTM" in ids


def test_hybrid_grid_job_configs_resolve():
    load_config, _ = grid.setup_modules("hybrid")
    for label, model_id, job in grids.GRIDS["hybrid"]["models"]:
        args = load_config(grid.args_path("hybrid", job))
        assert args.exp.model_id == model_id, f"hybrid {job}: {args.exp.model_id} != {model_id}"


def test_hybrid_grid_uses_multivariate_package_and_log_setup():
    assert grid.SETUP_PACKAGE["hybrid"] == "multivariate"
    assert grid.LOG_SETUP["hybrid"] == "hybrid"


def test_cell_done_round_trips_with_k_tag(tmp_path):
    run_dir = grid.build_run_dir("hybrid", "DLinear", "ED2022", 192, 96,
                                 conf_on=True, logs_dir=str(tmp_path), tag="k3")
    seed_dir = os.path.join(run_dir, str(grid.SEED))
    os.makedirs(seed_dir)
    with open(os.path.join(seed_dir, "log.json"), "w") as f:
        json.dump({"sl_test/interval_width_mean": 0.1}, f)

    assert grid.cell_done("hybrid", "DLinear", "ED2022", 192, 96,
                          logs_dir=str(tmp_path), tag="k3") is True
    # A different k must not be considered done.
    assert grid.cell_done("hybrid", "DLinear", "ED2022", 192, 96,
                          logs_dir=str(tmp_path), tag="k5") is False


def test_run_grid_dry_run_lists_all_hybrid_cells(tmp_path):
    failed = grid.run_grid("hybrid", dry_run=True, logs_dir=str(tmp_path),
                           windows=[(336, 96)])
    assert failed == []


def test_select_k_values_defaults_to_the_grid(tmp_path):
    assert grids.select_k_values(None, grids.GRIDS["hybrid"]["k_values"]) == [2, 3, 5, 9]
    assert grids.select_k_values("", grids.GRIDS["hybrid"]["k_values"]) == [2, 3, 5, 9]
    # A grid with no k dimension stays k-less, which is what tells run_grid to
    # leave k out of its cells entirely.
    assert grids.select_k_values(None, grids.GRIDS["multivariate"].get("k_values")) is None


def test_select_k_values_parses_single_and_list():
    ks = grids.GRIDS["hybrid"]["k_values"]
    assert grids.select_k_values("5", ks) == [5]
    assert grids.select_k_values("3,2", ks) == [2, 3]        # sorted
    assert grids.select_k_values("2,2,3", ks) == [2, 3]      # deduped


def test_select_k_values_allows_off_grid_k():
    # k=1 is deliberately not in the sweep; --k 1 must still reach it, since that
    # is the whole point of being able to run the endpoints as one-offs.
    assert 1 not in grids.GRIDS["hybrid"]["k_values"]
    assert grids.select_k_values("1", grids.GRIDS["hybrid"]["k_values"]) == [1]


def test_select_k_values_rejects_bad_values():
    ks = grids.GRIDS["hybrid"]["k_values"]
    for bad in ("0", "-3", "two", "2,x", ","):
        with pytest.raises(ValueError):
            grids.select_k_values(bad, ks)


def test_select_k_values_rejects_k_on_a_grid_without_one():
    with pytest.raises(ValueError):
        grids.select_k_values("3", grids.GRIDS["multivariate"].get("k_values"))


def test_run_grid_k_values_override_restricts_the_cells(tmp_path, capsys):
    grid.run_grid("hybrid", dry_run=True, logs_dir=str(tmp_path),
                  windows=[(336, 96)], k_values=[1])
    out = capsys.readouterr().out
    assert "k=1" in out
    for k in grids.GRIDS["hybrid"]["k_values"]:
        assert f"k={k}" not in out
    # one cell per model x dataset, and no k dimension left over
    assert out.count("[DRY ]") == len(grids.HYBRID_MODELS) * len(grids.MV_DATASETS)


def test_run_grid_without_k_values_runs_the_whole_sweep(tmp_path, capsys):
    grid.run_grid("hybrid", dry_run=True, logs_dir=str(tmp_path),
                  windows=[(336, 96)], k_values=None)
    out = capsys.readouterr().out
    assert out.count("[DRY ]") == (len(grids.HYBRID_MODELS) * len(grids.MV_DATASETS)
                                   * len(grids.GRIDS["hybrid"]["k_values"]))


def test_non_hybrid_grids_unaffected_by_k_dimension(tmp_path):
    # multivariate/univariate_series have no k_values key, so run_grid's k
    # dimension must be a no-op for them (cell count, tags, everything unchanged).
    assert "k_values" not in grids.GRIDS["multivariate"]
    assert "k_values" not in grids.GRIDS["univariate_series"]
    failed = grid.run_grid("multivariate", dry_run=True, logs_dir=str(tmp_path),
                           windows=[(336, 96)])
    assert failed == []


# ---------------------------------------------------------------------------
# Forward-pass shape tests (toy sizes) for the two models that needed forward()
# reordering: EMForecaster and RecurrentModel.
# ---------------------------------------------------------------------------


def test_emforecaster_hybrid_forward_shape():
    from forecasting.multivariate.models.emforecaster import EMForecaster

    args = Global()
    args.dlinear.moving_avg = 3
    args.dlinear.final_moving_avg = 3
    num_channels = 6
    seq_len = 32
    pred_len = 8
    cluster_assignment = [0, 0, 1, 1, 2, 2]

    model = EMForecaster(
        args=args,
        seq_len=seq_len,
        pred_len=pred_len,
        num_channels=num_channels,
        revin=True,
        revout=True,
        revin_affine=True,
        patch_model_id="DLinear",
        backbone_id="DLinear",
        patch_dim=8,
        patch_stride=4,
        patch_embed_dim=16,
        cluster_assignment=cluster_assignment,
        k_clusters=3,
    )
    x = torch.randn(2, num_channels, seq_len)
    out = model(x)
    assert out.shape == (2, num_channels, pred_len)
    assert not hasattr(model, "base_model")


def test_emforecaster_default_path_unaffected_by_hybrid_kwargs():
    from forecasting.multivariate.models.emforecaster import EMForecaster

    args = Global()
    args.dlinear.moving_avg = 3
    args.dlinear.final_moving_avg = 3
    model = EMForecaster(
        args=args,
        seq_len=32,
        pred_len=8,
        num_channels=1,
        revin=True,
        revout=True,
        revin_affine=True,
        patch_model_id="DLinear",
        backbone_id="DLinear",
        patch_dim=8,
        patch_stride=4,
        patch_embed_dim=16,
    )
    assert hasattr(model, "base_model")
    assert not hasattr(model, "clustered_head")
    x = torch.randn(2, 1, 32)
    assert model(x).shape == (2, 1, 8)


def test_recurrent_model_hybrid_forward_shape():
    from forecasting.multivariate.models.recurrent import RecurrentModel

    num_channels = 5
    seq_len = 40
    pred_len = 6
    cluster_assignment = [0, 0, 1, 1, 1]

    model = RecurrentModel(
        d_model=8,
        num_enc_layers=1,
        pred_len=pred_len,
        backbone_id="LSTM",
        seq_len=seq_len,
        patching=True,
        patch_dim=8,
        patch_stride=4,
        num_channels=num_channels,
        head_type="linear",
        revin=True,
        revout=True,
        last_state=False,
        avg_state=True,
        cluster_assignment=cluster_assignment,
        k_clusters=2,
    )
    x = torch.randn(2, num_channels, seq_len)
    out = model(x)
    assert out.shape == (2, num_channels, pred_len)


def test_recurrent_model_hybrid_rejects_mlp_head():
    from forecasting.multivariate.models.recurrent import RecurrentModel

    with pytest.raises(NotImplementedError):
        RecurrentModel(
            d_model=8,
            num_enc_layers=1,
            pred_len=6,
            backbone_id="LSTM",
            seq_len=40,
            patching=True,
            patch_dim=8,
            patch_stride=4,
            num_channels=5,
            head_type="mlp",
            cluster_assignment=[0, 0, 1, 1, 1],
            k_clusters=2,
        )


def test_dlinear_hybrid_forward_shape():
    from forecasting.multivariate.models.dlinear import DLinear

    num_channels = 4
    seq_len = 20
    pred_len = 5
    model = DLinear(
        seq_len=seq_len,
        pred_len=pred_len,
        num_channels=num_channels,
        cluster_assignment=[0, 0, 1, 1],
        k_clusters=2,
    )
    x = torch.randn(2, num_channels, seq_len)
    out = model(x)
    assert out.shape == (2, num_channels, pred_len)


def test_rf_emf_transformer_hybrid_forward_shape():
    from forecasting.multivariate.models.rf_emf_baselines import Transformer

    num_channels = 4
    seq_len = 20
    pred_len = 5
    model = Transformer(
        seq_len=seq_len,
        pred_len=pred_len,
        d_model=2,
        num_heads=2,
        num_enc_layers=1,
        revin=True,
        cluster_assignment=[0, 0, 1, 1],
        k_clusters=2,
    )
    x = torch.randn(2, num_channels, seq_len)
    out = model(x)
    assert out.shape == (2, num_channels, pred_len)
