"""Tests for the CCM cluster grid.

The cluster grid threads CCM (Channel Clustering Module, NeurIPS 2024) through
this repo's own model implementations: get_model builds the multivariate cell's
exact network with its per-channel temporal linears swapped for upstream's own
Cluster_wise_linear (imported verbatim from the vendored TimeSeriesCCM-main/
tree via utils/ccm_upstream.py) and wraps it in CCMForecaster
(forecasting/multivariate/models/ccm.py), which drives upstream's
Cluster_assigner and owns the similarity loss. These tests cover that
mechanism's contract with the rest of the pipeline (shapes, routing, loss,
checkpointing) and the grid wiring that gets a cell from `--setup cluster --k N`
to logs/cluster/.

They replaced the tests for the whole-network upstream adapter this grid ran
until 2026-08-25 (which had themselves replaced the KMeans-partition head's
tests on 2026-08-06); the upstream model wrappers and their TimesNet cells are
gone, and the shape-bug regression moved into the wrapper's refresh tests.
"""

import json
import os

import pytest
import torch

from forecasting import grid, grids
from forecasting.multivariate.config.config import Global
from forecasting.multivariate.models.ccm import CCM_MODELS, CCMForecaster
from forecasting.multivariate.models.dlinear import DLinear
from forecasting.multivariate.utils.ccm_upstream import (Cluster_assigner,
                                                         Cluster_wise_linear,
                                                         sinkhorn)
from forecasting.multivariate.utils.models import (compute_aux_loss, compute_loss,
                                                   get_model)

# Small enough that the whole file runs in seconds on CPU, large enough that
# PatchTST gets more than one patch out of the window.
NUM_CHANNELS = 6
SEQ_LEN = 32
PRED_LEN = 8
K = 3


def make_args(model_id, k=K, beta=0.3):
    """A Global config for one CCM cell, sized for a fast test."""
    args = Global()
    args.exp.model_id = model_id
    args.exp.task = "forecasting"
    args.data.seq_len = SEQ_LEN
    args.data.pred_len = PRED_LEN
    args.data.num_channels = NUM_CHANNELS
    args.data.dataset = "ED2022"
    args.data.patch_dim = 8
    args.data.patch_stride = 4
    args.sl.d_model = 16
    args.sl.d_ff = 16
    args.sl.num_heads = 2
    args.sl.num_enc_layers = 1
    args.cluster.enabled = True
    args.cluster.k_clusters = k
    args.cluster.beta = beta
    return args


# ---------------------------------------------------------------------------
# Upstream's CCM layers, as this repo drives them
# ---------------------------------------------------------------------------


def test_sinkhorn_rows_are_distributions():
    prob = sinkhorn(torch.randn(NUM_CHANNELS, K))
    torch.testing.assert_close(prob.sum(dim=1), torch.ones(NUM_CHANNELS))
    assert (prob > 0).all()


def test_cluster_wise_linear_blends_by_membership():
    # Upstream's trailing device arg is unused by the released class.
    layer = Cluster_wise_linear(K, NUM_CHANNELS, SEQ_LEN, PRED_LEN, None)
    x = torch.randn(4, NUM_CHANNELS, SEQ_LEN)
    prob = sinkhorn(torch.randn(NUM_CHANNELS, K))
    out = layer(x, prob)
    assert out.shape == (4, NUM_CHANNELS, PRED_LEN)
    expected = sum(
        prob[:, j].view(1, -1, 1) * layer.linears[j](x) for j in range(K)
    )
    torch.testing.assert_close(out, expected)


def test_cluster_wise_linear_k1_is_the_plain_shared_linear():
    # The --k 1 cell is the grid's controlled no-clustering baseline, which
    # only works if the blend genuinely collapses to the single linear.
    layer = Cluster_wise_linear(1, NUM_CHANNELS, SEQ_LEN, PRED_LEN, None)
    x = torch.randn(4, NUM_CHANNELS, SEQ_LEN)
    prob = sinkhorn(torch.randn(NUM_CHANNELS, 1))
    torch.testing.assert_close(prob, torch.ones(NUM_CHANNELS, 1))
    torch.testing.assert_close(layer(x, prob), layer.linears[0](x))


def test_assigner_prob_is_a_normalized_channel_by_cluster_matrix():
    assigner = Cluster_assigner(NUM_CHANNELS, K, SEQ_LEN, 16, device="cpu")
    assigner.eval()
    # Upstream layout: (batch, seq_len, num_channels), cluster_emb passed in.
    prob, _ = assigner(torch.randn(4, SEQ_LEN, NUM_CHANNELS),
                       assigner.cluster_emb)
    assert prob.shape == (NUM_CHANNELS, K)
    # sinkhorn row-normalizes, so each channel's assignment is a distribution
    # over clusters -- soft membership, not a hard partition.
    torch.testing.assert_close(prob.sum(dim=1), torch.ones(NUM_CHANNELS))
    assert (prob > 0).all()


def test_wrapper_refreshes_cluster_emb_on_training_forwards_only():
    """Also the regression test for upstream's refresh shape bug: the assigner
    returns a [batch, k, d_model] cross-attention result, and writing it back
    unpooled kills training batch two in the released code. The wrapper must
    pool it to [k, d_model] and only on training forwards.
    """
    model = get_model(make_args("DLinear"))
    x = torch.randn(4, NUM_CHANNELS, SEQ_LEN)

    model.eval()
    before = model.cluster_emb.clone()
    with torch.no_grad():
        model(x)
    torch.testing.assert_close(model.cluster_emb, before)

    model.train()
    model(x)
    assert not torch.allclose(model.cluster_emb, before)
    assert model.cluster_emb.shape == (K, 16)


# ---------------------------------------------------------------------------
# CCMForecaster on the repo's models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_id", sorted(CCM_MODELS))
def test_ccm_forward_shape(model_id):
    model = get_model(make_args(model_id))
    x = torch.randn(2, NUM_CHANNELS, SEQ_LEN)
    assert model(x).shape == (2, NUM_CHANNELS, PRED_LEN)


@pytest.mark.parametrize("model_id", sorted(CCM_MODELS))
def test_ccm_survives_multiple_training_steps(model_id):
    model = get_model(make_args(model_id))
    x = torch.randn(2, NUM_CHANNELS, SEQ_LEN)
    model.train()
    for _ in range(3):
        model(x)
        assert model.cluster_emb.dim() == 2
        assert model.cluster_emb.shape[0] == K


@pytest.mark.parametrize("model_id", sorted(CCM_MODELS))
def test_ccm_state_dict_is_stable_across_training_steps(model_id):
    """The key set must not depend on how many steps have run.

    Experiment.test() loads supervised.pth into a freshly built model with
    strict=True before the conformal recalibration, so the cluster-embedding
    state has to live in a registered buffer, not appear mid-training.
    """
    model = get_model(make_args(model_id))
    before = set(model.state_dict())
    assert "cluster_emb" in before
    # The assigner's own bare-tensor attribute must never become tracked state:
    # the wrapper's buffer is the one canonical copy.
    assert "assigner.cluster_emb" not in before

    x = torch.randn(2, NUM_CHANNELS, SEQ_LEN)
    model.train()
    model(x)
    model(x)
    model.eval()
    with torch.no_grad():
        model(x)

    assert set(model.state_dict()) == before


def test_ccm_checkpoint_round_trips_into_a_fresh_model():
    model = get_model(make_args("DLinear"))
    x = torch.randn(2, NUM_CHANNELS, SEQ_LEN)
    model.train()
    model(x)  # mutates cluster_emb away from its init

    fresh = get_model(make_args("DLinear"))
    fresh.load_state_dict(model.state_dict())  # strict=True, as Experiment.test does
    torch.testing.assert_close(fresh.cluster_emb, model.cluster_emb)

    model.eval()
    fresh.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x), fresh(x))


def test_ccm_membership_routes_the_cluster_wise_layers():
    # With k>1, the same input under a different membership must produce a
    # different forecast; otherwise the "clustering" is decoration.
    model = get_model(make_args("DLinear"))
    model.eval()
    x = torch.randn(2, NUM_CHANNELS, SEQ_LEN)
    one_hot = torch.eye(K)[torch.tensor([0] * NUM_CHANNELS)]
    other = torch.eye(K)[torch.tensor([1] * NUM_CHANNELS)]
    with torch.no_grad():
        out_a = model.backbone(x, cluster_prob=one_hot)
        out_b = model.backbone(x, cluster_prob=other)
    assert not torch.allclose(out_a, out_b)


def test_ccm_k1_dlinear_matches_the_multivariate_implementation():
    """The whole point of the native rewrite: the cluster cell's network IS the
    multivariate cell's. At k=1 the blend collapses to the single shared linear,
    so with the weights copied over, the forecasts must match exactly (the old
    upstream DLinearC could not do this -- different kernel, different init).
    """
    args = make_args("DLinear", k=1)
    ccm = get_model(args)

    plain_args = make_args("DLinear", k=1)
    plain_args.cluster.enabled = False
    plain = get_model(plain_args)
    assert isinstance(plain, DLinear)

    with torch.no_grad():
        for src, dst in ((plain.Linear_Seasonal, ccm.backbone.Linear_Seasonal),
                         (plain.Linear_Trend, ccm.backbone.Linear_Trend)):
            dst.linears[0].weight.copy_(src.weight)
            dst.linears[0].bias.copy_(src.bias)

    ccm.eval()
    plain.eval()
    x = torch.randn(2, NUM_CHANNELS, SEQ_LEN)
    with torch.no_grad():
        torch.testing.assert_close(ccm(x), plain(x))


def test_get_model_rejects_a_model_without_a_ccm_adaptation():
    # EMForecaster (and TimesNet, since the upstream bridge left) have no
    # per-channel temporal linear to route; building something else under their
    # name would put a column in the results table that is not the method it
    # claims to be.
    for model_id in ("EMForecaster", "TimesNet"):
        with pytest.raises(ValueError, match="no CCM adaptation"):
            get_model(make_args(model_id))


def test_cluster_built_models_require_the_membership_at_forward():
    # A cluster-built backbone silently run without its membership would be an
    # un-clustered model wearing the cluster cell's name.
    model = get_model(make_args("DLinear"))
    with pytest.raises(ValueError, match="needs the"):
        model.backbone(torch.randn(2, NUM_CHANNELS, SEQ_LEN))


def test_channel_similarity_is_a_symmetric_channel_matrix():
    x = torch.randn(4, NUM_CHANNELS, SEQ_LEN)
    sim = CCMForecaster.channel_similarity(x)
    assert sim.shape == (NUM_CHANNELS, NUM_CHANNELS)
    torch.testing.assert_close(sim, sim.t())
    # exp(-5 * 0 / max) == 1 for a channel against itself.
    torch.testing.assert_close(sim.diagonal(), torch.ones(NUM_CHANNELS))
    assert ((sim > 0) & (sim <= 1)).all()


def test_similarity_loss_is_finite_and_needs_a_forward_first():
    model = get_model(make_args("DLinear"))
    x = torch.randn(4, NUM_CHANNELS, SEQ_LEN)

    with pytest.raises(RuntimeError, match="before forward"):
        model.similarity_loss(x)

    model(x)
    loss = model.similarity_loss(x)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_ccm_backward_reaches_the_cluster_assigner():
    model = get_model(make_args("DLinear"))
    x = torch.randn(4, NUM_CHANNELS, SEQ_LEN)
    model.train()
    out = model(x)
    (out.square().mean() + model.similarity_loss(x)).backward()
    # The assigner's projection is the learned part of the assignment; if it got
    # no gradient the clustering would never adapt to the data.
    grad = model.assigner.linear.weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_multivariate_models_are_untouched_when_clustering_is_off():
    # The MV and per-channel grids must keep building the exact baselines: no
    # cluster modules in the tree, and the plain forward signature still works.
    args = make_args("DLinear")
    args.cluster.enabled = False
    model = get_model(args)
    assert not any(isinstance(m, (Cluster_wise_linear, Cluster_assigner))
                   for m in model.modules())
    assert model(torch.randn(2, NUM_CHANNELS, SEQ_LEN)).shape == (
        2, NUM_CHANNELS, PRED_LEN)


# ---------------------------------------------------------------------------
# Pipeline wiring: get_model and compute_loss
# ---------------------------------------------------------------------------


def test_get_model_returns_ccm_when_clustering_is_enabled():
    model = get_model(make_args("DLinear"))
    assert isinstance(model, CCMForecaster)
    assert isinstance(model.backbone, DLinear)


def test_get_model_ignores_cluster_when_disabled():
    args = make_args("DLinear")
    args.cluster.enabled = False
    assert not isinstance(get_model(args), CCMForecaster)


def test_compute_loss_stays_the_forecast_loss_on_a_cluster_run():
    """The reported loss and the selection metric must not include the aux term.

    At 27 channels the similarity term runs 15-250 against an MSE around 0.5, so
    folding it in hands early stopping to the regularizer and makes a cluster
    cell's loss incomparable to every other cell.
    """
    args = make_args("DLinear", beta=0.5)
    model = get_model(args)
    x = torch.randn(4, NUM_CHANNELS, SEQ_LEN)
    y = torch.randn(4, NUM_CHANNELS, PRED_LEN)
    out = model(x)
    criterion = torch.nn.MSELoss()
    loss = compute_loss(out, (x, y), criterion, "DLinear", args, "cpu")
    torch.testing.assert_close(loss, criterion(out, y))


def test_compute_aux_loss_is_the_beta_scaled_similarity_term():
    args = make_args("DLinear", beta=0.5)
    model = get_model(args)
    x = torch.randn(4, NUM_CHANNELS, SEQ_LEN)
    model(x)

    torch.manual_seed(0)
    aux = compute_aux_loss(model, (x, None), args, "cpu")
    torch.manual_seed(0)
    expected = 0.5 * model.similarity_loss(x)
    torch.testing.assert_close(aux, expected)


def test_compute_aux_loss_is_none_when_there_is_no_aux_term():
    args = make_args("DLinear")
    model = get_model(args)
    x = torch.randn(4, NUM_CHANNELS, SEQ_LEN)
    model(x)

    args.cluster.beta = 0.0
    assert compute_aux_loss(model, (x, None), args, "cpu") is None
    args.cluster.beta = 0.3
    args.cluster.enabled = False
    assert compute_aux_loss(model, (x, None), args, "cpu") is None


def test_compute_loss_is_unchanged_for_non_cluster_runs():
    args = make_args("DLinear")
    args.cluster.enabled = False
    out = torch.randn(4, NUM_CHANNELS, PRED_LEN)
    y = torch.randn(4, NUM_CHANNELS, PRED_LEN)
    criterion = torch.nn.MSELoss()
    loss = compute_loss(out, (None, y), criterion, "DLinear", args, "cpu")
    torch.testing.assert_close(loss, criterion(out, y))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_cluster_config_defaults_are_disabled():
    args = Global()
    assert args.cluster.enabled is False
    assert args.cluster.k_clusters == 1
    assert args.cluster.beta == 0.3


def test_cluster_yaml_block_populates_config():
    # Guards the exact silent-drop foot-gun the repo's own `arima:` comment
    # warns about in config.py: forgetting to register a section on Global
    # means pydantic's extra='ignore' drops it with no error.
    args = Global(**{"cluster": {"enabled": True, "k_clusters": 5, "beta": 0.7}})
    assert args.cluster.enabled is True
    assert args.cluster.k_clusters == 5
    assert args.cluster.beta == 0.7


def test_kmeans_cluster_fields_are_gone():
    # Old runs' args.yaml carried these. They are not resumable against CCM, and
    # a stale field silently accepted would imply otherwise.
    args = Global()
    for dead in ("cluster_assignment", "cluster_method", "cluster_seed"):
        assert not hasattr(args.cluster, dead)


# ---------------------------------------------------------------------------
# grids.py / grid.py wiring
# ---------------------------------------------------------------------------


def test_cluster_models_are_exactly_the_natively_adapted_three():
    ids = {model_id for _, model_id, _ in grids.CLUSTER_MODELS}
    assert ids == set(CCM_MODELS)
    assert ids == {"DLinear", "PatchTST", "TSMixer"}


def test_cluster_models_all_have_multivariate_baselines():
    # The grid exists to compare clustering against the multivariate and
    # per-channel paradigms, so every cluster cell needs its baseline cells.
    mv = {model_id for _, model_id, _ in grids.MV_MODELS}
    assert {model_id for _, model_id, _ in grids.CLUSTER_MODELS} <= mv


def test_cluster_grid_job_configs_resolve():
    load_config, _ = grid.setup_modules("cluster")
    for label, model_id, job in grids.GRIDS["cluster"]["models"]:
        args = load_config(grid.args_path("cluster", job))
        assert args.exp.model_id == model_id, f"cluster {job}: {args.exp.model_id} != {model_id}"


def test_cluster_grid_uses_multivariate_package_and_log_setup():
    assert grid.SETUP_PACKAGE["cluster"] == "multivariate"
    assert grid.LOG_SETUP["cluster"] == "cluster"


def test_cell_done_round_trips_with_k_tag(tmp_path):
    run_dir = grid.build_run_dir("cluster", "DLinear", "ED2022", 192, 96,
                                 conf_on=True, logs_dir=str(tmp_path), tag="k3")
    seed_dir = os.path.join(run_dir, str(grid.SEED))
    os.makedirs(seed_dir)
    with open(os.path.join(seed_dir, "log.json"), "w") as f:
        json.dump({"sl_test/interval_width_mean": 0.1}, f)

    assert grid.cell_done("cluster", "DLinear", "ED2022", 192, 96,
                          logs_dir=str(tmp_path), tag="k3") is True
    # A different k must not be considered done.
    assert grid.cell_done("cluster", "DLinear", "ED2022", 192, 96,
                          logs_dir=str(tmp_path), tag="k5") is False


def test_run_grid_dry_run_lists_all_cluster_cells(tmp_path):
    failed = grid.run_grid("cluster", dry_run=True, logs_dir=str(tmp_path),
                           windows=[(336, 96)])
    assert failed == []


def test_select_k_values_defaults_to_the_grid(tmp_path):
    assert grids.select_k_values(None, grids.GRIDS["cluster"]["k_values"]) == [2, 3, 5, 9]
    assert grids.select_k_values("", grids.GRIDS["cluster"]["k_values"]) == [2, 3, 5, 9]
    # A grid with no k dimension stays k-less, which is what tells run_grid to
    # leave k out of its cells entirely.
    assert grids.select_k_values(None, grids.GRIDS["multivariate"].get("k_values")) is None


def test_select_k_values_parses_single_and_list():
    ks = grids.GRIDS["cluster"]["k_values"]
    assert grids.select_k_values("5", ks) == [5]
    assert grids.select_k_values("3,2", ks) == [2, 3]        # sorted
    assert grids.select_k_values("2,2,3", ks) == [2, 3]      # deduped


def test_select_k_values_allows_off_grid_k():
    # k=1 is deliberately not in the sweep; --k 1 must still reach it, since that
    # is the whole point of being able to run the endpoints as one-offs.
    assert 1 not in grids.GRIDS["cluster"]["k_values"]
    assert grids.select_k_values("1", grids.GRIDS["cluster"]["k_values"]) == [1]


def test_select_k_values_rejects_bad_values():
    ks = grids.GRIDS["cluster"]["k_values"]
    for bad in ("0", "-3", "two", "2,x", ","):
        with pytest.raises(ValueError):
            grids.select_k_values(bad, ks)


def test_select_k_values_rejects_k_on_a_grid_without_one():
    with pytest.raises(ValueError):
        grids.select_k_values("3", grids.GRIDS["multivariate"].get("k_values"))


def test_run_grid_k_values_override_restricts_the_cells(tmp_path, capsys):
    grid.run_grid("cluster", dry_run=True, logs_dir=str(tmp_path),
                  windows=[(336, 96)], k_values=[1])
    out = capsys.readouterr().out
    assert "k=1" in out
    for k in grids.GRIDS["cluster"]["k_values"]:
        assert f"k={k}" not in out
    # one cell per model x default dataset, and no k dimension left over
    assert out.count("[DRY ]") == (len(grids.CLUSTER_MODELS)
                                   * len(grids.GRIDS["cluster"]["default_datasets"]))


def test_run_grid_without_k_values_runs_the_whole_sweep(tmp_path, capsys):
    grid.run_grid("cluster", dry_run=True, logs_dir=str(tmp_path),
                  windows=[(336, 96)], k_values=None)
    out = capsys.readouterr().out
    assert out.count("[DRY ]") == (len(grids.CLUSTER_MODELS)
                                   * len(grids.GRIDS["cluster"]["default_datasets"])
                                   * len(grids.GRIDS["cluster"]["k_values"]))


def test_cluster_grid_defaults_to_ed2022_but_keeps_the_rest_reachable(tmp_path, capsys):
    # The clustering study runs on ED2022, so a bare sweep covers only it ...
    grid.run_grid("cluster", dry_run=True, logs_dir=str(tmp_path),
                  windows=[(336, 96)])
    out = capsys.readouterr().out
    assert "ED2022" in out
    assert "UH2022" not in out and "emf2023" not in out.lower()
    # ... while --dataset still reaches the full multivariate dataset list.
    assert grids.select_datasets("uh2022", grids.GRIDS["cluster"]["datasets"]) == ["UH2022"]


def test_select_models_matches_label_or_model_id():
    mv = grids.GRIDS["multivariate"]["models"]
    # The Transformer's label and model_id differ; both spellings must resolve.
    assert grids.select_models("Transformer", mv) == [("Transformer", "RF_EMF_Transformer", "rf_emf_mv/transformer")]
    assert grids.select_models("rf_emf_transformer", mv) == grids.select_models("Transformer", mv)
    assert grids.select_models(None, mv) == list(mv)
    # Grid order, deduped, regardless of how it was typed.
    picked = grids.select_models("patchtst,DLinear,dlinear", mv)
    assert [m[1] for m in picked] == ["DLinear", "PatchTST"]


def test_select_models_rejects_a_model_from_another_grid():
    # EMForecaster is in the multivariate grid but not the cluster grid; asking
    # for it there has to fail rather than sweep nothing.
    with pytest.raises(ValueError, match="not in this grid"):
        grids.select_models("EMForecaster", grids.GRIDS["cluster"]["models"])


def test_select_datasets_matches_name_or_logs_slug():
    ds = grids.MV_DATASETS
    assert grids.select_datasets("ED2022", ds) == ["ED2022"]
    assert grids.select_datasets("ed2022", ds) == ["ED2022"]
    # The slug is what the logs/ tree shows, and the full emf2023 name is long.
    assert grids.select_datasets("emf2023", ds) == ["EMF_2023_university_5_with_PTV"]
    assert grids.select_datasets(None, ds) == list(ds)
    with pytest.raises(ValueError, match="not in this grid"):
        grids.select_datasets("ettm1", ds)


def test_run_grid_model_and_dataset_filters_restrict_the_cells(tmp_path, capsys):
    grid.run_grid("cluster", dry_run=True, logs_dir=str(tmp_path),
                  windows=[(336, 96)], k_values=[1, 2, 4, 6],
                  models=grids.select_models("DLinear", grids.GRIDS["cluster"]["models"]),
                  datasets=grids.select_datasets("ed2022", grids.MV_DATASETS))
    out = capsys.readouterr().out
    assert out.count("[DRY ]") == 4          # one cell per k, nothing else
    assert "PatchTST" not in out and "UH2022" not in out


def test_grid_report_records_what_a_filtered_run_covered(tmp_path):
    # A trial run writes its report to the same path as a full sweep, so the
    # report has to name the subset or it reads as a complete grid.
    #
    # dry_run=False is needed (a dry run writes no report), so the cell is marked
    # complete first to make run_grid skip it. Without that this test would
    # really train a model -- logs_dir only redirects the resume check, never
    # Experiment.run(), which always writes under the default logs/ tree.
    seed_dir = os.path.join(
        grid.build_run_dir("cluster", "DLinear", "ED2022", 336, 96, conf_on=True,
                           logs_dir=str(tmp_path), tag="k1"),
        str(grid.SEED),
    )
    os.makedirs(seed_dir)
    with open(os.path.join(seed_dir, "log.json"), "w") as f:
        json.dump({grid.MARKER: 0.1}, f)

    grid.run_grid("cluster", dry_run=False, logs_dir=str(tmp_path),
                  windows=[(336, 96)], k_values=[1],
                  models=grids.select_models("DLinear", grids.GRIDS["cluster"]["models"]),
                  datasets=grids.select_datasets("ed2022", grids.MV_DATASETS))
    report = json.load(open(os.path.join(str(tmp_path), "grid_report_cluster.json")))
    assert report["skipped"] and not report["passed"]   # nothing was trained
    assert report["models"] == ["DLinear"]
    assert report["datasets"] == ["ED2022"]
    assert report["windows"] == ["L336_O96"]
    assert report["k_values"] == [1]


def test_non_cluster_grids_unaffected_by_k_dimension(tmp_path):
    # multivariate/univariate_series have no k_values key, so run_grid's k
    # dimension must be a no-op for them (cell count, tags, everything unchanged).
    assert "k_values" not in grids.GRIDS["multivariate"]
    assert "k_values" not in grids.GRIDS["univariate_series"]
    failed = grid.run_grid("multivariate", dry_run=True, logs_dir=str(tmp_path),
                           windows=[(336, 96)])
    assert failed == []
