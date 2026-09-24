"""Tests for the CCM cluster grid.

The cluster grid threads CCM (Channel Clustering Module, NeurIPS 2024) through
this repo's own model implementations: get_model builds the multivariate cell's
exact network with its per-channel temporal linears swapped for upstream's own
Cluster_wise_linear (imported verbatim from the vendored TimeSeriesCCM-main/
tree via utils/ccm_upstream.py) and wraps it in CCMForecaster
(forecasting/multivariate/models/ccm.py), which drives upstream's
Cluster_assigner and owns the similarity loss. These tests cover that
mechanism's contract with the rest of the pipeline (shapes, routing, loss,
checkpointing), the fixed-membership evaluation protocol that sits beside
upstream's batch one, and the grid wiring that gets a cell from
`--setup cluster --k N` to logs/cluster/.

They replaced the tests for the whole-network upstream adapter this grid ran
until 2026-08-25 (which had themselves replaced the KMeans-partition head's
tests on 2026-08-06); the upstream model wrappers and their TimesNet cells are
gone, and the shape-bug regression moved into the wrapper's refresh tests.
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from forecasting import grid, grids
from forecasting.multivariate.config.config import Global
from forecasting.multivariate.exp.exp import Experiment
from forecasting.multivariate.models.ccm import CCM_MODELS, CCMForecaster
from forecasting.multivariate.models.dlinear import DLinear
from forecasting.multivariate.models.emforecaster import EMForecaster
from forecasting.multivariate.models.recurrent import RecurrentModel
from forecasting.multivariate.models.rf_emf_baselines import Transformer
from forecasting.multivariate.utils import models
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
    if model_id == "EMForecaster":
        # Its production structure (rf_emf_mv/emforecaster): a TSMixer patch
        # embedding into the Linear read-out, the layer CCM clusters.
        args.exp.patch_model_id = "TSMixer"
        args.exp.backbone_id = "Linear"
        args.sl.patch_embed_dim = 16
    if model_id == "RF_EMF_Transformer":
        # The one width its backbone runs at (rf_emf_mv/transformer): conv1
        # folds d_model 2 -> 1 ahead of the fc2 read-out CCM clusters.
        args.sl.d_model = 2
    if model_id == "LSTM":
        # Its production structure (rf_emf_mv/lstm): patches projected into a
        # two-layer LSTM whose averaged state feeds the linear head CCM
        # clusters.
        args.data.patching = True
        args.sl.last_state = False
        args.sl.avg_state = True
        args.sl.norm_mode = "layer"
        args.sl.num_enc_layers = 2
    return args


# The grids' production shapes: the EMF benchmark on ED2022's 27 bands, and
# VoIP with its grid overrides.
JOB_SHAPES = {
    "emf": ("ED2022", 27, 336, 96, None),
    "voip": ("VOIP_MOBILE", 6, 48, 12, grids.VOIP_OVERRIDES),
}


def make_job_args(model_id, k=None, shape="emf", seed=grid.SEED):
    """The config a grid cell hands Experiment, at production size.

    The model's rf_emf_mv/* job config with the cell's window, the grid's
    overrides and, for k, the cluster switch: what run_grid sets, minus the
    host profile. num_channels is set by hand because get_loaders infers it
    from the data, which a build does not load.
    """
    dataset, num_channels, seq_len, pred_len, overrides = JOB_SHAPES[shape]
    job = {mv_id: job for _, mv_id, job in grids.MV_MODELS}[model_id]
    load_config, _ = grid.setup_modules("multivariate")
    args = load_config(grid.args_path("multivariate", job))
    args.data.dataset = dataset
    args.data.num_channels = num_channels
    args.data.seq_len = seq_len
    args.data.pred_len = pred_len
    args.exp.seed_list = [seed]
    args.exp.seed = seed
    if k is not None:
        args.cluster.enabled = True
        args.cluster.k_clusters = k
    grid.apply_overrides(args, overrides)
    return args


def build(args):
    """get_model from the RNG state Experiment.run seeds before it builds."""
    torch.manual_seed(args.exp.seed)
    return get_model(args)


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


@pytest.mark.parametrize("model_id", sorted(CCM_MODELS))
def test_ccm_membership_routes_the_cluster_wise_layers(model_id):
    # With k>1, the same input under a different membership must produce a
    # different forecast; otherwise the "clustering" is decoration.
    model = get_model(make_args(model_id))
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
    # TimesNet (since the upstream bridge left) has no linear map applied to
    # each channel on its own to route, and ARIMA and Persistence forecast each
    # band alone with no gradient path; building something else under their
    # name would put a column in the results table that is not the method it
    # claims to be. The bare RecurrentModel id is out too: the grids run the
    # LSTM backbone under its LSTM alias, and no other backbone is adapted.
    for model_id in ("TimesNet", "ARIMA", "Persistence", "RecurrentModel"):
        with pytest.raises(ValueError, match="no CCM adaptation"):
            get_model(make_args(model_id))


@pytest.mark.parametrize("model_id", sorted(CCM_MODELS))
def test_cluster_built_models_require_the_membership_at_forward(model_id):
    # A cluster-built backbone silently run without its membership would be an
    # un-clustered model wearing the cluster cell's name.
    model = get_model(make_args(model_id))
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


@pytest.mark.parametrize("model_id", sorted(CCM_MODELS))
def test_multivariate_models_are_untouched_when_clustering_is_off(model_id):
    # The MV and per-channel grids must keep building the exact baselines: no
    # cluster modules in the tree, and the plain forward signature still works.
    args = make_args(model_id)
    args.cluster.enabled = False
    model = get_model(args)
    assert not any(isinstance(m, (Cluster_wise_linear, Cluster_assigner))
                   for m in model.modules())
    assert model(torch.randn(2, NUM_CHANNELS, SEQ_LEN)).shape == (
        2, NUM_CHANNELS, PRED_LEN)


# ---------------------------------------------------------------------------
# EMForecaster: CCM on the read-out its channels share
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", sorted(JOB_SHAPES))
def test_ccm_k1_emforecaster_is_the_multivariate_build(shape):
    """k=1 is the multivariate cell bit for bit, with no weights copied over:
    its one read-out copy is drawn as the multivariate read-out, and the all-
    ones membership passes the forecast through unchanged."""
    mv = build(make_job_args("EMForecaster", shape=shape))
    ccm = build(make_job_args("EMForecaster", k=1, shape=shape))
    assert isinstance(ccm.backbone, EMForecaster)

    want = mv.state_dict()
    got = {name.replace("base_model.linears.0.", "base_model."): tensor
           for name, tensor in ccm.backbone.state_dict().items()}
    assert list(got) == list(want)
    for name, tensor in want.items():
        assert torch.equal(got[name], tensor), name

    _, num_channels, seq_len, _, _ = JOB_SHAPES[shape]
    x = torch.randn(3, num_channels, seq_len)
    mv.eval()
    ccm.eval()
    with torch.no_grad():
        assert torch.equal(ccm(x), mv(x))


def test_emforecaster_backbone_init_is_the_multivariate_build_at_every_k(
        monkeypatch):
    """Clustering the read-out moves nothing else. At every k the rest of the
    network is the multivariate init, and the build leaves the global RNG where
    the multivariate build does, so the assigner and dropout draw the same
    stream. Copy 0 of the read-out is the multivariate one; the others are
    distinct draws, and copy j is the same whatever k is."""
    # The backbone exactly as get_model builds it, before the wrapper's
    # assigner draws from the RNG.
    monkeypatch.setattr(models, "CCMForecaster", lambda args, backbone: backbone)
    mv = build(make_job_args("EMForecaster"))
    mv_rng = torch.get_rng_state()
    want = mv.state_dict()
    del mv

    reference = None
    for k in (9, 5, 3, 2, 1):
        backbone = build(make_job_args("EMForecaster", k=k))
        assert torch.equal(torch.get_rng_state(), mv_rng), k

        state = backbone.state_dict()
        copies = {f"base_model.linears.{i}.{p}" for i in range(k)
                  for p in ("weight", "bias")}
        assert set(state) == set(want) - {"base_model.weight",
                                          "base_model.bias"} | copies
        for name, tensor in want.items():
            if not name.startswith("base_model."):
                assert torch.equal(state[name], tensor), (k, name)

        linears = backbone.base_model.linears
        assert torch.equal(linears[0].weight, want["base_model.weight"])
        assert all(torch.equal(linear.bias, want["base_model.bias"])
                   for linear in linears)
        for i in range(k):
            assert not any(torch.equal(linears[i].weight, linears[j].weight)
                           for j in range(i)), (k, i)
        if reference is None:
            reference = [linear.weight for linear in linears]
        for i in range(k):
            assert torch.equal(linears[i].weight, reference[i]), (k, i)


def test_emforecaster_routes_each_channel_by_its_own_membership_row():
    """Channel c's read-out is its own mixture, sum_j p[c, j] * copy_j(h[b, c]),
    so moving one channel's membership row moves that channel's forecast and no
    other's. A uniform membership, or one shared by every channel, would pass
    with the rows misaligned."""
    model = get_model(make_args("EMForecaster"))
    model.eval()
    linears = model.backbone.base_model.linears
    with torch.no_grad():
        # xavier_init leaves the biases at zero; nonzero ones check their
        # mixture too.
        for linear in linears:
            linear.bias.normal_()
    seen = {}
    model.backbone.base_model.register_forward_hook(
        lambda module, inputs, out: seen.update(h=inputs[0], out=out))

    x = torch.randn(2, NUM_CHANNELS, SEQ_LEN)
    prob = torch.softmax(3 * torch.randn(NUM_CHANNELS, K), dim=-1)
    moved = prob.clone()
    moved[2] = torch.eye(K)[prob[2].argmin()]
    with torch.no_grad():
        out = model.backbone(x, cluster_prob=prob)
        want = sum(prob[:, j, None] * linears[j](seen["h"]) for j in range(K))
        torch.testing.assert_close(seen["out"], want)
        out_moved = model.backbone(x, cluster_prob=moved)
    assert [c for c in range(NUM_CHANNELS)
            if not torch.equal(out_moved[:, c], out[:, c])] == [2]


# Each cluster beyond the first adds one more copy of the read-out,
# Linear(num_patches x patch_embed_dim -> pred_len): 84 x 256 -> 96, 2,064,480
# parameters, on EMF; 12 x 256 -> 12, 36,876, on VoIP. Any k adds the
# assigner's Linear(seq_len -> sl.d_model=128): 43,136 on EMF, 6,272 on VoIP.
@pytest.mark.parametrize("shape, k, expected", [
    ("emf", None, 2_416_959), ("emf", 1, 2_460_095), ("emf", 2, 4_524_575),
    ("emf", 3, 6_589_055), ("emf", 5, 10_718_015), ("emf", 9, 18_975_935),
    ("voip", None, 315_441), ("voip", 1, 321_713), ("voip", 2, 358_589),
    ("voip", 3, 395_465), ("voip", 6, 506_093),
])
def test_emforecaster_parameter_counts_at_production_size(shape, k, expected):
    model = build(make_job_args("EMForecaster", k=k, shape=shape))
    assert sum(p.numel() for p in model.parameters()) == expected


def test_emforecaster_clusters_only_the_patching_linear_read_out():
    # Any other structure either has no Linear read-out to swap or has lost the
    # channel axis the membership indexes; it must fail at build rather than
    # train something else under the cluster cell's name.
    args = make_args("EMForecaster")
    args.exp.backbone_id = "DLinear"
    with pytest.raises(ValueError, match="Linear read-out"):
        get_model(args)

    args = make_args("EMForecaster")
    for structure in ({"patching_on": False, "num_patches": 7},
                      {"return_head": False}):
        with pytest.raises(ValueError, match="Linear read-out"):
            EMForecaster(args=args, seq_len=SEQ_LEN, pred_len=PRED_LEN,
                         num_channels=NUM_CHANNELS, patch_model_id="TSMixer",
                         backbone_id="Linear", patch_dim=8, patch_stride=4,
                         patch_embed_dim=16, n_clusters=K, **structure)


def test_cluster_emforecaster_cells_keep_the_multivariate_micro_batches():
    # The memory profile is keyed on the model, so a cluster EMForecaster cell
    # trains 128 x 4 like the multivariate cell it is compared against.
    for dataset in ("ED2022", "UH2022", "EMF_2023_university_5_with_PTV"):
        for k in (None, 1, 9):
            ablation = grid.cell_ablation("EMForecaster", dataset, 336, 96, 0.1,
                                          grid.SEED, k=k)
            assert (ablation["global.sl.batch_size"],
                    ablation["global.sl.accum_steps"]) == (128, 4), (dataset, k)


# ---------------------------------------------------------------------------
# Transformer: CCM on the fc2 read-out its channels share
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", sorted(JOB_SHAPES))
def test_ccm_k1_transformer_is_the_multivariate_build(shape):
    """k=1 is the multivariate cell bit for bit, with no weights copied over:
    its one fc2 copy is drawn where the multivariate fc2 is, and the all-ones
    membership passes the forecast through unchanged."""
    mv = build(make_job_args("RF_EMF_Transformer", shape=shape))
    ccm = build(make_job_args("RF_EMF_Transformer", k=1, shape=shape))
    assert isinstance(ccm.backbone, Transformer)

    want = mv.state_dict()
    got = {name.replace("fc2.linears.0.", "fc2."): tensor
           for name, tensor in ccm.backbone.state_dict().items()}
    assert list(got) == list(want)
    for name, tensor in want.items():
        assert torch.equal(got[name], tensor), name

    _, num_channels, seq_len, _, _ = JOB_SHAPES[shape]
    x = torch.randn(3, num_channels, seq_len)
    mv.eval()
    ccm.eval()
    with torch.no_grad():
        assert torch.equal(ccm(x), mv(x))


def test_transformer_backbone_init_is_the_multivariate_build_at_every_k(
        monkeypatch):
    """Clustering fc2 moves nothing else. fc2 is the last module the build
    draws for, so at every k the rest of the network is the multivariate init
    and copy 0 is the multivariate fc2; the other copies are the draws that
    follow it, distinct, and copy j is the same whatever k is. Those draws
    leave the global RNG where the multivariate build does only at k=1: past
    it the assigner and training draw a shifted stream, as for DLinear,
    PatchTST and TSMixer."""
    # The backbone exactly as get_model builds it, before the wrapper's
    # assigner draws from the RNG.
    monkeypatch.setattr(models, "CCMForecaster", lambda args, backbone: backbone)
    mv = build(make_job_args("RF_EMF_Transformer"))
    mv_rng = torch.get_rng_state()
    want = mv.state_dict()
    del mv

    reference = None
    for k in (9, 5, 3, 2, 1):
        backbone = build(make_job_args("RF_EMF_Transformer", k=k))
        assert torch.equal(torch.get_rng_state(), mv_rng) == (k == 1), k

        state = backbone.state_dict()
        copies = {f"fc2.linears.{i}.{p}" for i in range(k)
                  for p in ("weight", "bias")}
        assert set(state) == set(want) - {"fc2.weight", "fc2.bias"} | copies
        for name, tensor in want.items():
            if not name.startswith("fc2."):
                assert torch.equal(state[name], tensor), (k, name)

        linears = backbone.fc2.linears
        assert torch.equal(linears[0].weight, want["fc2.weight"])
        assert torch.equal(linears[0].bias, want["fc2.bias"])
        for i in range(k):
            for p in ("weight", "bias"):
                assert not any(torch.equal(getattr(linears[i], p),
                                           getattr(linears[j], p))
                               for j in range(i)), (k, i, p)
        if reference is None:
            reference = [(linear.weight, linear.bias) for linear in linears]
        for i in range(k):
            assert torch.equal(linears[i].weight, reference[i][0]), (k, i)
            assert torch.equal(linears[i].bias, reference[i][1]), (k, i)


def test_transformer_routes_each_channel_by_its_own_membership_row():
    """Channel c's read-out is its own mixture, sum_j p[c, j] * copy_j(h[b, c]),
    where h[b, c] is channel c's encoding of window b alone: the fold into the
    batch and back keeps every channel on its own row. So moving one channel's
    membership row, or its input, moves that channel's forecast and no
    other's."""
    model = get_model(make_args("RF_EMF_Transformer"))
    model.eval()
    linears = model.backbone.fc2.linears
    seen = {}
    model.backbone.fc2.register_forward_hook(
        lambda module, inputs, out: seen.update(h=inputs[0], out=out))

    x = torch.randn(2, NUM_CHANNELS, SEQ_LEN)
    prob = torch.softmax(3 * torch.randn(NUM_CHANNELS, K), dim=-1)
    moved = prob.clone()
    moved[2] = torch.eye(K)[prob[2].argmin()]
    x_moved = x.clone()
    x_moved[:, 4] = torch.randn(2, SEQ_LEN)
    with torch.no_grad():
        out = model.backbone(x, cluster_prob=prob)
        assert seen["h"].shape == (2, NUM_CHANNELS, SEQ_LEN)
        want = sum(prob[:, j, None] * linears[j](seen["h"]) for j in range(K))
        torch.testing.assert_close(seen["out"], want)
        out_moved = model.backbone(x, cluster_prob=moved)
        out_x_moved = model.backbone(x_moved, cluster_prob=prob)

    def changed(other):
        return [c for c in range(NUM_CHANNELS)
                if not torch.equal(other[:, c], out[:, c])]

    assert changed(out_moved) == [2]
    assert changed(out_x_moved) == [4]


# Each cluster beyond the first adds one more copy of fc2, Linear(seq_len ->
# pred_len): 336 -> 96, 32,352 parameters, on EMF; 48 -> 12, 588, on VoIP. Any
# k adds the assigner's Linear(seq_len -> sl.d_model=2): 674 on EMF, 98 on VoIP.
@pytest.mark.parametrize("shape, k, expected", [
    ("emf", None, 372_738), ("emf", 1, 373_412), ("emf", 2, 405_764),
    ("emf", 3, 438_116), ("emf", 5, 502_820), ("emf", 9, 632_228),
    ("voip", None, 7_782), ("voip", 1, 7_880), ("voip", 2, 8_468),
    ("voip", 3, 9_056), ("voip", 6, 10_820),
])
def test_transformer_parameter_counts_at_production_size(shape, k, expected):
    model = build(make_job_args("RF_EMF_Transformer", k=k, shape=shape))
    assert sum(p.numel() for p in model.parameters()) == expected


def test_cluster_transformer_needs_its_channel_count():
    # The plain Transformer folds any channel count into the batch and never
    # needs it; the cluster-wise read-out is sized for the membership's rows.
    with pytest.raises(ValueError, match="needs num_channels"):
        Transformer(seq_len=SEQ_LEN, pred_len=PRED_LEN, d_model=2, n_clusters=K)
    plain = Transformer(seq_len=SEQ_LEN, pred_len=PRED_LEN, d_model=2)
    assert plain(torch.randn(2, NUM_CHANNELS, SEQ_LEN)).shape == (
        2, NUM_CHANNELS, PRED_LEN)


# ---------------------------------------------------------------------------
# LSTM: CCM on the linear head its channels share
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", sorted(JOB_SHAPES))
def test_ccm_k1_lstm_is_the_multivariate_build(shape):
    """k=1 is the multivariate cell bit for bit, with no weights copied over:
    its one head copy is drawn where the multivariate head is, and the all-ones
    membership passes the forecast through unchanged."""
    mv = build(make_job_args("LSTM", shape=shape))
    ccm = build(make_job_args("LSTM", k=1, shape=shape))
    assert isinstance(ccm.backbone, RecurrentModel)

    want = mv.state_dict()
    got = {name.replace("head.linears.0.", "head."): tensor
           for name, tensor in ccm.backbone.state_dict().items()}
    assert list(got) == list(want)
    for name, tensor in want.items():
        assert torch.equal(got[name], tensor), name

    _, num_channels, seq_len, _, _ = JOB_SHAPES[shape]
    x = torch.randn(3, num_channels, seq_len)
    mv.eval()
    ccm.eval()
    with torch.no_grad():
        assert torch.equal(ccm(x), mv(x))


def test_lstm_backbone_init_is_the_multivariate_build_at_every_k(monkeypatch):
    """Clustering the head moves nothing else. xavier_init redraws the weights
    at the end of the build, so the head's construction must leave the stream
    it redraws from as the multivariate build leaves it: at every k the
    rest of the network is the multivariate init and copy 0 is the
    multivariate head; the other copies are the draws that follow it, distinct,
    and copy j is the same whatever k is. Those draws leave the global RNG
    where the multivariate build does only at k=1: past it the assigner and
    training draw a shifted stream, as for DLinear, PatchTST and TSMixer."""
    # The backbone exactly as get_model builds it, before the wrapper's
    # assigner draws from the RNG.
    monkeypatch.setattr(models, "CCMForecaster", lambda args, backbone: backbone)
    mv = build(make_job_args("LSTM"))
    mv_rng = torch.get_rng_state()
    want = mv.state_dict()
    del mv

    reference = None
    for k in (9, 5, 3, 2, 1):
        backbone = build(make_job_args("LSTM", k=k))
        assert torch.equal(torch.get_rng_state(), mv_rng) == (k == 1), k

        state = backbone.state_dict()
        copies = {f"head.linears.{i}.{p}" for i in range(k)
                  for p in ("weight", "bias")}
        assert set(state) == set(want) - {"head.weight", "head.bias"} | copies
        for name, tensor in want.items():
            if not name.startswith("head."):
                assert torch.equal(state[name], tensor), (k, name)

        linears = backbone.head.linears
        assert torch.equal(linears[0].weight, want["head.weight"])
        # xavier_init zeroes every bias, the multivariate head's included.
        assert all(torch.equal(linear.bias, want["head.bias"])
                   for linear in linears)
        for i in range(k):
            assert not any(torch.equal(linears[i].weight, linears[j].weight)
                           for j in range(i)), (k, i)
        if reference is None:
            reference = [linear.weight for linear in linears]
        for i in range(k):
            assert torch.equal(linears[i].weight, reference[i]), (k, i)


def test_lstm_routes_each_channel_by_its_own_membership_row():
    """Channel c's read-out is its own mixture, sum_j p[c, j] * copy_j(h[b, c]),
    where h[b, c] is channel c's averaged LSTM state for window b alone: the
    fold into the batch and back keeps every channel on its own row. So moving
    one channel's membership row, or its input, moves that channel's forecast
    and no other's."""
    model = get_model(make_args("LSTM"))
    model.eval()
    linears = model.backbone.head.linears
    with torch.no_grad():
        # xavier_init leaves the biases at zero; nonzero ones check their
        # mixture too.
        for linear in linears:
            linear.bias.normal_()
    seen = {}
    model.backbone.head.register_forward_hook(
        lambda module, inputs, out: seen.update(h=inputs[0], out=out))

    x = torch.randn(2, NUM_CHANNELS, SEQ_LEN)
    prob = torch.softmax(3 * torch.randn(NUM_CHANNELS, K), dim=-1)
    moved = prob.clone()
    moved[2] = torch.eye(K)[prob[2].argmin()]
    x_moved = x.clone()
    x_moved[:, 4] = torch.randn(2, SEQ_LEN)
    with torch.no_grad():
        out = model.backbone(x, cluster_prob=prob)
        assert seen["h"].shape == (2, NUM_CHANNELS, 16)
        want = sum(prob[:, j, None] * linears[j](seen["h"]) for j in range(K))
        torch.testing.assert_close(seen["out"], want)
        out_moved = model.backbone(x, cluster_prob=moved)
        out_x_moved = model.backbone(x_moved, cluster_prob=prob)

    def changed(other):
        return [c for c in range(NUM_CHANNELS)
                if not torch.equal(other[:, c], out[:, c])]

    assert changed(out_moved) == [2]
    assert changed(out_x_moved) == [4]


# Each cluster beyond the first adds one more copy of the head, Linear(d_model
# -> pred_len): 32 -> 96, 3,168 parameters, on EMF; 32 -> 12, 396, on VoIP. Any
# k adds the assigner's Linear(seq_len -> sl.d_model=32): 10,784 on EMF, 1,568
# on VoIP.
@pytest.mark.parametrize("shape, k, expected", [
    ("emf", None, 20_726), ("emf", 1, 31_510), ("emf", 2, 34_678),
    ("emf", 3, 37_846), ("emf", 5, 44_182), ("emf", 9, 56_854),
    ("voip", None, 17_912), ("voip", 1, 19_480), ("voip", 2, 19_876),
    ("voip", 3, 20_272), ("voip", 6, 21_460),
])
def test_lstm_parameter_counts_at_production_size(shape, k, expected):
    model = build(make_job_args("LSTM", k=k, shape=shape))
    assert sum(p.numel() for p in model.parameters()) == expected


def test_lstm_clusters_only_the_patched_linear_head():
    # Unpatched, the LSTM reads the channels as the features of one sequence,
    # so no channel axis reaches the head for the membership to index; an MLP
    # head has no cluster-wise form, and with the head off there is nothing to
    # route. Another backbone under the LSTM alias would be a different network
    # in the LSTM column. Each must fail at build rather than train something
    # else under the cluster cell's name.
    for setting in ({"data.patching": False}, {"sl.head_type": "mlp"},
                    {"exp.backbone_id": "GRU"}, {"exp.backbone_id": "RNN"},
                    {"exp.backbone_id": "Mamba"}):
        args = make_args("LSTM")
        grid.apply_overrides(args, setting)
        with pytest.raises(ValueError, match="linear head"):
            get_model(args)

    with pytest.raises(ValueError, match="linear head"):
        RecurrentModel(d_model=16, num_enc_layers=2, pred_len=PRED_LEN,
                       backbone_id="LSTM", seq_len=SEQ_LEN, patching=True,
                       patch_dim=8, patch_stride=4, num_channels=NUM_CHANNELS,
                       last_state=False, avg_state=True, return_head=False,
                       n_clusters=K)


# ---------------------------------------------------------------------------
# Fixed-membership evaluation (the primary protocol; upstream's batch one stays
# logged beside it)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_id", sorted(CCM_MODELS))
def test_window_prob_pooled_over_the_batch_is_the_assigners_membership(model_id):
    # The fixed membership reuses the first half of upstream's assigner; pooled
    # the way upstream pools it, it must be that assigner's output, bit for bit.
    model = get_model(make_args(model_id))
    model.eval()
    x = torch.randn(5, NUM_CHANNELS, SEQ_LEN)
    with torch.no_grad():
        upstream, _ = model.assigner(x.permute(0, 2, 1), model.cluster_emb)
        pooled = sinkhorn(model.window_prob(x).mean(dim=0),
                          epsilon=model.assigner.epsilon)
    assert torch.equal(pooled, upstream)


def test_fixed_membership_is_the_train_mean_however_it_is_batched():
    model = get_model(make_args("DLinear"))
    x = torch.randn(10, NUM_CHANNELS, SEQ_LEN)
    with torch.no_grad():
        expected = sinkhorn(model.window_prob(x).mean(dim=0),
                            epsilon=model.assigner.epsilon)

    whole = model.fit_fixed_membership([x]).clone()
    split = model.fit_fixed_membership([x[:3], x[3:4], x[4:]]).clone()
    shuffled = model.fit_fixed_membership([x[torch.randperm(10)]]).clone()
    torch.testing.assert_close(whole, expected)
    torch.testing.assert_close(split, whole)
    torch.testing.assert_close(shuffled, whole)
    assert model.fixed_prob.shape == (NUM_CHANNELS, K)
    torch.testing.assert_close(model.fixed_prob.sum(dim=1), torch.ones(NUM_CHANNELS))


@pytest.mark.parametrize("model_id", sorted(CCM_MODELS))
def test_fixed_forecasts_do_not_depend_on_the_eval_batch(model_id):
    """The point of the fixed protocol (finding C1): a window's forecast is a
    function of its own lookback, not of which windows share its batch."""
    model = get_model(make_args(model_id))
    model.eval()
    x = torch.randn(8, NUM_CHANNELS, SEQ_LEN)
    model.fit_fixed_membership([torch.randn(16, NUM_CHANNELS, SEQ_LEN)])
    perm = torch.randperm(8)

    with torch.no_grad():
        # The batch protocol is batch-dependent, or this test would prove nothing.
        assert not torch.allclose(model(x[:1]), model(x)[:1])
        with model.fixed_membership():
            whole = model(x)
            one_by_one = torch.cat([model(x[i:i + 1]) for i in range(8)])
            reordered = torch.empty_like(whole)
            reordered[perm] = model(x[perm])
            halves = torch.cat([model(x[:5]), model(x[5:])])
    torch.testing.assert_close(one_by_one, whole)
    torch.testing.assert_close(reordered, whole)
    torch.testing.assert_close(halves, whole)


@pytest.mark.parametrize("model_id", sorted(CCM_MODELS))
def test_k1_fixed_protocol_is_the_batch_protocol(model_id):
    # One cluster: the membership is all ones however it is pooled, so the two
    # protocols are the same forecast, bit for bit.
    model = get_model(make_args(model_id, k=1))
    model.eval()
    x = torch.randn(4, NUM_CHANNELS, SEQ_LEN)
    model.fit_fixed_membership([torch.randn(16, NUM_CHANNELS, SEQ_LEN)])
    assert torch.equal(model.fixed_prob, torch.ones(NUM_CHANNELS, 1))
    with torch.no_grad():
        batch = model(x)
        with model.fixed_membership():
            fixed = model(x)
    assert torch.equal(fixed, batch)


def test_fixed_prob_is_checkpointed_only_once_fitted():
    """A fresh build and every checkpoint saved during training keep their old
    key set (the 375 existing cluster checkpoints load as before); a fitted
    membership travels with the weights it was fitted for."""
    model = get_model(make_args("DLinear"))
    old_keys = set(model.state_dict())
    assert "fixed_prob" not in old_keys
    old_checkpoint = model.state_dict()

    model.fit_fixed_membership([torch.randn(8, NUM_CHANNELS, SEQ_LEN)])
    assert set(model.state_dict()) == old_keys | {"fixed_prob"}

    fresh = get_model(make_args("DLinear"))
    fresh.load_state_dict(model.state_dict())  # strict=True, as Experiment.test does
    torch.testing.assert_close(fresh.fixed_prob, model.fixed_prob)
    model.eval()
    fresh.eval()
    x = torch.randn(3, NUM_CHANNELS, SEQ_LEN)
    with torch.no_grad(), model.fixed_membership(), fresh.fixed_membership():
        torch.testing.assert_close(fresh(x), model(x))

    # An old checkpoint loads strictly into a fitted model too, and drops the
    # membership that belonged to other weights.
    fresh.load_state_dict(old_checkpoint)
    assert fresh.fixed_prob is None
    assert set(fresh.state_dict()) == old_keys


def test_fixed_membership_needs_a_fit_and_stays_out_of_training():
    model = get_model(make_args("DLinear"))
    x = torch.randn(2, NUM_CHANNELS, SEQ_LEN)
    with pytest.raises(RuntimeError, match="fit_fixed_membership"):
        with model.fixed_membership():
            pass

    model.fit_fixed_membership([x])
    model.train()
    with pytest.raises(RuntimeError, match="evaluation protocol"):
        with model.fixed_membership():
            model(x)
    # The block restores upstream's protocol even when it raised.
    assert model.use_fixed_prob is False
    model(x)


def test_cluster_run_rejects_conformal_validation_eval():
    # The fixed protocol recalibrates on validation at test time; validation_eval
    # would score it against the batch calibration saved during training.
    args = make_args("DLinear")
    args.conf.conf = True
    args.conf.validation_eval = True
    with pytest.raises(ValueError, match="validation_eval"):
        get_model(args)


def _stub_experiment(args, **windows):
    """An Experiment with a shuffled, drop_last loader over each split's windows
    and nothing else set up, as a re-score of a saved checkpoint builds one."""
    exp = Experiment(args)
    exp.device = torch.device("cpu")
    for split, x in windows.items():
        dataset = TensorDataset(x, torch.zeros(len(x), NUM_CHANNELS, PRED_LEN))
        setattr(exp, f"{split}_loader",
                DataLoader(dataset, batch_size=5, shuffle=True, drop_last=True))
    return exp


def test_experiment_fits_the_membership_on_the_whole_train_split_only():
    """The run's membership must see every training window -- whatever the
    training loader's shuffle and drop_last -- and no evaluation window: fitted
    on validation or test it would leak the windows it then scores."""
    args = make_args("DLinear")
    model = get_model(args)
    windows = {"train": torch.randn(12, NUM_CHANNELS, SEQ_LEN),
               "val": torch.randn(7, NUM_CHANNELS, SEQ_LEN),
               "test": torch.randn(7, NUM_CHANNELS, SEQ_LEN)}
    exp = _stub_experiment(args, **windows)

    fitted = exp.fit_fixed_membership(model).clone()
    own = {split: model.fit_fixed_membership([x]).clone()
           for split, x in windows.items()}
    torch.testing.assert_close(fitted, own["train"])
    assert not torch.allclose(fitted, own["val"])
    assert not torch.allclose(fitted, own["test"])


def test_fixed_evaluation_refuses_a_forward_handle_on_another_model():
    # evaluate() forwards through exp.model_fwd; were that a different module
    # than the one holding the fixed membership, the batch protocol would be
    # logged under the sl_*_fixed keys.
    args = make_args("DLinear")
    model = get_model(args)
    model.fit_fixed_membership([torch.randn(4, NUM_CHANNELS, SEQ_LEN)])
    exp = _stub_experiment(args)
    exp.model_fwd = get_model(args)
    with pytest.raises(RuntimeError, match="model_fwd"):
        exp.evaluate_fixed_membership(model, "DLinear", torch.nn.MSELoss())


# One VoIP cell trained end to end (Experiment.run, CPU, two epochs) per case,
# shared by the tests below: the multivariate DLinear cell and its k=1 and k=3
# cluster twins.
E2E_CASES = {"mv": None, "k1": 1, "k3": 3}


def _run_voip_cell(k, logs_root, job="rf_emf_mv/dlinear", settings=None):
    setup = "voip_cluster" if k else "voip"
    load_config, Experiment = grid.setup_modules(setup)
    args = load_config(grid.args_path(setup, job))
    # What run_grid sets for a cell, minus the host profile.
    args.data.dataset = "VOIP_MOBILE"
    args.data.seq_len = 48
    args.data.pred_len = 12
    args.conf.conf = True
    args.conf.alpha = grids.GRIDS[setup]["alpha"]
    args.exp.neptune = False
    args.exp.id = f"k{k}" if k else ""
    args.exp.seed_list = [grid.SEED]
    args.exp.seed = grid.SEED
    args.exp.compile = False
    if k:
        args.cluster.enabled = True
        args.cluster.k_clusters = k
    grid.apply_overrides(args, grids.GRIDS[setup]["overrides"])
    args.sl.epochs = 2
    args.data.num_workers = 0
    args.data.pin_memory = False
    # The test's own settings, last so they win.
    grid.apply_overrides(args, settings)

    # Experiment writes under ./logs. One thread: on a shared many-core box the
    # default thread count makes this tiny model ~20x slower.
    cwd, threads = os.getcwd(), torch.get_num_threads()
    os.chdir(logs_root)
    torch.set_num_threads(1)
    exp = Experiment(args)
    try:
        exp.run()
    finally:
        os.chdir(cwd)
        torch.set_num_threads(threads)
        plt.close("all")
    (log_path,) = glob.glob(os.path.join(logs_root, "logs", "**", "log.json"),
                            recursive=True)
    with open(log_path) as f:
        log = json.load(f)
    checkpoint = torch.load(os.path.join(os.path.dirname(log_path), "supervised.pth"))
    return log, checkpoint, exp


@pytest.fixture(scope="module")
def voip_runs(tmp_path_factory):
    return {name: _run_voip_cell(k, tmp_path_factory.mktemp(name))
            for name, k in E2E_CASES.items()}


def _metric_keys(log):
    return [key for key in log if not key.startswith("parameters/")]


def _assert_both_protocols_before_the_marker(log, checkpoint, k):
    """A VoIP cluster run: every batch-protocol test metric has its _fixed
    twin, all logged before the resume marker, and the checkpoint carries the
    six channels' k-way membership. The eval path is the same for every model,
    so each model's end-to-end run is held to this."""
    keys = _metric_keys(log)
    test_keys = [key for key in keys if key.startswith("sl_test/")]
    assert "sl_test/ic" in test_keys and grid.MARKER in test_keys

    fixed = [key for key in keys if "_fixed/" in key]
    assert "sl_val_fixed/loss" in fixed
    assert {key for key in fixed if key.startswith("sl_test_fixed/")} == {
        key.replace("sl_test/", "sl_test_fixed/") for key in test_keys}

    # The resume marker is still the last thing logged: only its own sibling,
    # written with it as before, comes after.
    marker = keys.index(grid.MARKER)
    assert keys[marker + 1:] == ["sl_test/interval_width_std"]
    assert all(keys.index(key) < marker for key in fixed)

    assert checkpoint["fixed_prob"].shape == (6, k)
    torch.testing.assert_close(checkpoint["fixed_prob"].sum(dim=1), torch.ones(6))


@pytest.mark.parametrize("name", ["k1", "k3"])
def test_cluster_run_logs_both_protocols_before_the_marker(voip_runs, name):
    log, checkpoint, _ = voip_runs[name]
    _assert_both_protocols_before_the_marker(log, checkpoint, E2E_CASES[name])


def test_k1_run_fixed_metrics_equal_the_batch_metrics(voip_runs):
    log, _, _ = voip_runs["k1"]
    for key in _metric_keys(log):
        if key.startswith("sl_test/"):
            assert log[key.replace("sl_test/", "sl_test_fixed/")] == log[key], key
    # Validation is shuffled, so the two passes sum the same losses in a
    # different order: equal up to float summation, not bitwise.
    assert log["sl_val_fixed/loss"][0] == pytest.approx(min(log["sl_val/loss"]),
                                                         rel=1e-6)


def test_k3_run_fixed_metrics_are_their_own(voip_runs):
    # Not a copy of the batch protocol: at k>1 the membership differs.
    log, _, _ = voip_runs["k3"]
    assert log["sl_test_fixed/mse"] != log["sl_test/mse"]
    # The interval width is a function of the calibration scores alone: equal
    # widths would mean the fixed test reused the batch calibration.
    assert (log["sl_test_fixed/interval_width_mean"]
            != log["sl_test/interval_width_mean"])


def test_k3_run_calibrates_the_fixed_intervals_under_the_fixed_membership(
        voip_runs):
    """The scores the fixed test intervals were drawn from (the run's last
    calibration) are validation's under the fixed membership, recomputed here,
    and not validation's under the batch protocol."""
    _, _, exp = voip_runs["k3"]
    run_scores = exp.calibration_scores

    def calibrate_on_validation():
        exp.evaluate(model=exp.model, model_id=exp.args.exp.model_id,
                     loader=exp.val_loader, criterion=exp.criterion, flag="sl",
                     conformal_calibrate=True)
        return exp.calibration_scores

    with exp.model.fixed_membership():
        fixed = calibrate_on_validation()
    batch = calibrate_on_validation()
    for key in ("uncorrected", "corrected"):
        torch.testing.assert_close(run_scores[key], fixed[key])
        assert not torch.allclose(run_scores[key], batch[key])


def test_cluster_emforecaster_trains_on_accumulated_micro_batches(tmp_path):
    """A cluster EMForecaster cell trains and scores end to end through
    Experiment.run under gradient accumulation, as its EMF cells do (128 x 4):
    every micro-batch forward computes its own membership and similarity loss.
    Scaled down (16 x 4, a narrower patch embedding, one epoch) to run on CPU
    in seconds."""
    log, checkpoint, exp = _run_voip_cell(
        2, tmp_path, job="rf_emf_mv/emforecaster",
        settings={"sl.batch_size": 16, "sl.accum_steps": 4, "sl.epochs": 1,
                  "sl.patch_embed_dim": 32, "tsmixer.d_model": 32})
    assert isinstance(exp.model.backbone, EMForecaster)
    assert (exp.args.sl.batch_size, exp.args.sl.accum_steps) == (16, 4)
    # At least two optimizer steps of four micro-batches each.
    assert len(exp.train_loader) >= 2 * 4
    for key in ("sl_train/loss", "sl_train/similarity_loss", "sl_val/loss",
                "sl_test/mse", "sl_test_fixed/mse", grid.MARKER):
        assert torch.isfinite(torch.tensor(log[key])).all(), key
    _assert_both_protocols_before_the_marker(log, checkpoint, 2)


def test_cluster_transformer_trains_end_to_end(tmp_path):
    """A cluster Transformer cell trains and scores through Experiment.run,
    both protocols included, at the VoIP grid's own settings (one epoch)."""
    log, checkpoint, exp = _run_voip_cell(
        2, tmp_path, job="rf_emf_mv/transformer", settings={"sl.epochs": 1})
    assert isinstance(exp.model.backbone, Transformer)
    for key in ("sl_train/loss", "sl_train/similarity_loss", "sl_val/loss",
                "sl_test/mse", "sl_test_fixed/mse", grid.MARKER):
        assert torch.isfinite(torch.tensor(log[key])).all(), key
    _assert_both_protocols_before_the_marker(log, checkpoint, 2)


def test_cluster_lstm_trains_end_to_end(tmp_path):
    """A cluster LSTM cell trains and scores through Experiment.run, both
    protocols included, at the VoIP grid's own settings (one epoch)."""
    log, checkpoint, exp = _run_voip_cell(
        2, tmp_path, job="rf_emf_mv/lstm", settings={"sl.epochs": 1})
    assert isinstance(exp.model.backbone, RecurrentModel)
    for key in ("sl_train/loss", "sl_train/similarity_loss", "sl_val/loss",
                "sl_test/mse", "sl_test_fixed/mse", grid.MARKER):
        assert torch.isfinite(torch.tensor(log[key])).all(), key
    _assert_both_protocols_before_the_marker(log, checkpoint, 2)


def test_non_cluster_run_is_untouched_by_the_fixed_protocol(voip_runs):
    log, checkpoint, _ = voip_runs["mv"]
    assert not any("_fixed" in key for key in log)
    assert "fixed_prob" not in checkpoint
    assert _metric_keys(log)[-2:] == [grid.MARKER, "sl_test/interval_width_std"]


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


def test_cluster_models_all_have_a_ccm_adaptation():
    # get_model can build a cluster cell for more models than the grid sweeps
    # yet; the reverse would be a grid cell that fails at build.
    ids = {model_id for _, model_id, _ in grids.CLUSTER_MODELS}
    assert ids <= set(CCM_MODELS)
    # And the sweep itself, so a model joins or leaves it only on purpose.
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
