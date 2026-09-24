"""CCM (Channel Clustering Module) threaded through this repo's own models.

CCM ("From Similarity to Superiority: Channel Clustering for Time Series
Forecasting", NeurIPS 2024) replaces a model's fully-joint or channel-
independent linear maps with k *learned, soft* channel clusters:

  1. Cluster_assigner projects each channel's input window to d_model, takes the
     cosine similarity against k trainable cluster embeddings, and sinkhorn-
     normalizes the result into prob[num_channels, k].
  2. Cluster_wise_linear runs k parallel Linear layers over every channel and
     mixes their outputs by prob -- a soft mixture, not a hard partition.
  3. The cluster embeddings are refreshed from the batch each training step via
     cross-attention, and an auxiliary similarity loss pulls the assignment
     toward the data's own channel-similarity structure.

Both clustering modules are UPSTREAM'S OWN CODE, imported verbatim from the
vendored TimeSeriesCCM-main/ tree (utils/ccm_upstream.py). What changed on
2026-08-25 is the network around them: until then the cluster grid ran
upstream's whole model wrappers (DLinearC, PatchTSTC, ...), a different
implementation from the multivariate cell each cluster cell exists to be
compared against -- different init, RevIN, decomposition kernel. Now the
clustering layers are threaded through this repo's own models instead: a
cluster cell is the multivariate cell's exact network, with the same job
config, hyperparameters and init, its per-channel temporal linears swapped for
upstream's Cluster_wise_linear.

Where the clustering enters each model (mirroring where upstream's own wrappers
put Cluster_wise_linear):

  DLinear   the seasonal and trend linears (the whole model).
  PatchTST  the flatten head; the per-channel attention encoder is untouched.
  TSMixer   the temporal MLP linears in every ResBlock, and the head.

Repo-side adaptations, each marked ADAPTED below or at its site:

  - Cluster-embedding state (ADAPTED 1). Upstream keeps cluster_emb as a bare
    tensor attribute and rebinds it to a fresh Parameter every training
    forward: invisible to .to(device) and state_dict, and a key set that
    depends on how many steps have run. Here it is a registered buffer on this
    wrapper, passed into the assigner explicitly each forward.
  - Refresh shape bug (ADAPTED 2). Cluster_assigner consumes cluster_emb as
    [k, d_model] but returns the cross-attention result [batch, k, d_model];
    upstream writes that 3-D tensor straight back, so training batch two dies
    on `.t()` in the released code. The refresh is pooled over the batch here,
    matching the assigner's own batch-mean convention for the probabilities.
  - Similarity matrix (ADAPTED 3, channel_similarity below). Upstream's
    Exp_CCM builds a batch-wise matrix that is only the right shape in its
    M4/stock setups; its multivariate path loads a precomputed .npy the
    release does not ship. Same kernel, applied along the channel axis.
  - PatchTST depth. Upstream additionally injects the cluster embeddings into
    its attention layers as extra tokens; ours clusters only the head, so the
    encoder stays byte-identical to the multivariate cell's.
  - TimesNet is out. Our TimesNet embeds all channels jointly from the first
    layer, so there is no per-channel linear for CCM to route; upstream's
    TimesNetC restructures the network to create one, which is exactly the
    different-implementation problem this design removes. It also never had a
    multivariate baseline cell (not in MV_MODELS).

get_model (utils/models.py) builds the backbone with the same kwargs as the
multivariate branch plus n_clusters, then wraps it here. The run's data and
optimizer are unchanged; its evaluation adds the fixed-membership protocol
below, with its own conformal recalibration and sl_*_fixed keys, and so refuses
conf.validation_eval=True.

Two evaluation protocols (see fit_fixed_membership):

  batch  upstream's own: every forward pools the membership over the windows
         in its batch, so an eval forecast depends on which other windows
         share its batch -- on the sequential test loader, up to batch_size-1
         later windows whose lookbacks overlap its target. Logged as sl_val/*
         and sl_test/*, and kept as the sensitivity row.
  fixed  one membership per run, pooled over the whole training split after
         the best checkpoint is restored, so a forecast depends on its own
         lookback only. Logged as sl_val_fixed/* and sl_test_fixed/*; the
         primary cluster evaluation.

At k=1 the membership is all ones either way, so the two protocols agree.
"""

import contextlib

import torch
import torch.nn as nn

# The models the cluster grid covers: exactly the ones whose implementations in
# models/ have a per-channel temporal linear for CCM to route. get_model rejects
# a cluster cell for any other model_id before building anything.
CCM_MODELS = frozenset({"DLinear", "PatchTST", "TSMixer"})


class CCMForecaster(nn.Module):
    """One of this repo's models with CCM's channel clustering threaded through.

    The backbone must have been built with n_clusters set (get_model does this);
    this wrapper owns upstream's Cluster_assigner and the similarity loss,
    computes the soft membership once per forward, and hands it to the
    backbone's cluster-wise layers. The Experiment sees the usual
    (batch, num_channels, pred_len) contract, plus `cluster_prob` /
    `similarity_loss` for compute_aux_loss.
    """

    def __init__(self, args, backbone):
        super().__init__()

        model_id = args.exp.model_id
        if model_id not in CCM_MODELS:
            raise ValueError(
                f"{model_id} has no CCM adaptation. The cluster grid covers "
                f"exactly the models whose implementations have a per-channel "
                f"temporal linear to route: {', '.join(sorted(CCM_MODELS))}."
            )
        if args.cluster.k_clusters < 1:
            raise ValueError(
                f"cluster.k_clusters must be at least 1, got {args.cluster.k_clusters}"
            )
        if getattr(backbone, "n_clusters", None) != args.cluster.k_clusters:
            raise ValueError(
                "backbone was not built with n_clusters="
                f"{args.cluster.k_clusters}; build it through get_model."
            )
        # Checked at build, not at test time, so it fails before any training.
        if args.conf.conf and args.conf.validation_eval:
            raise ValueError(
                "A cluster run needs conf.validation_eval=False: the fixed-"
                "membership evaluation recalibrates conformal on validation at "
                "test time, and validation_eval=True would score it against the "
                "batch-protocol calibration saved during training."
            )

        from forecasting.multivariate.utils.ccm_upstream import Cluster_assigner

        self.model_id = model_id
        self.beta = args.cluster.beta
        self.k_clusters = args.cluster.k_clusters
        self.pred_len = args.data.pred_len

        # Upstream's assigner, verbatim. Its width is taken uniformly from the
        # job config's sl block (upstream's own wrappers are inconsistent here:
        # d_ff for DLinearC/TSMixerC, d_model for PatchTSTC); it only sizes the
        # assignment embedding, never the backbone. Built on CPU -- the device
        # arg only places the init-time cluster_emb tensor, which is captured
        # into our buffer on the next line, and Experiment.init_model() moves
        # the whole module afterwards.
        self.assigner = Cluster_assigner(
            args.data.num_channels,
            args.cluster.k_clusters,
            args.data.seq_len,
            args.sl.d_model,
            device="cpu",
        )

        # ADAPTED (1) -- cluster-embedding state. Upstream holds this as a bare
        # tensor attribute (self.assigner.cluster_emb), invisible to .to() and
        # state_dict, and rebinds it mid-training. A registered buffer moves
        # with the module, persists into supervised.pth, and keeps the key set
        # fixed; forward() passes it into the assigner explicitly, so the stale
        # attribute left on the assigner is never read again.
        self.register_buffer(
            "cluster_emb", self.assigner.cluster_emb.detach().clone(), persistent=True
        )

        self.backbone = backbone

        # Set by the last forward; read by similarity_loss().
        self.cluster_prob = None

        # The fixed membership, [num_channels, k], set by fit_fixed_membership.
        # A None buffer is left out of state_dict, so a fresh build and every
        # checkpoint saved during training keep the key set they always had;
        # once set, it is saved with the weights it was fitted for.
        self.register_buffer("fixed_prob", None)
        # Whether forward uses fixed_prob; toggled by fixed_membership().
        self.use_fixed_prob = False

    def forward(self, x):
        if self.use_fixed_prob:
            # Eval only: training keeps upstream's batch membership and refresh.
            if self.training:
                raise RuntimeError(
                    "The fixed membership is an evaluation protocol; switch to "
                    "eval() or leave fixed_membership() before training."
                )
            self.cluster_prob = self.fixed_prob
            return self.backbone(x, cluster_prob=self.fixed_prob)

        # x: (batch, num_channels, seq_len); upstream's assigner works in
        # (batch, seq_len, num_channels).
        prob, refreshed = self.assigner(x.permute(0, 2, 1), self.cluster_emb)

        # ADAPTED (2) -- the cross-attention refresh comes back per-example,
        # [batch, k, d_model]; upstream writes it back as-is and dies on the
        # next batch. Pool it to one [k, d_model] table, the same batch-mean
        # convention the assigner uses for prob, and only on training batches
        # (upstream refreshes on if_update=True forwards, i.e. training only).
        if self.training:
            with torch.no_grad():
                refreshed = refreshed.detach()
                if refreshed.dim() == 3:
                    refreshed = refreshed.mean(dim=0)
                self.cluster_emb.copy_(refreshed)

        self.cluster_prob = prob
        return self.backbone(x, cluster_prob=prob)

    def window_prob(self, x):
        """Each window's own membership scores, before any pooling.

        The first half of upstream's Cluster_assigner.forward
        (TimeSeriesCCM-main/models/layers.py:280-287) on the assigner's own
        layers and this wrapper's cluster_emb: cosine similarity of each
        channel's projected lookback against the k cluster embeddings. Upstream
        then averages this over the batch (:291) and sinkhorns the mean (:292);
        fit_fixed_membership takes the same two steps over the training split.

        Args:
            x: (batch, num_channels, seq_len), as forward() receives it.

        Returns:
            (batch, num_channels, k).
        """
        assigner = self.assigner
        # Upstream permutes its (batch, seq_len, C) input back to this layout.
        x_emb = assigner.linear(x).reshape(-1, assigner.d_model)
        prob = torch.mm(assigner.l2norm(x_emb), assigner.l2norm(self.cluster_emb).t())
        return prob.reshape(-1, x.shape[1], assigner.n_cluster)

    @torch.no_grad()
    def fit_fixed_membership(self, inputs):
        """Fit and store the run's one fixed membership.

        Upstream pools the membership over whatever windows share a batch, in
        evaluation too, so a test forecast depends on its batch-mates (see the
        module docstring). This pools it once over the training split instead:
        the mean of window_prob over every training window, then upstream's
        sinkhorn with the assigner's epsilon -- the batch mean with the whole
        split as the batch. The sum is taken in float64 on the CPU, so the
        result does not depend on how the split is batched or ordered (beyond
        float32 rounding of each window's scores), and works on MPS.

        Args:
            inputs: iterable of (batch, num_channels, seq_len) model inputs that
                covers the training split once.

        Returns:
            fixed_prob, (num_channels, k), also stored on the module.
        """
        from forecasting.multivariate.utils.ccm_upstream import sinkhorn

        was_training = self.training
        self.eval()
        total, count = None, 0
        try:
            for x in inputs:
                prob = self.window_prob(x).to("cpu", torch.float64)
                total = prob.sum(dim=0) if total is None else total + prob.sum(dim=0)
                count += prob.shape[0]
        finally:
            self.train(was_training)
        if count == 0:
            raise ValueError("fit_fixed_membership got no training windows.")

        mean = (total / count).to(self.cluster_emb.device, self.cluster_emb.dtype)
        self.fixed_prob = sinkhorn(mean, epsilon=self.assigner.epsilon)
        return self.fixed_prob

    @contextlib.contextmanager
    def fixed_membership(self):
        """Forward with fixed_prob in place of the batch membership, for the block."""
        if self.fixed_prob is None:
            raise RuntimeError(
                "No fixed membership yet; call fit_fixed_membership() first."
            )
        previous = self.use_fixed_prob
        self.use_fixed_prob = True
        try:
            yield self.fixed_prob
        finally:
            self.use_fixed_prob = previous

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # A checkpoint carries fixed_prob only once it was fitted. Follow the
        # checkpoint either way, so a strict load takes old and new checkpoints
        # alike and a membership fitted for other weights never survives one.
        key = prefix + "fixed_prob"
        if key in state_dict:
            self.fixed_prob = torch.empty_like(
                state_dict[key], device=self.cluster_emb.device
            )
        else:
            self.fixed_prob = None
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @staticmethod
    def channel_similarity(x):
        """Upstream's Euclidean similarity kernel, over channels.

        ADAPTED (3) -- similarity_loss needs a [num_channels, num_channels]
        matrix: it computes membership.t() @ simMatrix @ membership with
        membership of shape [num_channels, k]. Upstream's
        Exp_CCM.get_similarity_matrix builds a *batch-wise* matrix instead --
        only the right shape in their M4 and stock setups, where channels ride
        the batch axis. On genuinely multivariate input the result comes out
        [bsz, bsz, seq_len] and torch.mm rejects it; their intended
        multivariate path loads temp_store/SimilarityMatrix_<data>.npy, which
        the release does not ship and whose loader is commented out. So this is
        the same kernel -- squared Euclidean distance, normalized by its max,
        exp(-5 * d / max) -- applied along the channel axis the loss actually
        indexes, with the batch axis averaged out.

        Args:
            x: (batch, num_channels, length).

        Returns:
            (num_channels, num_channels) similarity matrix.
        """
        diff = x.unsqueeze(2) - x.unsqueeze(1)  # (B, C, C, L)
        dist_squared = torch.sum(diff**2, dim=-1).mean(dim=0)  # (C, C)
        param = torch.max(dist_squared)
        return torch.exp(-5 * dist_squared / param.clamp_min(1e-12))

    def similarity_loss(self, x):
        """CCM's auxiliary similarity loss for the most recent forward.

        Ported verbatim from upstream's Exp_CCM.similarity_loss_batch (using
        the assigner's own concrete_bern): a concrete-Bernoulli relaxation of
        the cluster membership, a spectral-style term rewarding within-cluster
        similarity and penalizing across-cluster similarity, plus an entropy
        term that discourages an all-channels-in-one-cluster collapse.
        Weighted by cluster.beta in compute_aux_loss, and deliberately kept out
        of the reported loss and the early-stopping metric (see
        utils/models.py::compute_aux_loss).

        Args:
            x: the same (batch, num_channels, length) batch passed to forward().
        """
        prob = self.cluster_prob
        if prob is None:
            raise RuntimeError(
                "similarity_loss() called before forward(); cluster_prob is unset."
            )

        sim_matrix = self.channel_similarity(x)
        membership = self.assigner.concrete_bern(prob)  # [num_channels, k]

        temp_1 = torch.mm(membership.t(), sim_matrix)
        sas = torch.mm(temp_1, membership)
        _ss = 1 - torch.mm(membership, membership.t())
        loss = -torch.trace(sas) + torch.trace(torch.mm(_ss, sim_matrix)) + membership.shape[0]
        ent_loss = (-prob * torch.log(prob + 1e-15)).sum(dim=-1).mean()
        return loss + ent_loss
