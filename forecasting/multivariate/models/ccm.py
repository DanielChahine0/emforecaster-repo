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
multivariate branch plus n_clusters, then wraps it here. Everything else about
the run -- data, optimizer, conformal recalibration, logging -- is unchanged.
"""

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

    def forward(self, x):
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
