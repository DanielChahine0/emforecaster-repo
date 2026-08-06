"""CCM (Channel Clustering Module) adapter.

Wraps the upstream CCM model wrappers from TimeSeriesCCM-main/ so they run
inside this repo's Experiment: our dataloader, our conformal recalibration, our
logging tree. The forecasting network itself is upstream's code, unmodified --
this file only translates between the two sides.

CCM replaces a model's fully-joint or channel-independent head with k
*learned, soft* channel clusters:

  1. Cluster_assigner projects each channel's input window to d_model, takes the
     cosine similarity against k trainable cluster embeddings, and sinkhorn-
     normalizes the result into prob[n_vars, k].
  2. Cluster_wise_linear runs k parallel Linear layers over every channel and
     mixes their outputs by prob -- a soft mixture, not a hard partition.
  3. The cluster embeddings are refreshed from the batch each training step via
     cross-attention, and an auxiliary similarity loss pulls the assignment
     toward the data's own channel-similarity structure.

Four things need adapting, each marked ADAPTED below: the cluster-embedding
state, tensor layout, the similarity matrix, and a shape bug in upstream's
cluster-embedding refresh that stops it after one training batch.
"""

import torch
import torch.nn as nn

from forecasting.multivariate.utils.ccm_upstream import UPSTREAM_MODELS


def _device_index(device):
    """CUDA ordinal for upstream's `cuda:{args.cuda}` device string.

    Upstream hardcodes `torch.device(f"cuda:{args.cuda}" if
    torch.cuda.is_available() else "cpu")` in every wrapper's __init__. That
    only decides where a handful of init-time tensors land; the Experiment moves
    the whole module with .to(device) afterwards, and the one tensor .to() would
    normally miss (cluster_emb) is a registered buffer here. So this just needs
    to name a valid device, not to be the single source of truth.
    """
    device = torch.device(device)
    if device.type == "cuda":
        return device.index if device.index is not None else 0
    return 0


def _backbone_dims(args):
    """(d_model, d_ff) for the CCM backbone, from this model's own config block.

    Most models size themselves from args.sl, but TimesNet has always read its
    own args.timesnet block here (see get_model in utils/models.py) and the job
    config sets only that one. Falling through to the args.sl defaults would
    silently build a 128-wide TimesNet against the multivariate grid's 16-wide
    baseline, making the cluster cell incomparable with the cell it exists to be
    compared against.
    """
    if args.exp.model_id == "TimesNet":
        return args.timesnet.d_model, args.timesnet.d_ff
    return args.sl.d_model, args.sl.d_ff


def build_ccm_args(args, device):
    """Upstream's flat argparse Namespace, filled from our nested Global config.

    Upstream reads one `args` object across all four wrappers, so unused keys
    (patch_len for DLinear, n_heads for TSMixer) are still required to exist.
    """
    d_model, d_ff = _backbone_dims(args)
    return type(
        "CCMArgs",
        (),
        {
            # Window and channels.
            "in_len": args.data.seq_len,
            "out_len": args.data.pred_len,
            # num_channels is set from the training split by the dataloader, so
            # the 27-vs-29 band difference across the Italy datasets needs no
            # per-dataset config edit.
            "data_dim": args.data.num_channels,
            # Upstream branches on `data in ("M4", "stock")` to fold channels
            # into the batch axis. Our dataset names never collide with those,
            # so this stays on the ordinary multivariate path.
            "data": args.data.dataset,
            "batch_size": args.sl.batch_size,
            # Clustering. "c" is upstream's clustered mode ("i" is channel-
            # independent, anything else is fully joint); the cluster grid only
            # ever runs "c", the other two are already covered by the
            # univariate and multivariate grids respectively.
            "n_cluster": args.cluster.k_clusters,
            "individual": "c",
            # Backbone hyperparameters, from the same job configs the
            # multivariate grid uses, so a cluster cell differs from its
            # multivariate counterpart only in the head.
            "d_model": d_model,
            "d_ff": d_ff,
            "n_heads": args.sl.num_heads,
            "n_layers": args.sl.num_enc_layers,
            "dropout": args.sl.dropout,
            "attn_dropout": args.sl.attn_dropout,
            # PatchTST-only.
            "patch_len": args.data.patch_dim,
            "stride": args.data.patch_stride,
            "padding_patch": "end",
            "max_seq_len": 1024,
            "pre_norm": False,
            "pretrain_head": False,
            # TimesNet-only. Its DataEmbedding builds a TemporalEmbedding sized
            # by freq, but forward() calls it with x_mark=None, so the calendar
            # branch never runs and the value is inert.
            "freq": "h",
            "cuda": _device_index(device),
        },
    )()


class CCMForecaster(nn.Module):
    """One upstream CCM model, adapted to this repo's calling convention."""

    def __init__(self, args, device=None):
        super().__init__()

        if device is None:
            # Matches upstream's own choice; only decides where a few init-time
            # tensors land, since Experiment.init_model() calls .to(device) next.
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_id = args.exp.model_id
        if model_id not in UPSTREAM_MODELS:
            raise ValueError(
                f"{model_id} has no upstream CCM wrapper. The cluster grid "
                f"covers exactly the models CCM implements: "
                f"{', '.join(sorted(UPSTREAM_MODELS))}."
            )
        if args.cluster.k_clusters < 1:
            raise ValueError(
                f"cluster.k_clusters must be at least 1, got {args.cluster.k_clusters}"
            )

        self.model_id = model_id
        self.beta = args.cluster.beta
        self.k_clusters = args.cluster.k_clusters
        self.pred_len = args.data.pred_len

        ccm_args = build_ccm_args(args, device)
        self.net = UPSTREAM_MODELS[model_id](ccm_args)

        # ADAPTED (1/4) -- cluster-embedding state.
        #
        # Upstream keeps cluster_emb as a bare tensor attribute and, on every
        # training forward, rebinds it to a freshly constructed
        # nn.Parameter(..., requires_grad=True). Two consequences we cannot live
        # with: a bare attribute is invisible to .to(device) and to state_dict
        # (so it would stay on the init device and never reach supervised.pth),
        # and a parameter that appears mid-training makes the module's key set
        # depend on how many steps have run.
        #
        # Holding it as a registered buffer instead fixes both. It is not a
        # behaviour change: upstream's cluster_emb starts life with
        # requires_grad=False, is always overwritten by a detached value, and is
        # never handed to the optimizer -- it is forward-updated state, not a
        # learned parameter, so requires_grad only ever populated a .grad nobody
        # read. forward() feeds this buffer in and syncs upstream's result back.
        self.register_buffer(
            "cluster_emb", self.net.cluster_emb.detach().clone(), persistent=True
        )

        # Upstream's device guess is `cuda:0 if torch.cuda.is_available() else
        # cpu`, taken at construction and ignoring what the caller asked for. So
        # on a CUDA box a device="cpu" build comes back with cluster_emb on the
        # GPU and every nn.Linear on the CPU. Move the whole module onto the
        # requested device to settle it here; Experiment.init_model() then calls
        # .to(device) again, which is a no-op.
        self.to(device)

        # Set by the last forward; read by similarity_loss().
        self.cluster_prob = None

    def _sync_cluster_emb(self):
        """Pull upstream's post-forward cluster_emb back into our buffer.

        ADAPTED (4/4) -- upstream shape bug in the cluster-embedding refresh.

        Cluster_assigner consumes cluster_emb as [n_cluster, d_model] but
        returns the cross-attention result, which CrossAttention documents (and
        produces) as [batch, n_cluster, d_model]. Upstream writes that 3-D
        tensor straight back into self.cluster_emb on every if_update=True
        forward, so the *next* training batch hits
        `self.l2norm(cluster_emb).t()` with a 3-D tensor and dies:

            RuntimeError: t() expects a tensor with <= 2 dimensions, but self is 3D

        That happens on training batch two, in the released code, with no
        involvement from this repo -- Exp_CCM.train() cannot finish an epoch as
        shipped. Averaging the per-example embeddings back down to one shared
        table is the fix that matches the assigner's own convention: the line
        above it does exactly this to the assignment probabilities
        (`prob_avg = torch.mean(prob, dim=0)`), so the embeddings that those
        probabilities are computed against are pooled the same way.
        """
        emb = self.net.__dict__.get("cluster_emb")
        if emb is None:
            # Rebound to a Parameter by an if_update=True forward. Take the
            # value and unregister it, so the module's state_dict keys stay
            # fixed across the run.
            emb = self.net._parameters.pop("cluster_emb", None)
            if emb is None:
                return
        elif emb is self.cluster_emb:
            # An if_update=False forward left our own buffer in place.
            return
        emb = emb.detach()
        if emb.dim() == 3:
            emb = emb.mean(dim=0)
        with torch.no_grad():
            self.cluster_emb.copy_(emb)

    def forward(self, x):
        # ADAPTED (2/4) -- tensor layout. This repo passes and expects
        # (batch, num_channels, length); upstream works in (batch, length,
        # num_channels) throughout.
        x = x.permute(0, 2, 1)

        # Feed the buffer in as the plain attribute upstream expects. Assigning
        # a non-Parameter tensor lands it in net.__dict__, exactly where
        # upstream's own __init__ put it.
        self.net.cluster_emb = self.cluster_emb

        # if_update drives the cross-attention refresh of the cluster
        # embeddings, which upstream does on training batches only.
        out = self.net(x, if_update=self.training)

        self._sync_cluster_emb()
        self.cluster_prob = self.net.cluster_prob

        return out.permute(0, 2, 1)

    @staticmethod
    def channel_similarity(x):
        """Upstream's Euclidean similarity kernel, over channels.

        ADAPTED (3/4) -- similarity matrix.

        similarity_loss_batch needs a [n_vars, n_vars] matrix: it computes
        membership.t() @ simMatrix @ membership with membership of shape
        [n_vars, n_cluster]. Upstream's Exp_CCM.get_similarity_matrix builds a
        *batch-wise* matrix instead -- it does batch_x.squeeze(-1) then takes
        pairwise distances along dim 0, giving [bsz, bsz]. That is only the
        right shape in their M4 and stock setups, where n_vars IS the batch axis
        (see `args.batch_size if args.data in ("M4", "stock") else args.data_dim`
        in every wrapper). On genuinely multivariate input the squeeze is a
        no-op, the result comes out [bsz, bsz, in_len], and torch.mm rejects it.
        Their intended multivariate path loads a precomputed [n_vars, n_vars]
        matrix from temp_store/SimilarityMatrix_<data>.npy, but that file is not
        in the release and the code that reads it is commented out in
        exp/exp_ccm.py.

        So this is the same kernel -- squared Euclidean distance, normalized by
        its max, exp(-5 * d / max) -- applied along the channel axis the loss
        actually indexes, with the batch axis averaged out.

        Args:
            x: (batch, length, num_channels), as upstream's forward sees it.

        Returns:
            (num_channels, num_channels) similarity matrix.
        """
        # (B, L, C) -> (B, C, L): one window vector per channel per example.
        sample = x.permute(0, 2, 1)
        diff = sample.unsqueeze(2) - sample.unsqueeze(1)  # (B, C, C, L)
        dist_squared = torch.sum(diff**2, dim=-1).mean(dim=0)  # (C, C)
        param = torch.max(dist_squared)
        return torch.exp(-5 * dist_squared / param.clamp_min(1e-12))

    def similarity_loss(self, x):
        """CCM's auxiliary similarity loss for the most recent forward.

        Ported verbatim from Exp_CCM.similarity_loss_batch: a concrete-Bernoulli
        relaxation of the cluster membership, a spectral-style term rewarding
        within-cluster similarity and penalizing across-cluster similarity, plus
        an entropy term that discourages an all-channels-in-one-cluster collapse.

        Args:
            x: the same (batch, num_channels, length) batch passed to forward().
        """
        prob = self.cluster_prob
        if prob is None:
            raise RuntimeError(
                "similarity_loss() called before forward(); cluster_prob is unset."
            )

        sim_matrix = self.channel_similarity(x.permute(0, 2, 1))
        membership = self._concrete_bern(prob)  # [n_vars, n_cluster]

        temp_1 = torch.mm(membership.t(), sim_matrix)
        sas = torch.mm(temp_1, membership)
        _ss = 1 - torch.mm(membership, membership.t())
        loss = -torch.trace(sas) + torch.trace(torch.mm(_ss, sim_matrix)) + membership.shape[0]
        ent_loss = (-prob * torch.log(prob + 1e-15)).sum(dim=-1).mean()
        return loss + ent_loss

    @staticmethod
    def _concrete_bern(prob, temp=0.07):
        """Concrete (Gumbel-sigmoid) relaxation, verbatim from Exp_CCM."""
        random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
        return ((prob + random_noise) / temp).sigmoid()
