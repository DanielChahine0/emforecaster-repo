"""Import bridge to the vendored CCM reference implementation -- layers only.

CCM ("From Similarity to Superiority: Channel Clustering for Time Series
Forecasting", NeurIPS 2024) ships as a flat script tree in TimeSeriesCCM-main/,
whose modules import each other by top-level absolute name -- `from models.layers
import *`, `from models.patch_layer import *`. The directory name also carries a
hyphen, so it can never be a package itself.

Rather than copy the classes in (a fork that silently drifts from the paper's
code), this module puts the CCM root on sys.path once and re-exports the pieces
the cluster grid needs. Everything imported here therefore executes upstream's
code verbatim.

What is exported changed on 2026-08-25. The bridge used to hand out upstream's
whole model wrappers (DLinearC, PatchTSTC, TSMixerC, TimesNetC), so a cluster
cell ran an entirely different network implementation from the multivariate cell
it is compared against -- different init, RevIN, decomposition kernel. Now only
the CLUSTERING layers cross the bridge:

  Cluster_assigner     the learned soft channel-to-cluster assignment (cosine
                       similarity against k trainable cluster embeddings,
                       sinkhorn-normalized) plus the cross-attention refresh of
                       those embeddings.
  Cluster_wise_linear  k parallel Linear layers over every channel, mixed by
                       the assignment -- the drop-in for a per-channel temporal
                       nn.Linear.
  sinkhorn             upstream's row normalization (a temperature softmax; the
                       released code runs zero Sinkhorn iterations despite the
                       name). Exported for the fixed membership (models/ccm.py),
                       which normalizes its train-split mean with it, and tests.

They are threaded through THIS repo's own models (see models/dlinear.py,
models/tsmixer.py, layers/patchtst/backbone.py, models/emforecaster.py,
models/rf_emf_baselines.py, orchestrated by models/ccm.py), so the clustering
mechanism is the paper's code while the forecasting network, its
hyperparameters and its init are exactly the multivariate cell's. The upstream
model wrappers are deliberately not imported anywhere any more.

Putting the CCM root on sys.path exposes `models`, `utils`, `exp` and `data` as
importable top-level names. That is safe here only because nothing in this repo
imports those bare names -- every internal import is fully qualified from
`forecasting.`. The append (rather than insert) keeps repo-local modules winning
any future tie.

The upstream tree also contains exp/, data/ and utils/ modules that pull in wandb
and torchinfo. None of those are imported here: the two cluster layers need
nothing beyond torch, numpy and einops.

This module is imported lazily, from inside the cluster branches of the model
constructors and CCMForecaster, so a missing vendored tree only breaks cluster
cells -- with the error below -- and never a multivariate or per-channel run.
"""

import os
import sys

CCM_ROOT = os.path.join(
    os.path.dirname(  # repo root
        os.path.dirname(  # forecasting/
            os.path.dirname(  # forecasting/multivariate/
                os.path.dirname(os.path.abspath(__file__))  # .../utils/
            )
        )
    ),
    "TimeSeriesCCM-main",
)

if not os.path.isdir(CCM_ROOT):
    raise ImportError(
        f"CCM reference implementation not found at {CCM_ROOT}. The cluster "
        f"grid runs upstream's clustering layers directly; restore the "
        f"TimeSeriesCCM-main/ directory (it is tracked in git) before running "
        f"--setup cluster."
    )

if CCM_ROOT not in sys.path:
    sys.path.append(CCM_ROOT)

# Imported after the sys.path append, so these resolve inside TimeSeriesCCM-main/.
from models.layers import Cluster_assigner, sinkhorn  # noqa: E402
from models.patch_layer import Cluster_wise_linear  # noqa: E402

__all__ = [
    "CCM_ROOT",
    "Cluster_assigner",
    "Cluster_wise_linear",
    "sinkhorn",
]
