"""Import bridge to the vendored CCM reference implementation.

CCM ("From Similarity to Superiority: Channel Clustering for Time Series
Forecasting", NeurIPS 2024) ships as a flat script tree in TimeSeriesCCM-main/,
whose modules import each other by top-level absolute name -- `from models.layers
import *`, `from models.patch_layer import *`. The directory name also carries a
hyphen, so it can never be a package itself.

Rather than copy the classes in (a fork that silently drifts from the paper's
code), this module puts the CCM root on sys.path once and re-exports the pieces
the cluster grid needs. Everything below therefore executes upstream's code
verbatim; the only repo-side code is forecasting/multivariate/models/ccm.py,
which adapts our config and tensor layout to their constructors.

Putting the CCM root on sys.path exposes `models`, `utils`, `exp` and `data` as
importable top-level names. That is safe here only because nothing in this repo
imports those bare names -- every internal import is fully qualified from
`forecasting.`. The append (rather than insert) keeps repo-local modules winning
any future tie.

The upstream tree also contains exp/, data/ and utils/ modules that pull in wandb
and torchinfo. None of those are imported here: only the four model wrappers and
the two cluster layers, which need nothing beyond torch, numpy and einops.
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
        f"grid runs upstream's code directly; restore the TimeSeriesCCM-main/ "
        f"directory (it is tracked in git) before running --setup cluster."
    )

if CCM_ROOT not in sys.path:
    sys.path.append(CCM_ROOT)

# Imported after the sys.path append, so these resolve inside TimeSeriesCCM-main/.
from models.Dlinear import DLinearC  # noqa: E402
from models.layers import Cluster_assigner  # noqa: E402
from models.patch_layer import Cluster_wise_linear  # noqa: E402
from models.patchtst import PatchTSTC  # noqa: E402
from models.timesnet import TimesNetC  # noqa: E402
from models.tsmixer import TSMixerC  # noqa: E402

# Our model_id (as it appears in the job configs and in the logs/ path) -> the
# upstream wrapper that implements it with clustering threaded through its
# layers. These four are exactly the models CCM provides wrappers for; a model
# outside this map has no upstream counterpart and cannot join the cluster grid
# without an adaptation that would no longer be the paper's code.
UPSTREAM_MODELS = {
    "DLinear": DLinearC,
    "PatchTST": PatchTSTC,
    "TSMixer": TSMixerC,
    "TimesNet": TimesNetC,
}

__all__ = [
    "CCM_ROOT",
    "UPSTREAM_MODELS",
    "Cluster_assigner",
    "Cluster_wise_linear",
    "DLinearC",
    "PatchTSTC",
    "TSMixerC",
    "TimesNetC",
]
