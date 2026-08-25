import torch
import torch.nn as nn
import torch.nn.functional as F
from forecasting.multivariate.utils.utils import *
from forecasting.multivariate.layers.patchtst import *
from forecasting.multivariate.layers.patchtst.enc_block import EncoderBlock
from typing import Optional

class PatchTSTBackbone(nn.Module):
    def __init__(
        self,
        num_enc_layers,
        d_model,
        d_ff,
        num_heads,
        num_channels,
        num_patches,
        pred_len,
        attn_dropout=0.0,
        ff_dropout=0.0,
        pred_dropout=0.0,
        batch_first=True,
        norm_mode="batch1d",
        return_head=True,
        head_type="linear",
        channel_independent=False,
        target_channels: Optional[list] = None,
        n_clusters=None,
    ):
        super(PatchTSTBackbone, self).__init__()

        # Parameters
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.d_model = d_model
        self.return_head = return_head
        self.pred_len = pred_len
        self.target_channels = target_channels
        self.n_clusters = n_clusters
        if n_clusters is not None and (head_type != "linear" or target_channels):
            raise ValueError(
                "n_clusters only replaces the plain linear head; cluster cells "
                "run head_type='linear' with no target_channels."
            )

        # Encoder
        self.enc = nn.Sequential(
            *(
                EncoderBlock(
                    d_model,
                    d_ff,
                    num_heads,
                    num_channels,
                    num_patches,
                    attn_dropout,
                    ff_dropout,
                    batch_first,
                    norm_mode,
                )
                for i in range(num_enc_layers)
            )
        )

        # Prediction head
        self.flatten = nn.Flatten(start_dim=-2)
        num_output_channels = (
            len(target_channels) if target_channels is not None else num_channels
        )
        if n_clusters is not None:
            # The channel-independent head with k cluster-wise copies routed by
            # the soft membership -- upstream's own Cluster_wise_linear, run
            # verbatim from the vendored tree (lazy import so only cluster
            # cells touch it; its trailing device arg is unused).
            from forecasting.multivariate.utils.ccm_upstream import Cluster_wise_linear

            self.head = Cluster_wise_linear(
                n_clusters, num_channels, num_patches * d_model, pred_len, None
            )
        elif head_type == "linear":
            if channel_independent:
                self.head = nn.Linear(num_patches * d_model, pred_len)
            elif target_channels is not None or not channel_independent:
                self.head = nn.Sequential(
                    Reshape(-1, num_output_channels * num_patches * d_model),
                    nn.Linear(
                        num_output_channels * num_patches * d_model,
                        num_output_channels * pred_len,
                    ),
                    Reshape(-1, num_output_channels, pred_len),
                )
        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(num_patches * d_model, num_patches * d_model),
                nn.GELU(),
                nn.Dropout(pred_dropout),
                nn.Linear(num_patches * d_model, pred_len),
            )
        else:
            raise ValueError(f"Invalid head type: {head_type}")

    def forward(
        self, x: torch.Tensor, y: torch.Tensor = None, ch_ids: torch.Tensor = None,
        cluster_prob: torch.Tensor = None,
    ) -> torch.Tensor:

        # Encoding
        batch_size = x.shape[0]
        x = x.view(
            batch_size * self.num_channels, self.num_patches, -1
        )  # (batch_size * num_channels, num_patches, d_model)
        x = self.enc(x)  # (batch_size * num_channels, num_patches, d_model)

        if self.return_head:
            x = x.reshape(
                batch_size, self.num_channels, self.num_patches * self.d_model
            )

            if self.target_channels is not None:
                x = x[:, self.target_channels, :]

            if self.n_clusters is not None:
                out = self.head(x, cluster_prob)
            else:
                out = self.head(x)
        else:
            out = x.view(batch_size, self.num_channels, self.num_patches, -1)

        return out
