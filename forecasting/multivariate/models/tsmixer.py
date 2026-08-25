import torch
import torch.nn as nn
from forecasting.multivariate.utils.weight_init import xavier_init
from forecasting.multivariate.layers.patchtst.revin import RevIN

# Taken from: https://github.com/thuml/emforecaster-Library/blob/main/models/TSMixer.py


class ResBlock(nn.Module):
    def __init__(
        self,
        seq_len,
        d_model,
        dropout,
        num_channels,
        n_clusters=None,
    ):
        super(ResBlock, self).__init__()

        # CCM cluster mode replaces the per-channel temporal linears with
        # upstream's cluster-wise layer, run verbatim from the vendored tree
        # (upstream TSMixerC does the same to its MLPTime); the channel-mixing
        # MLP stays plain there too, so it stays plain here. The layer's
        # trailing device arg is unused by the released class.
        self.n_clusters = n_clusters
        if n_clusters is not None:
            from forecasting.multivariate.utils.ccm_upstream import Cluster_wise_linear

            self.temporal_in = Cluster_wise_linear(
                n_clusters, num_channels, seq_len, d_model, None
            )
            self.temporal_out = Cluster_wise_linear(
                n_clusters, num_channels, d_model, seq_len, None
            )
            self.temporal_act = nn.GELU()
            self.temporal_dropout = nn.Dropout(dropout)
        else:
            self.temporal = nn.Sequential(
                nn.Linear(seq_len, d_model),
                nn.GELU(),
                nn.Linear(d_model, seq_len),
                nn.Dropout(dropout),
            )

        self.channel = nn.Sequential(
            nn.Linear(num_channels, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_channels),
            nn.Dropout(dropout),
        )

    def forward(self, x, cluster_prob=None):
        """
        Args:
            x (torch.Tensor): Shape (batch_size, seq_len, num_vars)

        Returns:
            torch.Tensor: Shape (batch_size, seq_len, num_vars)
        """
        if self.n_clusters is not None:
            t = self.temporal_in(x.transpose(1, 2), cluster_prob)
            t = self.temporal_out(self.temporal_act(t), cluster_prob)
            x = x + self.temporal_dropout(t).transpose(1, 2)
        else:
            x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)

        return x


class TSMixer(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        num_enc_layers,
        d_model,
        num_channels,
        dropout=0.0,
        revin=True,
        revin_affine=True,
        revout=False,
        eps_revin=1e-5,
        return_head=True,
        n_clusters=None,
    ):
        super(TSMixer, self).__init__()

        # Parameters
        self.num_enc_layers = num_enc_layers
        self.eps_revin = eps_revin
        self.revin_affine = revin_affine
        self.num_channels = num_channels
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.return_head = return_head
        self.n_clusters = n_clusters

        # Layers
        self.backbone = nn.ModuleList(
            [
                ResBlock(seq_len, d_model, dropout, num_channels, n_clusters)
                for _ in range(num_enc_layers)
            ]
        )

        if n_clusters is not None:
            from forecasting.multivariate.utils.ccm_upstream import Cluster_wise_linear

            self.head = Cluster_wise_linear(
                n_clusters, num_channels, seq_len, pred_len, None
            )
        else:
            self.head = nn.Linear(seq_len, pred_len)

        # Initialize layers
        if revin:
            self._init_revin(revout, revin_affine)
        else:
            self._revin = None
            self.revout = None

        self.apply(xavier_init)

    def _init_revin(self, revout: bool, revin_affine: bool):
        self._revin = True
        self.revout = revout
        self.revin_affine = revin_affine
        self.revin = RevIN(self.num_channels, self.eps_revin, self.revin_affine)

    def forward(self, x, cluster_prob=None):
        if self.n_clusters is not None and cluster_prob is None:
            raise ValueError(
                "This TSMixer was built with n_clusters; forward() needs the "
                "ClusterAssigner's membership (see models/ccm.py)."
            )

        # RevIN
        if self._revin:
            x = self.revin(x, mode="norm")

        x = x.permute(
            0, 2, 1
        )  # (batch_size, num_channels, seq_len) => (batch_size, seq_len, num_channels)
        for i in range(self.num_enc_layers):
            x = self.backbone[i](x, cluster_prob)

        if self.return_head:
            if self.n_clusters is not None:
                out = self.head(x.transpose(1, 2), cluster_prob)
            else:
                out = self.head(x.transpose(1, 2))
        else:
            out = x.transpose(1, 2)

        # RevOUT
        if self.revout:
            out = self.revin(out, mode="denorm")

        return out


# Test
if __name__ == "__main__":
    batch_size = 32
    seq_len = 512
    pred_len = 96
    num_channels = 7
    num_enc_layers = 3
    d_model = 16
    dropout = 0.1
    revin = True
    revin_affine = True
    revout = True
    eps_revin = 1e-5

    x = torch.rand(batch_size, num_channels, seq_len)

    model = TSMixer(
        seq_len=seq_len,
        pred_len=pred_len,
        num_enc_layers=num_enc_layers,
        d_model=d_model,
        dropout=dropout,
        num_channels=num_channels,
        revin=revin,
        revin_affine=revin_affine,
        revout=revout,
        eps_revin=eps_revin,
    )

    y = model(x)
    print(f"x: {x.shape} => y: {y.shape}")
