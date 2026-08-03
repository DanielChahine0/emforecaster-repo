import unittest
from typing import List, Optional

import torch
import torch.nn as nn
from forecasting.multivariate.layers.patchtst.enc_block import _MultiheadAttention
from forecasting.multivariate.layers.patchtst.revin import RevIN
from forecasting.multivariate.layers.clustered_head import ClusteredHead


class MLP(nn.Module):
    def __init__(self, seq_len=20, pred_len=20, d_model=32):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(seq_len, d_model)
        self.fc2 = nn.Linear(d_model, pred_len)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x.squeeze()


class CNN(nn.Module):
    def __init__(self, seq_len=20, num_channels=1, pred_len=20, num_kernels=32):
        super(CNN, self).__init__()
        kernel_size = seq_len
        self.conv1 = nn.Conv1d(
            in_channels=num_channels, out_channels=num_kernels, kernel_size=kernel_size
        )
        self.fc = nn.Linear(num_kernels, pred_len)
        self.act = nn.Tanh()

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


class LSTM(nn.Module):
    def __init__(self, seq_len=20, pred_len=20, d_model=32):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(seq_len, d_model)
        self.fc = nn.Linear(d_model, pred_len)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x.squeeze()


class EncoderBlock(nn.Module):
    def __init__(self, seq_len=20, d_ff=2, d_model=1, num_heads=2, ff_dropout=0.0):
        super(EncoderBlock, self).__init__()
        self.attn = _MultiheadAttention(num_heads, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Tanh(),
            nn.Dropout(ff_dropout),
            nn.Linear(d_ff, d_model),
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            fc_out: Output of the transformer block, a tensor of shape (batch_size, seq_len, d_model).
        """

        # Multihead Attention -> Add & Norm
        attn_out, _ = self.attn(x, x, x)
        attn_norm = self.norm(
            attn_out + x
        )  # Treat the input as the query, key and value for MHA.

        # Feedforward layer -> Add & Norm
        fc_out = self.ff(attn_norm)
        fc_norm = self.norm(fc_out + attn_out)

        return fc_norm


class Transformer(nn.Module):
    def __init__(
        self,
        seq_len=20,
        pred_len=20,
        d_model=32,
        num_heads=2,
        num_enc_layers=2,
        revin=False,
        revin_affine=True,
        eps_revin=1e-5,
        cluster_assignment: Optional[List[int]] = None,
        k_clusters: int = 1,
    ):
        super(Transformer, self).__init__()
        self.enc = nn.ModuleList(
            [
                EncoderBlock(seq_len, pred_len, d_model, num_heads)
                for _ in range(num_enc_layers)
            ]
        )
        self.fc1 = nn.Linear(1, d_model)
        self.act = nn.Tanh()
        self.conv1 = nn.Conv1d(in_channels=seq_len, out_channels=seq_len, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=seq_len, out_channels=seq_len, kernel_size=1)

        self.clustered = cluster_assignment is not None
        if self.clustered:
            self.head = ClusteredHead(cluster_assignment, k_clusters, seq_len, pred_len)
        else:
            self.fc2 = nn.Linear(seq_len, pred_len)

        # RevIN parity with every other model in the benchmark: without
        # per-window renormalization this tanh-bounded backbone cannot track the
        # non-stationary level of the z-scored bands and collapses to the
        # train-mean forecast at long horizons (test MSE == climatology on
        # ED2022/UH2022 at O336). Applied in the folded (B*C, seq_len) space, so
        # the mean/std are per window per channel while the single affine pair is
        # shared across channels -- matching the channel-independent backbone and
        # keeping the layer valid for any channel count.
        self.revin = RevIN(1, eps=eps_revin, affine=revin_affine) if revin else None

    def forward(self, x):
        # Channel-independent multivariate I/O: fold the channel dim into the
        # batch so the univariate backbone below runs once per channel, then
        # unfold back to (B, C, pred_len). Mirrors the strategy EMForecaster and
        # PatchTST use; for C == 1 this reduces to the original univariate path.
        B, C, L = x.shape
        x = x.reshape(B * C, L)  # (B, C, seq_len) -> (B*C, seq_len)
        if self.revin is not None:
            x = self.revin(x, "norm").reshape(B * C, L)
        x = x.unsqueeze(-1)  # (B*C, seq_len) -> (B*C, seq_len, 1)
        x = self.fc1(
            x
        )  # (B*C, seq_len, 1) -> (B*C, seq_len, d_model)

        # Transformer
        for enc_layer in self.enc:
            x = enc_layer(x)

        # Convolutional layers
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)  # (B*C, seq_len, 1) for d_model == 2

        if self.clustered:
            # Unfold the channels back out BEFORE the head: the clustered head
            # needs to see all channels in a cluster simultaneously, unlike
            # fc2 which runs on the folded (B*C, seq_len) tensor.
            x = x.squeeze(-1).reshape(B, C, L)  # (B, C, seq_len)
            x = self.head(x)  # (B, C, pred_len)
            if self.revin is not None:
                x = x.reshape(B * C, -1)
                x = self.revin(x, "denorm").reshape(B, C, -1)
            return x
        else:
            x = self.fc2(x.squeeze(-1))  # (B*C, seq_len) -> (B*C, pred_len)
            if self.revin is not None:
                x = self.revin(x, "denorm").reshape(B * C, -1)
            return x.reshape(B, C, -1)  # (B*C, pred_len) -> (B, C, pred_len)


if __name__ == "__main__":
    mlp = MLP(seq_len=20, pred_len=20, d_model=32)
    cnn = CNN(seq_len=20, pred_len=20, num_kernels=32)
    lstm = LSTM(seq_len=20, pred_len=20, d_model=32)
    transformer = Transformer(
        seq_len=20, pred_len=20, d_model=2, num_heads=2, num_enc_layers=2
    )

    x = torch.randn(32, 1, 20)  # (batch_size, num_channels, seq_len)

    mlp_output = mlp(x)
    cnn_output = cnn(x)
    lstm_output = lstm(x)
    transformer_output = transformer(x)
    print(f"MLP output shape: {mlp_output.shape}")
    print(f"CNN output shape: {cnn_output.shape}")
    print(f"LSTM output shape: {lstm_output.shape}")
    print(f"Transformer output shape: {transformer_output.shape}")
