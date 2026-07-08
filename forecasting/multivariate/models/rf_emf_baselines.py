import unittest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from forecasting.multivariate.layers.patchtst.enc_block import _MultiheadAttention


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
        self, seq_len=20, pred_len=20, d_model=32, num_heads=2, num_enc_layers=2
    ):
        super(Transformer, self).__init__()
        self.enc = nn.ModuleList(
            [
                EncoderBlock(seq_len, pred_len, d_model, num_heads)
                for _ in range(num_enc_layers)
            ]
        )
        self.fc1 = nn.Linear(1, d_model)
        self.fc2 = nn.Linear(seq_len, pred_len)
        self.act = nn.Tanh()
        self.conv1 = nn.Conv1d(in_channels=seq_len, out_channels=seq_len, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=seq_len, out_channels=seq_len, kernel_size=1)

        # Memory cap for the folded-batch attention. This backbone runs full-length
        # self-attention, which materializes a (rows, num_heads, seq_len, seq_len)
        # score tensor whose size grows with seq_len**2. In the multivariate grid
        # the folded batch is rows = B*C (e.g. 128*27 = 3456), so the long-context
        # cells (seq_len=336) need ~15 GB and silently spill to system RAM over
        # PCIe on a 12 GB GPU (~8x slower). We therefore process the folded batch
        # in row-chunks no larger than (attn_row_budget / seq_len**2) rows. The
        # backbone is channel-independent (no op mixes rows: attention/FF/LayerNorm/
        # conv all act within a single row, dropout is 0), so chunking is
        # numerically identical to a single pass -- results are unchanged; only the
        # peak memory is bounded. The budget keeps the common seq_len<=192 cells a
        # single chunk (unchanged path) and splits only the oversized ones.
        self.attn_row_budget = 128 * 1024 * 1024  # ~5 GB peak/chunk at ~40 B per row*L^2

    def _backbone(self, x):
        # x: (rows, seq_len, 1) -> (rows, pred_len). Exactly the original per-row
        # compute; factored out so it can be run per chunk and gradient-checkpointed.
        x = self.fc1(x)  # (rows, seq_len, 1) -> (rows, seq_len, d_model)

        # Transformer
        for enc_layer in self.enc:
            x = enc_layer(x)

        # Convolutional layers
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)  # (rows, seq_len, 1) for d_model == 2

        return self.fc2(x.squeeze(-1))  # (rows, seq_len) -> (rows, pred_len)

    def forward(self, x):
        # Channel-independent multivariate I/O: fold the channel dim into the
        # batch so the univariate backbone below runs once per channel, then
        # unfold back to (B, C, pred_len). Mirrors the strategy EMForecaster and
        # PatchTST use; for C == 1 this reduces to the original univariate path.
        B, C, L = x.shape
        x = x.reshape(B * C, L)  # (B, C, seq_len) -> (B*C, seq_len)
        x = x.unsqueeze(-1)  # (B*C, seq_len) -> (B*C, seq_len, 1)

        # Row-chunk the folded batch so the (rows, heads, L, L) attention scores fit
        # in VRAM. When the whole folded batch already fits (rows <= chunk, e.g. the
        # seq_len<=192 cells) we take the ORIGINAL single-pass path -- no chunking,
        # no checkpoint -- so those runs are byte-for-byte and speed-for-speed
        # unchanged. Only oversized cells (seq_len=336) split into equal passes.
        rows = x.shape[0]
        chunk = max(1, self.attn_row_budget // (L * L))
        if rows <= chunk:
            out = self._backbone(x)  # (B*C, pred_len) -- unchanged path
        else:
            # Gradient-checkpoint each chunk during training so the large attention
            # activations are recomputed in backward rather than all held at once
            # (that is what bounds the training peak). No dropout here, so outputs
            # are bit-identical and gradients match to fp summation order.
            use_ckpt = self.training and torch.is_grad_enabled()
            outs = []
            for s in range(0, rows, chunk):
                xc = x[s : s + chunk]
                outs.append(
                    checkpoint(self._backbone, xc, use_reentrant=False)
                    if use_ckpt
                    else self._backbone(xc)
                )
            out = torch.cat(outs, dim=0)

        return out.reshape(B, C, -1)  # (B*C, pred_len) -> (B, C, pred_len)


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
