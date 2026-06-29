import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, norm_mode="layer"):
        super(Linear, self).__init__()

        # Normalization (operates over the input feature dimension, i.e. seq_len)
        if norm_mode == "layer":
            self.norm = nn.LayerNorm(in_features)
        elif norm_mode == "batch1d":
            self.norm = nn.BatchNorm1d(in_features)
        else:
            self.norm = nn.Identity()

        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.norm(x.squeeze())
        out = self.linear(x)
        return out
