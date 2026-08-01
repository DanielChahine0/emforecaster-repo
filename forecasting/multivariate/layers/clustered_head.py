from typing import List

import torch
import torch.nn as nn


class ClusteredHead(nn.Module):
    """
    One Linear head per channel cluster. Each head flattens and mixes only the
    channels assigned to its own cluster (k=1 is equivalent to a fully joint
    head over all channels; k=num_channels gives each channel its own
    independently-trained head, unlike a weight-tied per-channel head).

    Input: (B, C, F) in the original channel order.
    Output: (B, C, pred_len), restored to the same original channel order.
    """

    def __init__(
        self,
        cluster_assignment: List[int],
        k_clusters: int,
        in_features: int,
        pred_len: int,
    ):
        super().__init__()

        if len(cluster_assignment) == 0:
            raise ValueError("cluster_assignment must not be empty")
        if not set(cluster_assignment) <= set(range(k_clusters)):
            raise ValueError(
                f"cluster_assignment contains ids outside range(k_clusters={k_clusters})"
            )

        groups = [[] for _ in range(k_clusters)]
        for channel_idx, cluster_id in enumerate(cluster_assignment):
            groups[cluster_id].append(channel_idx)

        if any(len(group) == 0 for group in groups):
            raise ValueError(
                "empty cluster in cluster_assignment: reduce k_clusters or "
                "recompute the clustering"
            )

        perm = [idx for group in groups for idx in group]
        self.register_buffer("_perm", torch.tensor(perm, dtype=torch.long))
        self.register_buffer("_inverse_perm", torch.argsort(self._perm))

        self._group_sizes = [len(group) for group in groups]
        self.linears = nn.ModuleList(
            [
                nn.Linear(size * in_features, size * pred_len)
                for size in self._group_sizes
            ]
        )
        self.pred_len = pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F)
        batch_size, num_features = x.shape[0], x.shape[-1]
        x = x[:, self._perm, :]

        outputs = []
        offset = 0
        for size, linear in zip(self._group_sizes, self.linears):
            group = x[:, offset : offset + size, :].reshape(
                batch_size, size * num_features
            )
            outputs.append(linear(group).reshape(batch_size, size, self.pred_len))
            offset += size

        out = torch.cat(outputs, dim=1)  # (B, C, pred_len), in grouped order
        return out[:, self._inverse_perm, :]  # restore original channel order
