from typing import List

import numpy as np
from sklearn.cluster import KMeans


def compute_channel_clusters(
    train_data: np.ndarray,
    k: int,
    method: str = "kmeans",
    seed: int = 69421,
) -> List[int]:
    """
    Assign each channel to one of k clusters using data-driven clustering over
    the training split's channel correlation structure.

    Args:
        train_data (np.ndarray): Training data of shape (num_channels, T).
        k (int): Number of clusters.
        method (str): Clustering method. Options: 'kmeans'.
        seed (int): Random seed for the clustering algorithm.

    Returns:
        List[int]: Cluster id (0..k-1) for each channel, in the same channel
            order as train_data's first axis.
    """
    num_channels = train_data.shape[0]

    if k <= 1:
        return [0] * num_channels

    if k >= num_channels:
        return list(range(num_channels))

    if method != "kmeans":
        raise ValueError(f"Unsupported cluster_method: {method}")

    # Each channel is represented by its correlation vector to every other
    # channel; channels with similar correlation profiles are grouped together.
    corr = np.corrcoef(train_data)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(corr)
    return labels.tolist()
