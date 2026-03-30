"""Federated data partitioning strategies (Section V-A)."""

import numpy as np
from typing import List


def partition_data(train_data: np.ndarray, mode: str,
                   num_clients: int = 3, seed: int = 42) -> List[np.ndarray]:
    """Partition training data across federated clients.

    Args:
        train_data: (N, num_sensors, T) array.
        mode: 'iid' | 'mild' | 'severe'
            - IID: Random uniform split.
            - Mild non-IID: Sensor-dimension split (each client gets a
              subset of sensors, all time windows).
            - Severe non-IID: K-means clustering on flattened samples,
              clients receive cluster-assigned samples.
        num_clients: Number of federated clients M.
        seed: Random seed.

    Returns:
        List of M arrays, one per client.
    """
    raise NotImplementedError("Coming soon")
