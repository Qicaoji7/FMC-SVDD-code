"""Federated Client — Section IV-C.

Each client performs:
  - Local training with SVDD + SimGRACE contrastive loss
  - Inflation signal computation (Eq. 12): ΔR_m = R_agg - R_local
  - Cluster management (re-estimation in Phase I, fixed in Phase II)
"""

import torch
from typing import Dict, Tuple, Optional


class FederatedClient:
    """FMC-SVDD federated client.

    Args:
        client_id: Unique client identifier.
        local_data: Training data array (N, num_sensors, T).
        config: Training configuration.
        device: Torch device.
    """

    def __init__(self, client_id: int, local_data, config: dict,
                 device: str = 'cuda:0'):
        raise NotImplementedError("Coming soon")

    def train_local(self, global_model, epochs: int, lr: float,
                    **kwargs) -> Tuple[Dict, float, float]:
        """Local training with combined loss.

        L_total = L_SVDD + λ_cl · L_SimGRACE

        Returns:
            (updated_state_dict, avg_loss, weighted_avg_radius)
        """
        raise NotImplementedError("Coming soon")

    def compute_inflation_signal(self, global_model) -> float:
        """Compute inflation signal ΔR_m (Eq. 12).

        ΔR_m = R̄_m(W^{t+1}) - R̄_m(W_m^t)

        Positive value indicates aggregation inflated this client's boundary.
        """
        raise NotImplementedError("Coming soon")

    def recluster(self, encoder):
        """Re-estimate SVDD cluster centers (Phase I only)."""
        raise NotImplementedError("Coming soon")

    def fix_clusters(self):
        """Lock cluster assignments (transition to Phase II)."""
        raise NotImplementedError("Coming soon")
