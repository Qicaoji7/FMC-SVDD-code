"""Radius-Aware Aggregation (RAA) — Section IV-A, Eq. 13-15.

Upweights clients exhibiting boundary inflation to counteract
the radius expansion caused by heterogeneous aggregation.
"""

import numpy as np
from typing import List, Dict, Tuple


def compute_raa_weights(base_weights: List[float],
                        inflation_signals: List[float],
                        gamma: float) -> List[float]:
    """Compute RAA aggregation weights (Eq. 14-15).

    α_m = p_m · exp(γ · [ΔR_m]⁺)
    w_m = α_m / Σ_j α_j

    where [·]⁺ = max(·, 0) and ΔR_m is the inflation signal.

    Args:
        base_weights: Data-proportional weights p_m, summing to 1.
        inflation_signals: Per-client radius change ΔR_m.
        gamma: Sensitivity parameter γ ≥ 0.

    Returns:
        Normalized RAA weights w_m.
    """
    raise NotImplementedError("Coming soon")


def aggregate_with_raa(client_state_dicts: List[Dict],
                       inflation_signals: List[float],
                       gamma: float,
                       base_weights: List[float]
                       ) -> Tuple[Dict, List[float]]:
    """Perform RAA-weighted model aggregation.

    W^{t+1} = Σ_m w_m · W_m^t

    Args:
        client_state_dicts: List of local model state dicts.
        inflation_signals: Per-client ΔR_m values.
        gamma: RAA sensitivity.
        base_weights: Data-proportional base weights.

    Returns:
        Aggregated state dict and weights used.
    """
    raise NotImplementedError("Coming soon")
