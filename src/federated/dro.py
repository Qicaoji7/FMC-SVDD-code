"""Distributionally Robust Optimization (DRO) for γ Calibration — Section IV-B, Eq. 24-27.

Solves a Wasserstein-ball worst-case optimization to automatically
determine the optimal RAA sensitivity γ at each round.
"""

import numpy as np
from scipy.optimize import linprog
from typing import List, Tuple


def solve_dro_weights(client_losses: List[float],
                      base_weights: List[float],
                      kappa: float = 1.0) -> List[float]:
    """Solve DRO optimization for target weights (Eq. 24-26).

    max_{q ∈ U} Σ_m q_m · L_m
    s.t. U = {q : W(q, p) ≤ ε}, ε = κ · std(L)

    Uses total-variation ball approximation solved via linear programming.

    Args:
        client_losses: Per-client average losses L_m.
        base_weights: Nominal distribution p_m.
        kappa: Scale factor for ambiguity set radius.

    Returns:
        DRO-optimal weights q*.
    """
    raise NotImplementedError("Coming soon")


def calibrate_gamma(inflation_signals: List[float],
                    client_losses: List[float],
                    base_weights: List[float],
                    kappa: float = 1.0) -> Tuple[float, List[float]]:
    """Calibrate γ to match DRO-optimal weights (Eq. 27).

    γ* = argmin_γ ||w_RAA(γ) - w_DRO||₂

    Uses binary search over γ ∈ [0, 20].

    Args:
        inflation_signals: Per-client ΔR_m.
        client_losses: Per-client losses.
        base_weights: Data-proportional weights.
        kappa: DRO scale factor.

    Returns:
        Optimal gamma and corresponding weights.
    """
    raise NotImplementedError("Coming soon")
