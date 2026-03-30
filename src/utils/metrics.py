"""Evaluation metrics for anomaly detection."""

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area Under the ROC Curve."""
    return roc_auc_score(labels, scores)


def compute_best_f1(labels: np.ndarray, scores: np.ndarray) -> tuple:
    """Best F1 score with optimal threshold."""
    prec, rec, thresh = precision_recall_curve(labels, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    idx = np.argmax(f1)
    return float(f1[idx]), float(thresh[idx]) if idx < len(thresh) else float(thresh[-1])


def compute_inflation_ratio(fed_radius: float, local_radius: float) -> float:
    """Inflation Ratio: IR = R_fed / R_local (Definition 2)."""
    return fed_radius / max(local_radius, 1e-12)
