"""Report metrics for PAC-Bayes, regret, and calibration diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReportMetrics:
    pac_bayes_bound: float
    regret: float
    calibration_error: float
    effective_sample_size: float

    def to_json(self) -> dict[str, float]:
        return {
            "pac_bayes_bound": self.pac_bayes_bound,
            "regret": self.regret,
            "calibration_error": self.calibration_error,
            "effective_sample_size": self.effective_sample_size,
        }


def pac_bayes_bound(empirical_loss: float, kl_qp: float, n: int, delta: float = 0.05) -> float:
    """Compute a standard PAC-Bayes upper-bound proxy."""

    if n <= 0:
        raise ValueError("n must be positive for a PAC-Bayes bound.")
    complexity = (kl_qp + math.log((2.0 * math.sqrt(n)) / delta)) / (2.0 * n)
    return empirical_loss + math.sqrt(max(0.0, complexity))


def bayesian_regret(optimal_returns: np.ndarray, realised_returns: np.ndarray) -> float:
    """Estimate cumulative Bayesian regret from return traces."""

    return float(np.maximum(optimal_returns - realised_returns, 0.0).sum())


def expected_calibration_error(confidences: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> float:
    """Compute scalar ECE for posterior predictive confidence diagnostics."""

    confidences = np.asarray(confidences, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidences >= left) & (confidences < right)
        if not mask.any():
            continue
        ece += mask.mean() * abs(confidences[mask].mean() - outcomes[mask].mean())
    return float(ece)


def effective_sample_size(chain_values: np.ndarray) -> float:
    """Conservative ESS proxy using lag-one autocorrelation."""

    values = np.asarray(chain_values, dtype=float)
    if values.size < 3:
        return float(values.size)
    centred = values - values.mean()
    denom = np.dot(centred, centred)
    if denom <= 1.0e-12:
        return float(values.size)
    rho1 = float(np.dot(centred[:-1], centred[1:]) / denom)
    rho1 = min(max(rho1, -0.99), 0.99)
    return float(values.size * (1.0 - rho1) / (1.0 + rho1))

