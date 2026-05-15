"""Diagnostics for reward-scale stability experiments."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Sequence


@dataclass(frozen=True)
class RewardScaleDiagnostics:
    """Summary statistics for adaptive reward scaling."""

    mean_scale: float
    volatility: float
    maximum_scale: float
    minimum_scale: float
    drift_ratio: float


def summarise_reward_scale(scales: Sequence[float]) -> RewardScaleDiagnostics:
    """Summarise scale drift from a sequence of recorded wrapper diagnostics."""

    if not scales:
        raise ValueError("At least one reward scale observation is required.")

    first = max(float(scales[0]), 1e-12)
    last = float(scales[-1])
    return RewardScaleDiagnostics(
        mean_scale=float(mean(scales)),
        volatility=float(pstdev(scales)) if len(scales) > 1 else 0.0,
        maximum_scale=float(max(scales)),
        minimum_scale=float(min(scales)),
        drift_ratio=last / first,
    )
