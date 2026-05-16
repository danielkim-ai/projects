"""Non-stationary reward perturbations for continuous-control experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym
import numpy as np


PerturbationMode = Literal["sinusoidal", "abrupt_collapse", "identity"]


@dataclass(frozen=True)
class RewardScaleState:
    """Observable state of the perturbation process at one environment step."""

    timestep: int
    alpha_t: float
    raw_reward: float
    scaled_reward: float


class DynamicRewardScaleWrapper(gym.RewardWrapper):
    """Applies a mathematically specified, time-varying reward scale.

    The wrapper implements alpha_t in (0, 1] and multiplies the environment's
    native reward by that factor at every step. The scaling process is deliberately
    exposed through ``last_scale_state`` so that a Bayesian controller can use the
    current perturbation as contextual evidence.
    """

    def __init__(self, env: gym.Env, perturbation_type: PerturbationMode = "sinusoidal") -> None:
        super().__init__(env)
        self.perturbation_type = perturbation_type
        self.timestep = 0
        self.last_scale_state = RewardScaleState(0, 1.0, 0.0, 0.0)

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self.timestep = 0
        self.last_scale_state = RewardScaleState(0, 1.0, 0.0, 0.0)
        return self.env.reset(**kwargs)

    def alpha(self, timestep: int | None = None) -> float:
        """Return the reward multiplier alpha_t for the requested timestep."""

        t = self.timestep if timestep is None else timestep
        if self.perturbation_type == "sinusoidal":
            alpha_t = 0.55 + 0.45 * np.sin(t / 100.0)
        elif self.perturbation_type == "abrupt_collapse":
            alpha_t = 0.1 if 200 < t < 400 else 1.0
        else:
            alpha_t = 1.0

        return float(np.clip(alpha_t, 0.1, 1.0))

    def reward(self, reward: float) -> float:
        self.timestep += 1
        alpha_t = self.alpha(self.timestep)
        scaled_reward = float(reward * alpha_t)
        self.last_scale_state = RewardScaleState(
            timestep=self.timestep,
            alpha_t=alpha_t,
            raw_reward=float(reward),
            scaled_reward=scaled_reward,
        )
        return scaled_reward
