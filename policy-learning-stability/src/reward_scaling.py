"""Dynamic reward scaling wrapper for non-stationary reward regimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class DynamicRewardScalingConfig:
    """Controls adaptive reward normalisation."""

    decay: float = 0.995
    epsilon: float = 1e-8
    clip: float = 10.0
    minimum_scale: float = 1e-3


class DynamicRewardScalingWrapper(gym.Wrapper):
    """Normalises rewards with an exponential moving variance estimate."""

    def __init__(self, env: gym.Env, config: DynamicRewardScalingConfig | None = None) -> None:
        super().__init__(env)
        self.config = config or DynamicRewardScalingConfig()
        self.running_mean = 0.0
        self.running_second_moment = 1.0
        self.steps = 0

    @property
    def scale(self) -> float:
        variance = max(self.running_second_moment - self.running_mean**2, self.config.minimum_scale)
        return float(np.sqrt(variance + self.config.epsilon))

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        scaled_reward = self._scale_reward(float(reward))
        info = dict(info)
        info["raw_reward"] = float(reward)
        info["reward_scale"] = self.scale
        return observation, scaled_reward, terminated, truncated, info

    def _scale_reward(self, reward: float) -> float:
        self.steps += 1
        if self.steps == 1:
            self.running_mean = reward
            self.running_second_moment = reward * reward
        else:
            decay = self.config.decay
            self.running_mean = decay * self.running_mean + (1.0 - decay) * reward
            self.running_second_moment = decay * self.running_second_moment + (1.0 - decay) * reward * reward

        centred = reward - self.running_mean
        scaled = centred / self.scale
        return float(np.clip(scaled, -self.config.clip, self.config.clip))
