"""Contextual Bayesian Soft Actor-Critic components.

The implementation is intentionally compact: it exposes the research-facing
interfaces required for uncertainty-conditioned policy updates without binding
the archive to a single training framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.distributions import Normal
import torch.nn.functional as F


@dataclass(frozen=True)
class BayesianSACConfig:
    """Configuration for contextual Bayesian-SAC updates."""

    observation_dim: int
    action_dim: int
    context_dim: int
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    entropy_temperature: float = 0.2
    prior_precision: float = 1.0
    posterior_precision: float = 1.0


class ContextualPolicy(nn.Module):
    """Gaussian policy conditioned on observations and latent context."""

    def __init__(self, config: BayesianSACConfig) -> None:
        super().__init__()
        input_dim = config.observation_dim + config.context_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(config.hidden_dim, config.action_dim)
        self.log_std = nn.Linear(config.hidden_dim, config.action_dim)

    def forward(self, observation: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        features = self.backbone(torch.cat([observation, context], dim=-1))
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(-5.0, 2.0)
        return mean, log_std

    def sample(self, observation: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_std = self(observation, context)
        distribution = Normal(mean, log_std.exp())
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        log_prob = distribution.log_prob(raw_action) - torch.log1p(-action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)


class ContextualCritic(nn.Module):
    """Twin-compatible critic conditioned on observations, actions and context."""

    def __init__(self, config: BayesianSACConfig) -> None:
        super().__init__()
        input_dim = config.observation_dim + config.action_dim + config.context_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, observation: Tensor, action: Tensor, context: Tensor) -> Tensor:
        return self.network(torch.cat([observation, action, context], dim=-1))


class ContextualBayesianSAC(nn.Module):
    """Minimal Bayesian-SAC core with an explicit posterior regulariser."""

    def __init__(self, config: BayesianSACConfig) -> None:
        super().__init__()
        self.config = config
        self.policy = ContextualPolicy(config)
        self.critic_1 = ContextualCritic(config)
        self.critic_2 = ContextualCritic(config)

    def posterior_penalty(self) -> Tensor:
        penalty = torch.zeros((), device=next(self.parameters()).device)
        for parameter in self.parameters():
            penalty = penalty + parameter.pow(2).sum()
        precision_gap = self.config.posterior_precision / max(self.config.prior_precision, 1e-8)
        return 0.5 * precision_gap * penalty

    def actor_loss(self, batch: Mapping[str, Tensor]) -> Tensor:
        action, log_prob = self.policy.sample(batch["observation"], batch["context"])
        q_value = torch.minimum(
            self.critic_1(batch["observation"], action, batch["context"]),
            self.critic_2(batch["observation"], action, batch["context"]),
        )
        entropy_term = self.config.entropy_temperature * log_prob
        return (entropy_term - q_value).mean() + self.posterior_penalty() * 1e-6

    def critic_loss(self, batch: Mapping[str, Tensor]) -> Tensor:
        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample(batch["next_observation"], batch["context"])
            next_q = torch.minimum(
                self.critic_1(batch["next_observation"], next_action, batch["context"]),
                self.critic_2(batch["next_observation"], next_action, batch["context"]),
            )
            target = batch["reward"] + self.config.gamma * (1.0 - batch["done"]) * (
                next_q - self.config.entropy_temperature * next_log_prob
            )

        q_1 = self.critic_1(batch["observation"], batch["action"], batch["context"])
        q_2 = self.critic_2(batch["observation"], batch["action"], batch["context"])
        return F.mse_loss(q_1, target) + F.mse_loss(q_2, target)
