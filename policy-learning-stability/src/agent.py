"""Bayesian-SAC research core with contextual temperature adaptation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Normal


@dataclass(frozen=True)
class BayesianSACConfig:
    """Architectural and Bayesian optimisation controls."""

    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    min_temperature: float = 0.01
    max_temperature: float = 1.0
    gp_noise: float = 1e-4
    kernel_length_scale: float = 0.25
    exploration_beta: float = 1.96


@dataclass
class GaussianProcessSurrogate:
    """A compact radial-basis GP surrogate over contextual reward scales."""

    noise: float = 1e-4
    length_scale: float = 0.25
    contexts: list[float] = field(default_factory=list)
    targets: list[float] = field(default_factory=list)

    def observe(self, context_alpha_t: float, realised_return: float) -> None:
        """Store an episode-level observation for posterior temperature search."""

        bounded_context = float(np.clip(context_alpha_t, 1e-6, 1.0))
        stable_return = float(np.tanh(realised_return / 1000.0))
        target_temperature = 0.2 / bounded_context
        adjusted_target = np.clip(target_temperature * (1.0 + 0.25 * stable_return), 0.01, 1.0)
        self.contexts.append(bounded_context)
        self.targets.append(float(adjusted_target))

    def posterior(self, context_alpha_t: float) -> tuple[float, float]:
        """Predict the posterior mean and variance for a candidate alpha context."""

        x_star = np.array([[float(np.clip(context_alpha_t, 1e-6, 1.0))]])
        if not self.contexts:
            prior_mean = float(np.clip(0.2 / x_star.item(), 0.01, 1.0))
            return prior_mean, 0.05

        x_train = np.asarray(self.contexts, dtype=np.float64).reshape(-1, 1)
        y_train = np.asarray(self.targets, dtype=np.float64).reshape(-1, 1)
        k_xx = self._kernel(x_train, x_train) + self.noise * np.eye(len(x_train))
        k_xs = self._kernel(x_train, x_star)
        k_ss = self._kernel(x_star, x_star)

        solved = np.linalg.solve(k_xx, y_train)
        mean = k_xs.T @ solved
        covariance = k_ss - k_xs.T @ np.linalg.solve(k_xx, k_xs)
        variance = float(np.clip(covariance.item(), 1e-6, 1.0))
        return float(mean.item()), variance

    def _kernel(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        squared_distance = (left - right.T) ** 2
        return np.exp(-0.5 * squared_distance / (self.length_scale**2))


class Actor(nn.Module):
    """Gaussian policy skeleton for entropy-regularised control."""

    def __init__(self, config: BayesianSACConfig) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(config.hidden_dim, config.action_dim)
        self.log_std = nn.Linear(config.hidden_dim, config.action_dim)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        features = self.backbone(state)
        return self.mean(features), self.log_std(features).clamp(-5.0, 2.0)

    def sample(self, state: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_std = self(state)
        distribution = Normal(mean, log_std.exp())
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        correction = torch.log1p(-action.pow(2) + 1e-6)
        log_probability = distribution.log_prob(raw_action) - correction
        return action, log_probability.sum(dim=-1, keepdim=True)


class Critic(nn.Module):
    """Action-value approximator used by the twin-critic SAC objective."""

    def __init__(self, config: BayesianSACConfig) -> None:
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(config.state_dim + config.action_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        return self.value(torch.cat([state, action], dim=-1))


class ContextualBayesianSAC(nn.Module):
    """SAC scaffold whose entropy temperature is inferred by Bayesian search."""

    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4) -> None:
        super().__init__()
        self.config = BayesianSACConfig(state_dim=state_dim, action_dim=action_dim)
        self.actor = Actor(self.config)
        self.critic_1 = Critic(self.config)
        self.critic_2 = Critic(self.config)
        self.gp = GaussianProcessSurrogate(
            noise=self.config.gp_noise,
            length_scale=self.config.kernel_length_scale,
        )
        self.actor_optimiser = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimiser = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=lr,
        )

    def sample_optimal_temperature(self, context_alpha_t: float) -> float:
        """Sample a BO-informed temperature by combining posterior mean and variance."""

        posterior_mean, posterior_variance = self.gp.posterior(context_alpha_t)
        uncertainty_bonus = self.config.exploration_beta * np.sqrt(posterior_variance)
        sampled_temperature = np.random.normal(posterior_mean + uncertainty_bonus, np.sqrt(posterior_variance))
        return float(np.clip(sampled_temperature, self.config.min_temperature, self.config.max_temperature))

    def update_surrogate(self, episode_contexts: Iterable[float], episode_return: float) -> None:
        """Update the GP after an episode using the realised return and mean alpha_t."""

        contexts = list(episode_contexts)
        if not contexts:
            return
        mean_context = float(np.mean(contexts))
        self.gp.observe(mean_context, episode_return)

    def entropy_regularised_actor_loss(self, states: Tensor, context_alpha_t: float) -> Tensor:
        """Compute the SAC actor loss using the Bayesian temperature posterior."""

        actions, log_probabilities = self.actor.sample(states)
        q_value = torch.minimum(self.critic_1(states, actions), self.critic_2(states, actions))
        temperature = torch.tensor(
            self.sample_optimal_temperature(context_alpha_t),
            dtype=states.dtype,
            device=states.device,
        )
        return (temperature * log_probabilities - q_value).mean()
