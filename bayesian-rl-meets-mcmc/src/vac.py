"""Variational Actor-Critic skeleton for epistemic uncertainty modelling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal


@dataclass(frozen=True)
class VACConfig:
    """Configuration for the variational actor-critic objective."""

    obs_dim: int
    action_dim: int
    hidden_dim: int = 64
    beta_kl: float = 1.0e-3
    min_log_std: float = -5.0
    max_log_std: float = 2.0


class VariationalLinear(nn.Module):
    """Mean-field Gaussian linear layer with reparameterised weights."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -5.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -5.0))
        nn.init.kaiming_uniform_(self.weight_mu, a=5**0.5)

    @staticmethod
    def _sigma(rho: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(rho) + 1.0e-6

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.weight_mu + self._sigma(self.weight_rho) * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + self._sigma(self.bias_rho) * torch.randn_like(self.bias_mu)
        return torch.nn.functional.linear(inputs, weight, bias)

    def kl_to_standard_normal(self) -> torch.Tensor:
        weight_sigma = self._sigma(self.weight_rho)
        bias_sigma = self._sigma(self.bias_rho)
        weight_kl = 0.5 * (
            self.weight_mu.pow(2) + weight_sigma.pow(2) - 2.0 * weight_sigma.log() - 1.0
        ).sum()
        bias_kl = 0.5 * (
            self.bias_mu.pow(2) + bias_sigma.pow(2) - 2.0 * bias_sigma.log() - 1.0
        ).sum()
        return weight_kl + bias_kl


class VariationalActorCritic(nn.Module):
    """Actor-critic model with variational posterior parameters."""

    def __init__(self, config: VACConfig):
        super().__init__()
        self.config = config
        self.actor_body = nn.Sequential(
            VariationalLinear(config.obs_dim, config.hidden_dim),
            nn.Tanh(),
            VariationalLinear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = VariationalLinear(config.hidden_dim, config.action_dim)
        self.actor_log_std = nn.Parameter(torch.full((config.action_dim,), -0.5))

        self.critic = nn.Sequential(
            VariationalLinear(config.obs_dim, config.hidden_dim),
            nn.Tanh(),
            VariationalLinear(config.hidden_dim, 1),
        )

    def policy_distribution(self, observations: torch.Tensor) -> Normal:
        features = self.actor_body(observations)
        mean = self.actor_mean(features)
        log_std = self.actor_log_std.clamp(self.config.min_log_std, self.config.max_log_std)
        return Normal(mean, log_std.exp())

    def value(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic(observations).squeeze(-1)

    def kl_to_prior(self) -> torch.Tensor:
        kl = torch.zeros((), device=self.actor_log_std.device)
        for module in self.modules():
            if isinstance(module, VariationalLinear):
                kl = kl + module.kl_to_standard_normal()
        return kl

    def elbo_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute a Phase 1 VAC loss from a rollout batch."""

        dist = self.policy_distribution(observations)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        values = self.value(observations)

        actor_loss = -(log_prob * advantages.detach()).mean()
        critic_loss = 0.5 * (returns - values).pow(2).mean()
        kl = self.kl_to_prior() / max(1, observations.shape[0])
        loss = actor_loss + critic_loss + self.config.beta_kl * kl

        return {
            "loss": loss,
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "kl": kl.detach(),
        }

