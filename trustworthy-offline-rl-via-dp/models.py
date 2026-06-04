from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from utils import EpisodeBatch


@dataclass(frozen=True)
class LossTerms:
    total: Tensor
    td: Tensor
    conservative: Tensor
    privacy_regularizer: Tensor
    actor: Tensor


class OfflineRLAlgorithm(nn.Module, ABC):
    @abstractmethod
    def trajectory_loss(self, trajectory: EpisodeBatch, noise_std: float) -> LossTerms:
        raise NotImplementedError

    @abstractmethod
    def loss_vector(self, trajectories: Sequence[EpisodeBatch]) -> Tensor:
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.model = MLP(state_dim + action_dim, 1, hidden_dim)

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        return self.model(torch.cat((states, actions), dim=-1)).squeeze(-1)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.model = MLP(state_dim, 1, hidden_dim)

    def forward(self, states: Tensor) -> Tensor:
        return self.model(states).squeeze(-1)


class BoundedPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.model = MLP(state_dim, action_dim, hidden_dim)

    def forward(self, states: Tensor) -> Tensor:
        return torch.tanh(self.model(states))


class PrivacyAwareCQL(OfflineRLAlgorithm):
    """Small continuous-action CQL critic with DP-noise-aware pessimism."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        discount: float = 0.97,
        cql_weight: float = 0.4,
        privacy_weight: float = 0.2,
        behavior_weight: float = 0.05,
        random_action_count: int = 10,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.discount = discount
        self.cql_weight = cql_weight
        self.privacy_weight = privacy_weight
        self.behavior_weight = behavior_weight
        self.random_action_count = random_action_count
        self.q = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q = QNetwork(state_dim, action_dim, hidden_dim)
        self.policy = BoundedPolicy(state_dim, action_dim, hidden_dim)
        self.target_q.load_state_dict(self.q.state_dict())
        for parameter in self.target_q.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update_target(self, tau: float = 0.02) -> None:
        for target, source in zip(self.target_q.parameters(), self.q.parameters()):
            target.lerp_(source, tau)

    def trajectory_loss(self, trajectory: EpisodeBatch, noise_std: float) -> LossTerms:
        states = trajectory.states
        actions = trajectory.actions
        rewards = trajectory.rewards
        with torch.no_grad():
            next_actions = self.policy(trajectory.next_states)
            bootstrapped = self.target_q(trajectory.next_states, next_actions)
            targets = rewards + self.discount * (1.0 - trajectory.dones) * bootstrapped

        data_q = self.q(states, actions)
        td_loss = F.mse_loss(data_q, targets)
        random_actions = torch.empty(
            states.shape[0], self.random_action_count, self.action_dim, device=states.device
        ).uniform_(-1.0, 1.0)
        expanded_states = states[:, None, :].expand(-1, self.random_action_count, -1)
        ood_q = self.q(
            expanded_states.reshape(-1, states.shape[-1]),
            random_actions.reshape(-1, self.action_dim),
        ).reshape(states.shape[0], self.random_action_count)
        conservative = (
            torch.logsumexp(ood_q, dim=1) - data_q
        ).mean()

        policy_actions = self.policy(states)
        actor_values = self.q(states, policy_actions)
        behavior_anchor = F.mse_loss(policy_actions, actions)
        actor_loss = -actor_values.mean() + self.behavior_weight * behavior_anchor

        # DP noise makes critic peaks harder to trust. Penalising the positive
        # OOD-data gap more aggressively as noise rises preserves CQL pessimism.
        ood_gap = ood_q.max(dim=1).values - data_q.detach()
        privacy_regularizer = (1.0 + noise_std) * F.relu(ood_gap).square().mean()
        total = (
            td_loss
            + self.cql_weight * conservative
            + self.privacy_weight * privacy_regularizer
            + actor_loss
        )
        return LossTerms(
            total=total,
            td=td_loss,
            conservative=conservative,
            privacy_regularizer=privacy_regularizer,
            actor=actor_loss,
        )

    def loss_vector(self, trajectories: Sequence[EpisodeBatch]) -> Tensor:
        losses = [self.trajectory_loss(trajectory, noise_std=0.0).td for trajectory in trajectories]
        return torch.stack(losses)


class PrivacyAwareIQL(OfflineRLAlgorithm):
    """DP-aware IQL using in-sample expectile regression instead of OOD sampling."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        discount: float = 0.97,
        expectile: float = 0.7,
        inverse_temperature: float = 2.0,
        value_weight: float = 0.7,
        actor_weight: float = 0.4,
        privacy_weight: float = 0.08,
    ) -> None:
        super().__init__()
        if not 0.0 < expectile < 1.0:
            raise ValueError("expectile must lie in (0, 1)")
        self.action_dim = action_dim
        self.discount = discount
        self.expectile = expectile
        self.inverse_temperature = inverse_temperature
        self.value_weight = value_weight
        self.actor_weight = actor_weight
        self.privacy_weight = privacy_weight
        self.q = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q = QNetwork(state_dim, action_dim, hidden_dim)
        self.value = ValueNetwork(state_dim, hidden_dim)
        self.policy = BoundedPolicy(state_dim, action_dim, hidden_dim)
        self.target_q.load_state_dict(self.q.state_dict())
        for parameter in self.target_q.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def expectile_loss(diff: Tensor, expectile: float) -> Tensor:
        weights = torch.where(diff > 0.0, expectile, 1.0 - expectile)
        return (weights * diff.square()).mean()

    @torch.no_grad()
    def update_target(self, tau: float = 0.02) -> None:
        for target, source in zip(self.target_q.parameters(), self.q.parameters()):
            target.lerp_(source, tau)

    def trajectory_loss(self, trajectory: EpisodeBatch, noise_std: float) -> LossTerms:
        states = trajectory.states
        actions = trajectory.actions
        rewards = trajectory.rewards
        values = self.value(states)
        with torch.no_grad():
            next_values = self.value(trajectory.next_states)
            targets = rewards + self.discount * (1.0 - trajectory.dones) * next_values

        q_values = self.q(states, actions)
        q_loss = F.mse_loss(q_values, targets)
        value_diff = q_values.detach() - values
        value_loss = self.expectile_loss(value_diff, self.expectile)

        advantages = q_values.detach() - values.detach()
        advantage_weights = torch.exp(self.inverse_temperature * advantages).clamp(max=30.0)
        policy_actions = self.policy(states)
        actor_loss = (advantage_weights * (policy_actions - actions).square().mean(dim=-1)).mean()

        # IQL remains in-distribution: the privacy term damps noisy in-sample
        # Q-V disagreement rather than sampling OOD actions as CQL does.
        privacy_regularizer = (1.0 + noise_std) * (q_values - values.detach()).square().mean()
        total = (
            q_loss
            + self.value_weight * value_loss
            + self.actor_weight * actor_loss
            + self.privacy_weight * privacy_regularizer
        )
        return LossTerms(
            total=total,
            td=q_loss,
            conservative=value_loss,
            privacy_regularizer=privacy_regularizer,
            actor=actor_loss,
        )

    def loss_vector(self, trajectories: Sequence[EpisodeBatch]) -> Tensor:
        losses = [self.trajectory_loss(trajectory, noise_std=0.0).td for trajectory in trajectories]
        return torch.stack(losses)
