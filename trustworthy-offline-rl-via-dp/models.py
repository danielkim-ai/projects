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
        hidden_dim: int = 64,
        discount: float = 0.97,
        cql_weight: float = 0.4,
        privacy_weight: float = 0.2,
        behavior_weight: float = 0.05,
        random_action_count: int = 6,
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
