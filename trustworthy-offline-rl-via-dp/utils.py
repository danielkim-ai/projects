from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class EpisodeBatch:
    """An offline trajectory whose transitions share one deletion/privacy unit."""

    episode_id: int
    states: Tensor
    actions: Tensor
    rewards: Tensor
    next_states: Tensor
    dones: Tensor

    def to(self, device: torch.device) -> "EpisodeBatch":
        return EpisodeBatch(
            episode_id=self.episode_id,
            states=self.states.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            next_states=self.next_states.to(device),
            dones=self.dones.to(device),
        )


@dataclass(frozen=True)
class MIAReport:
    before_accuracy: float
    after_accuracy: float
    before_gap: float
    after_gap: float


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def make_synthetic_episodes(
    episode_count: int,
    horizon: int,
    state_dim: int,
    action_dim: int,
    generator: torch.Generator,
) -> list[EpisodeBatch]:
    episodes: list[EpisodeBatch] = []
    action_projection = torch.randn(
        state_dim, action_dim, generator=generator
    ) / max(state_dim, 1) ** 0.5
    nonlinear_projection = torch.randn(
        state_dim, state_dim, generator=generator
    ) / max(state_dim, 1) ** 0.5
    reward_projection = torch.randn(
        state_dim, action_dim, generator=generator
    ) / max(state_dim, 1) ** 0.5
    time_grid = torch.linspace(0.0, 1.0, horizon).unsqueeze(-1)
    frequencies = torch.linspace(0.7, 2.3, state_dim).unsqueeze(0)
    for episode_id in range(episode_count):
        latent = torch.randn(horizon, state_dim, generator=generator)
        phase = 0.13 * episode_id
        seasonal = torch.sin(2.0 * torch.pi * time_grid * frequencies + phase)
        interaction = torch.cos(latent @ nonlinear_projection)
        states = 0.62 * latent + 0.28 * seasonal + 0.10 * interaction

        behavioural_signal = (
            states @ action_projection
            + 0.35 * torch.sin(states @ reward_projection)
            + 0.18 * torch.cos(states[:, :action_dim])
        )
        behavior_actions = torch.tanh(behavioural_signal)
        actions = torch.clamp(
            behavior_actions
            + 0.18 * torch.randn(horizon, action_dim, generator=generator),
            -1.0,
            1.0,
        )
        drift = 0.16 * torch.randn(horizon, state_dim, generator=generator)
        action_effect = actions.mean(dim=-1, keepdim=True)
        next_states = (
            0.74 * states
            + 0.18 * torch.sin(states @ nonlinear_projection)
            + 0.10 * torch.cos(states)
            + 0.12 * action_effect
            + drift
        )
        target_features = states @ reward_projection + 0.35 * torch.sin(states[:, :action_dim])
        target_action = torch.tanh(target_features)
        curvature_penalty = 0.08 * torch.sin(states).square().sum(dim=-1)
        transition_bonus = 0.18 * torch.cos(next_states - states).mean(dim=-1)
        rewards = (
            1.6
            - 1.35 * (actions - target_action).square().sum(dim=-1)
            - curvature_penalty
            + transition_bonus
        )
        rewards = rewards + 0.10 * torch.randn(horizon, generator=generator)
        dones = torch.zeros(horizon)
        dones[-1] = 1.0
        episodes.append(
            EpisodeBatch(
                episode_id=episode_id,
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones,
            )
        )
    return episodes


def choose_episode_batch(
    episodes: Sequence[EpisodeBatch],
    batch_size: int,
    generator: torch.Generator,
) -> list[EpisodeBatch]:
    if batch_size >= len(episodes):
        return list(episodes)
    indices = torch.randperm(len(episodes), generator=generator)[:batch_size]
    return [episodes[int(index)] for index in indices]


def flatten_tensors(tensors: Iterable[Tensor]) -> Tensor:
    pieces = [tensor.reshape(-1) for tensor in tensors]
    if not pieces:
        return torch.empty(0)
    return torch.cat(pieces)


def membership_features(losses: Tensor) -> Tensor:
    centered = losses.detach().float()
    return torch.stack((centered, centered.square()), dim=-1)


def simulate_membership_inference(
    member_losses_before: Tensor,
    member_losses_after: Tensor,
    nonmember_losses: Tensor,
) -> MIAReport:
    """Evaluate a threshold MIA where unusually low loss implies membership."""

    def attack_accuracy(member_losses: Tensor) -> tuple[float, float]:
        threshold = torch.quantile(
            torch.cat((member_losses, nonmember_losses)), q=0.5
        )
        member_predictions = member_losses <= threshold
        nonmember_predictions = nonmember_losses > threshold
        accuracy = torch.cat((member_predictions, nonmember_predictions)).float().mean()
        gap = nonmember_losses.mean() - member_losses.mean()
        return float(accuracy.item()), float(gap.item())

    before_accuracy, before_gap = attack_accuracy(member_losses_before)
    after_accuracy, after_gap = attack_accuracy(member_losses_after)
    return MIAReport(
        before_accuracy=before_accuracy,
        after_accuracy=after_accuracy,
        before_gap=before_gap,
        after_gap=after_gap,
    )
