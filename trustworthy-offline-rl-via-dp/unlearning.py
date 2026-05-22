from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Sequence

import torch
from torch import Tensor, nn

from utils import EpisodeBatch, flatten_tensors

LossClosure = Callable[[], Tensor]


class DistillablePolicy(Protocol):
    policy: nn.Module


class NoiseMechanism(ABC):
    @abstractmethod
    def privatize(self, tensors: Sequence[Tensor]) -> list[Tensor]:
        raise NotImplementedError


@dataclass(frozen=True)
class GaussianUpdateNoise(NoiseMechanism):
    std: float

    def privatize(self, tensors: Sequence[Tensor]) -> list[Tensor]:
        return [tensor + torch.randn_like(tensor) * self.std for tensor in tensors]


class UnlearningStrategy(ABC):
    @abstractmethod
    def forget(self) -> None:
        raise NotImplementedError


class LiSSAInfluenceUnlearner(UnlearningStrategy):
    """Approximate H^-1 g for a deleted trajectory and apply a noisy correction."""

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        train_loss: LossClosure,
        deleted_loss: LossClosure,
        recursion_depth: int = 8,
        damping: float = 0.03,
        scale: float = 12.0,
        update_lr: float = 0.05,
        max_update_norm: float = 1.0,
        noise: NoiseMechanism | None = None,
    ) -> None:
        self.parameters = [parameter for parameter in parameters if parameter.requires_grad]
        self.train_loss = train_loss
        self.deleted_loss = deleted_loss
        self.recursion_depth = recursion_depth
        self.damping = damping
        self.scale = scale
        self.update_lr = update_lr
        self.max_update_norm = max_update_norm
        self.noise = noise

    def _gradient(self, loss: Tensor, create_graph: bool) -> list[Tensor]:
        gradients = torch.autograd.grad(
            loss,
            self.parameters,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=True,
        )
        return [
            torch.zeros_like(parameter) if gradient is None else gradient
            for parameter, gradient in zip(self.parameters, gradients)
        ]

    def _hvp(self, loss: Tensor, vector: Sequence[Tensor]) -> list[Tensor]:
        gradients = self._gradient(loss, create_graph=True)
        directional = sum(
            (gradient * direction).sum() for gradient, direction in zip(gradients, vector)
        )
        hvp = torch.autograd.grad(
            directional,
            self.parameters,
            retain_graph=False,
            allow_unused=True,
        )
        return [
            torch.zeros_like(parameter) if product is None else product
            for parameter, product in zip(self.parameters, hvp)
        ]

    def inverse_hvp(self, vector: Sequence[Tensor]) -> list[Tensor]:
        estimate = [item.detach().clone() for item in vector]
        base_vector = [item.detach().clone() for item in vector]
        for _ in range(self.recursion_depth):
            curvature = self._hvp(self.train_loss(), estimate)
            estimate = [
                base + (1.0 - self.damping) * current - product / self.scale
                for base, current, product in zip(base_vector, estimate, curvature)
            ]
            estimate = [item.detach() for item in estimate]
        return [item / self.scale for item in estimate]

    def forget(self) -> None:
        deleted_gradient = self._gradient(self.deleted_loss(), create_graph=False)
        correction = self.inverse_hvp(deleted_gradient)
        if self.noise is not None:
            correction = self.noise.privatize(correction)
        correction_norm = flatten_tensors(correction).norm(p=2)
        scale = torch.clamp(
            torch.tensor(self.max_update_norm, device=correction_norm.device)
            / (correction_norm + 1e-12),
            max=1.0,
        )
        with torch.no_grad():
            # Removing an episode subtracts its influence approximation.
            for parameter, direction in zip(self.parameters, correction):
                parameter.add_(direction * scale, alpha=self.update_lr)

    def correction_norm(self) -> float:
        deleted_gradient = self._gradient(self.deleted_loss(), create_graph=False)
        return float(flatten_tensors(self.inverse_hvp(deleted_gradient)).norm().item())


@dataclass
class Shard:
    shard_id: int
    episodes: list[EpisodeBatch]


class EpisodeShardManager:
    """SISA-style sharding that never splits transitions from one episode."""

    def __init__(self, shard_count: int) -> None:
        if shard_count <= 0:
            raise ValueError("shard_count must be positive")
        self.shard_count = shard_count
        self.shards = [Shard(shard_id=index, episodes=[]) for index in range(shard_count)]

    def partition(self, episodes: Sequence[EpisodeBatch]) -> list[Shard]:
        self.shards = [Shard(shard_id=index, episodes=[]) for index in range(self.shard_count)]
        for episode in episodes:
            self.shards[episode.episode_id % self.shard_count].episodes.append(episode)
        return self.shards

    def delete_episode(self, episode_id: int) -> set[int]:
        affected: set[int] = set()
        for shard in self.shards:
            retained = [episode for episode in shard.episodes if episode.episode_id != episode_id]
            if len(retained) != len(shard.episodes):
                shard.episodes = retained
                affected.add(shard.shard_id)
        return affected

    @staticmethod
    def ensemble_actions(models: Sequence[DistillablePolicy], states: Tensor) -> Tensor:
        if not models:
            raise ValueError("models must contain at least one shard policy")
        with torch.no_grad():
            actions = [model.policy(states) for model in models]
        return torch.stack(actions).mean(dim=0)

    @staticmethod
    def distillation_loss(
        student: DistillablePolicy,
        teachers: Sequence[DistillablePolicy],
        states: Tensor,
    ) -> Tensor:
        targets = EpisodeShardManager.ensemble_actions(teachers, states)
        predictions = student.policy(states)
        return torch.nn.functional.mse_loss(predictions, targets)
