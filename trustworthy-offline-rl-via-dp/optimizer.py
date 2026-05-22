from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor
from torch.optim import Optimizer

from utils import flatten_tensors


class ClipNormSchedule(ABC):
    @abstractmethod
    def at_step(self, step: int) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class PhaseClipNormSchedule(ClipNormSchedule):
    warmup_steps: int
    early_norm: float
    late_norm: float

    def at_step(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.early_norm
        return self.late_norm


@dataclass(frozen=True)
class RDPBudget:
    epsilon: float
    delta: float
    best_order: float


class GaussianRDPAccountant:
    """Lightweight RDP accountant for repeated Gaussian trajectory releases."""

    def __init__(self, noise_multiplier: float, sampling_rate: float) -> None:
        self.noise_multiplier = noise_multiplier
        self.sampling_rate = sampling_rate
        self.steps = 0
        self.orders = tuple(float(order) for order in (2, 3, 4, 8, 16, 32, 64))

    def step(self) -> None:
        self.steps += 1

    def budget(self, delta: float) -> RDPBudget:
        if self.noise_multiplier <= 0.0:
            return RDPBudget(float("inf"), delta, self.orders[0])
        epsilons: list[tuple[float, float]] = []
        for order in self.orders:
            gaussian_rdp = order / (2.0 * self.noise_multiplier**2)
            sampled_rdp = self.sampling_rate**2 * gaussian_rdp
            epsilon = self.steps * sampled_rdp
            epsilon = epsilon + torch.log(torch.tensor(1.0 / delta)).item() / (order - 1.0)
            epsilons.append((epsilon, order))
        epsilon, best_order = min(epsilons, key=lambda item: item[0])
        return RDPBudget(epsilon=epsilon, delta=delta, best_order=best_order)


@dataclass(frozen=True)
class DPStepStats:
    clip_norm: float
    unclipped_norm_mean: float
    clipped_fraction: float
    noise_std: float
    trajectory_count: int


class TrajectoryDPSGD(Optimizer):
    """DP-SGD whose adjacency unit is one whole offline trajectory."""

    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float,
        clip_schedule: ClipNormSchedule,
        noise_multiplier: float,
        sampling_rate: float,
    ) -> None:
        defaults = {"lr": lr}
        super().__init__(parameters, defaults)
        self.clip_schedule = clip_schedule
        self.noise_multiplier = noise_multiplier
        self.accountant = GaussianRDPAccountant(noise_multiplier, sampling_rate)
        self.global_step = 0

    @property
    def flat_parameters(self) -> list[Tensor]:
        return [
            parameter
            for group in self.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]

    def dp_step(self, trajectory_losses: Sequence[Tensor]) -> DPStepStats:
        if not trajectory_losses:
            raise ValueError("trajectory_losses must contain at least one episode loss")
        parameters = self.flat_parameters
        clip_norm = self.clip_schedule.at_step(self.global_step)
        clipped_trajectory_grads: list[list[Tensor]] = []
        raw_norms: list[Tensor] = []
        for index, trajectory_loss in enumerate(trajectory_losses):
            # Transition-level DP-SGD clips each transition gradient separately.
            # Here the scalar is already the SUM of every transition loss in one
            # episode, so the next gradient represents trajectory adjacency.
            gradients = torch.autograd.grad(
                trajectory_loss,
                parameters,
                retain_graph=index < len(trajectory_losses) - 1,
                allow_unused=True,
            )
            dense_gradients = [
                torch.zeros_like(parameter) if gradient is None else gradient
                for parameter, gradient in zip(parameters, gradients)
            ]
            norm = flatten_tensors(dense_gradients).norm(p=2)
            scale = torch.clamp(torch.tensor(clip_norm, device=norm.device) / (norm + 1e-12), max=1.0)
            clipped_trajectory_grads.append([gradient * scale for gradient in dense_gradients])
            raw_norms.append(norm.detach())

        trajectory_count = len(clipped_trajectory_grads)
        noise_std = self.noise_multiplier * clip_norm
        for parameter_index, parameter in enumerate(parameters):
            aggregate = torch.stack(
                [
                    trajectory_gradients[parameter_index]
                    for trajectory_gradients in clipped_trajectory_grads
                ]
            ).sum(dim=0)
            noise = torch.randn_like(aggregate) * noise_std
            released_gradient = (aggregate + noise) / trajectory_count
            parameter.grad = released_gradient

        for group in self.param_groups:
            lr = float(group["lr"])
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.data.add_(parameter.grad, alpha=-lr)
                    parameter.grad = None
        self.global_step += 1
        self.accountant.step()
        norms = torch.stack(raw_norms)
        return DPStepStats(
            clip_norm=clip_norm,
            unclipped_norm_mean=float(norms.mean().item()),
            clipped_fraction=float((norms > clip_norm).float().mean().item()),
            noise_std=noise_std / trajectory_count,
            trajectory_count=trajectory_count,
        )
