"""Stochastic Gradient Langevin Dynamics for posterior-aware updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class SGLDConfig:
    """Configuration for an SGLD posterior update."""

    step_size: float = 1.0e-5
    temperature: float = 1.0
    prior_precision: float = 1.0
    gradient_clip_norm: float | None = 10.0


class SGLD:
    """Minimal SGLD optimiser.

    The update approximates posterior sampling by combining a noisy Langevin
    perturbation with the gradient of the negative log posterior objective.
    """

    def __init__(self, parameters: Iterable[nn.Parameter], config: SGLDConfig):
        self.parameters = list(parameters)
        self.config = config

    def prior_penalty(self) -> torch.Tensor:
        """Return the Gaussian log-prior penalty used by the sampler."""

        penalty = torch.zeros((), device=self.parameters[0].device)
        for param in self.parameters:
            penalty = penalty + 0.5 * self.config.prior_precision * param.pow(2).sum()
        return penalty

    @torch.no_grad()
    def step(self) -> None:
        """Apply one SGLD step to all parameters with available gradients."""

        params_with_grad = [param for param in self.parameters if param.grad is not None]
        if self.config.gradient_clip_norm is not None and params_with_grad:
            torch.nn.utils.clip_grad_norm_(params_with_grad, self.config.gradient_clip_norm)

        noise_scale = (2.0 * self.config.step_size * self.config.temperature) ** 0.5
        for param in params_with_grad:
            noise = torch.randn_like(param) * noise_scale
            param.add_(-self.config.step_size * param.grad + noise)

    def zero_grad(self) -> None:
        for param in self.parameters:
            param.grad = None

