"""Preconditioned Stochastic Gradient Langevin Dynamics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SGLDConfig:
    """Configuration for an SGLD posterior update."""

    step_size: float = 1.0e-5
    temperature: float = 1.0
    prior_precision: float = 1.0
    gradient_clip_norm: float | None = 10.0
    precondition_decay: float = 0.99
    precondition_eps: float = 1.0e-8
    burn_in_steps: int = 25


class SGLD:
    """Preconditioned SGLD optimiser for posterior-aware RL.

    The preconditioner tracks a running second moment of gradients, making the
    sampler less sensitive to reward-scale changes in MuJoCo rollouts.
    """

    def __init__(self, parameters: Iterable[nn.Parameter], config: SGLDConfig):
        self.parameters = list(parameters)
        self.config = config
        self.state: dict[nn.Parameter, torch.Tensor] = {
            param: torch.zeros_like(param) for param in self.parameters
        }
        self.steps = 0

    def prior_penalty(self) -> torch.Tensor:
        """Return the Gaussian log-prior penalty used by the sampler."""

        penalty = torch.zeros((), device=self.parameters[0].device)
        for param in self.parameters:
            penalty = penalty + 0.5 * self.config.prior_precision * param.pow(2).sum()
        return penalty

    @torch.no_grad()
    def step(self) -> None:
        """Apply one preconditioned SGLD step."""

        params_with_grad = [param for param in self.parameters if param.grad is not None]
        if self.config.gradient_clip_norm is not None and params_with_grad:
            torch.nn.utils.clip_grad_norm_(params_with_grad, self.config.gradient_clip_norm)

        self.steps += 1
        for param in params_with_grad:
            square_avg = self.state[param]
            square_avg.mul_(self.config.precondition_decay).addcmul_(
                param.grad,
                param.grad,
                value=1.0 - self.config.precondition_decay,
            )
            preconditioner = square_avg.add(self.config.precondition_eps).rsqrt()
            drift = -0.5 * self.config.step_size * preconditioner * param.grad
            noise_scale = torch.sqrt(
                self.config.step_size * self.config.temperature * preconditioner
            )
            param.add_(drift + torch.randn_like(param) * noise_scale)

    def zero_grad(self) -> None:
        for param in self.parameters:
            param.grad = None

    @torch.no_grad()
    def posterior_sample_state(self, module: nn.Module) -> dict[str, torch.Tensor]:
        """Return a detached posterior draw for Thompson Sampling."""

        return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}

    def sample_for_thompson(self, module: nn.Module) -> dict[str, torch.Tensor]:
        """Alias documenting the posterior draw used for Thompson Sampling."""

        return self.posterior_sample_state(module)

    @staticmethod
    @torch.no_grad()
    def load_posterior_sample(module: nn.Module, sample: dict[str, torch.Tensor]) -> None:
        """Load a posterior draw into a policy before an episode."""

        module.load_state_dict(sample, strict=True)

