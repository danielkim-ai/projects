"""Preconditioned Stochastic Gradient Langevin Dynamics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from math import log, sqrt

import torch
from torch import nn


@dataclass(frozen=True)
class DifferentialPrivacyConfig:
    """Gaussian mechanism settings for DP-SGLD experiments."""

    enabled: bool = False
    epsilon: float = 8.0
    delta: float = 1.0e-5
    clip_norm: float = 1.0
    noise_multiplier: float | None = None
    composition_steps: int = 1

    def resolved_noise_multiplier(self) -> float:
        if self.noise_multiplier is not None:
            return max(self.noise_multiplier, 1.0e-8)
        steps = max(float(self.composition_steps), 1.0)
        return sqrt(2.0 * log(1.25 / self.delta)) * sqrt(steps) / max(
            self.epsilon,
            1.0e-8,
        )


class PrivacyAccountant:
    """Conservative Gaussian mechanism privacy-cost tracker.

    This is a lightweight experiment monitor, not a replacement for a formal
    moments/RDP accountant in production privacy audits.
    """

    def __init__(self, config: DifferentialPrivacyConfig):
        self.config = config
        self.steps = 0
        self.epsilon_spent = 0.0

    def step(self) -> float:
        if not self.config.enabled:
            return self.epsilon_spent
        self.steps += 1
        sigma = self.config.resolved_noise_multiplier()
        per_step = sqrt(2.0 * log(1.25 / self.config.delta)) / sigma
        self.epsilon_spent = sqrt(self.steps) * per_step
        return self.epsilon_spent

    def snapshot(self) -> dict[str, float]:
        return {
            "privacy_enabled": float(self.config.enabled),
            "epsilon_spent": float(self.epsilon_spent),
            "epsilon_target": float(self.config.epsilon),
            "delta": float(self.config.delta),
            "dp_steps": float(self.steps),
            "planned_composition_steps": float(self.config.composition_steps),
            "noise_multiplier": float(self.config.resolved_noise_multiplier()),
            "clip_norm": float(self.config.clip_norm),
        }


def apply_dp_gradient_noise(
    parameters: Iterable[nn.Parameter],
    config: DifferentialPrivacyConfig,
) -> float:
    """Clip gradients and add Gaussian noise for DP-SGLD style updates."""

    params_with_grad = [param for param in parameters if param.grad is not None]
    if not params_with_grad or not config.enabled:
        return 0.0
    total_norm = torch.nn.utils.clip_grad_norm_(params_with_grad, config.clip_norm)
    noise_std = config.clip_norm * config.resolved_noise_multiplier()
    for param in params_with_grad:
        param.grad.add_(torch.randn_like(param.grad) * noise_std)
    return float(total_norm.detach().cpu())


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
    privacy: DifferentialPrivacyConfig = field(default_factory=DifferentialPrivacyConfig)
    privacy_accountant: PrivacyAccountant | None = None


@dataclass(frozen=True)
class HyperparameterSGLDConfig:
    """SGLD configuration for Bayesian RL hyperparameters."""

    step_size: float = 2.0e-4
    temperature: float = 1.0
    gamma_prior_mean: float = 0.99
    gamma_prior_precision: float = 20.0
    alpha_prior_log_mean: float = -2.3
    alpha_prior_precision: float = 5.0
    min_gamma: float = 0.80
    max_gamma: float = 0.999
    min_alpha: float = 1.0e-4
    max_alpha: float = 1.0


@dataclass(frozen=True)
class HyperparameterSample:
    """Posterior draw for Thompson-style hyperparameter selection."""

    gamma: float
    alpha: float


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
        if self.config.privacy.enabled:
            apply_dp_gradient_noise(params_with_grad, self.config.privacy)
            if self.config.privacy_accountant is not None:
                self.config.privacy_accountant.step()
        elif self.config.gradient_clip_norm is not None and params_with_grad:
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


class HyperparameterPosteriorSampler:
    """Preconditioned SGLD sampler for entropy and discount posteriors.

    The latent parameters are unconstrained, then transformed to the admissible
    RL ranges: ``gamma`` lies in ``(0, 1)`` and ``alpha`` remains positive.
    """

    def __init__(
        self,
        initial_gamma: float = 0.99,
        initial_alpha: float = 0.1,
        config: HyperparameterSGLDConfig | None = None,
    ):
        self.config = config or HyperparameterSGLDConfig()
        self.gamma_latent = torch.tensor(self._inverse_gamma(initial_gamma), dtype=torch.float32)
        self.alpha_latent = torch.tensor(log(max(initial_alpha, self.config.min_alpha)), dtype=torch.float32)
        self.gamma_square_avg = torch.zeros(())
        self.alpha_square_avg = torch.zeros(())
        self.return_baseline: float | None = None
        self.samples: list[dict[str, float]] = []

    def _inverse_gamma(self, gamma: float) -> float:
        span = self.config.max_gamma - self.config.min_gamma
        scaled = min(max((gamma - self.config.min_gamma) / span, 1.0e-5), 1.0 - 1.0e-5)
        return log(scaled / (1.0 - scaled))

    def _gamma(self) -> torch.Tensor:
        span = self.config.max_gamma - self.config.min_gamma
        return self.config.min_gamma + span * torch.sigmoid(self.gamma_latent)

    def _alpha(self) -> torch.Tensor:
        return torch.clamp(torch.exp(self.alpha_latent), self.config.min_alpha, self.config.max_alpha)

    def sample(self) -> HyperparameterSample:
        """Return the current posterior draw."""

        return HyperparameterSample(gamma=float(self._gamma()), alpha=float(self._alpha()))

    @torch.no_grad()
    def update(self, episode_return: float, episode: int) -> HyperparameterSample:
        """Apply one reward-informed SGLD update to gamma and alpha."""

        if self.return_baseline is None:
            self.return_baseline = float(episode_return)
        advantage = float(episode_return) - self.return_baseline
        self.return_baseline = 0.9 * self.return_baseline + 0.1 * float(episode_return)
        reward_signal = torch.tensor(advantage / (abs(self.return_baseline) + 1.0), dtype=torch.float32)

        gamma = self._gamma()
        alpha = self._alpha()
        gamma_prior_grad = self.config.gamma_prior_precision * (gamma - self.config.gamma_prior_mean)
        alpha_prior_grad = self.config.alpha_prior_precision * (
            self.alpha_latent - self.config.alpha_prior_log_mean
        )
        gamma_grad = gamma_prior_grad - reward_signal * (1.0 - gamma)
        alpha_grad = alpha_prior_grad + reward_signal * alpha

        self.gamma_square_avg.mul_(0.99).add_(gamma_grad.pow(2), alpha=0.01)
        self.alpha_square_avg.mul_(0.99).add_(alpha_grad.pow(2), alpha=0.01)
        gamma_preconditioner = (self.gamma_square_avg + 1.0e-8).rsqrt()
        alpha_preconditioner = (self.alpha_square_avg + 1.0e-8).rsqrt()
        noise = (self.config.step_size * self.config.temperature) ** 0.5

        self.gamma_latent.add_(
            -0.5 * self.config.step_size * gamma_preconditioner * gamma_grad
            + noise * gamma_preconditioner.sqrt() * torch.randn(())
        )
        self.alpha_latent.add_(
            -0.5 * self.config.step_size * alpha_preconditioner * alpha_grad
            + noise * alpha_preconditioner.sqrt() * torch.randn(())
        )

        sample = self.sample()
        self.samples.append(
            {
                "episode": float(episode),
                "gamma": sample.gamma,
                "alpha": sample.alpha,
                "return": float(episode_return),
            }
        )
        return sample

