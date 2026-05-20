"""Hybrid VI-to-MCMC recalibration logic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .metrics import effective_sample_size
from .sgld import SGLD, SGLDConfig
from .vac import VariationalActorCritic


@dataclass(frozen=True)
class HybridRecalibrationConfig:
    """Controls when and how SG-MCMC corrects the variational posterior."""

    recalibrate_every_episodes: int = 10
    mcmc_steps: int = 8
    warmup_steps: int = 16
    mode_collapse_kl_threshold: float = 100.0
    ess_floor: float = 16.0
    ess_window: int = 20
    sgld: SGLDConfig = SGLDConfig()


class HybridRecalibrator:
    """Periodically corrects a VAC posterior with SGLD transitions.

    VI supplies fast online updates. SGLD is then used as a warm-started
    posterior correction step, mitigating mean-field mode collapse without
    paying full MCMC burn-in cost at every episode.
    """

    def __init__(self, model: VariationalActorCritic, config: HybridRecalibrationConfig):
        self.model = model
        self.config = config
        self.sgld = SGLD(model.parameters(), config.sgld)
        self.episode_scores: list[float] = []
        self.latest_policy_sample: dict[str, torch.Tensor] | None = None

    def update_episode_score(self, score: float) -> float:
        """Record a scalar chain diagnostic and return rolling ESS."""

        self.episode_scores.append(float(score))
        window = self.episode_scores[-self.config.ess_window :]
        return effective_sample_size(np.asarray(window, dtype=float))

    def should_recalibrate(self, episode: int, kl_estimate: float, ess_estimate: float | None = None) -> bool:
        scheduled = episode > 0 and episode % self.config.recalibrate_every_episodes == 0
        collapsed = kl_estimate > self.config.mode_collapse_kl_threshold
        low_ess = ess_estimate is not None and ess_estimate < self.config.ess_floor
        return scheduled or collapsed or low_ess

    def needs_warmup(self, ess_estimate: float | None) -> bool:
        return ess_estimate is not None and ess_estimate < self.config.ess_floor

    def recalibrate(self, closure, force_warmup: bool = False) -> list[float]:
        """Run SGLD correction steps.

        The closure must return a differentiable negative log posterior proxy,
        typically the VAC ELBO loss plus the SGLD prior penalty.
        """

        losses: list[float] = []
        self.model.train()
        steps = self.config.mcmc_steps + (self.config.warmup_steps if force_warmup else 0)
        for _ in range(steps):
            self.sgld.zero_grad()
            loss = closure() + self.sgld.prior_penalty()
            loss.backward()
            self.sgld.step()
            losses.append(float(loss.detach().cpu()))
        self.latest_policy_sample = self.sgld.sample_for_thompson(self.model)
        return losses

    def sample_policy_state(self) -> dict[str, torch.Tensor]:
        """Return the latest posterior policy draw for Thompson Sampling."""

        if self.latest_policy_sample is None:
            self.latest_policy_sample = self.sgld.sample_for_thompson(self.model)
        return self.latest_policy_sample

    @torch.no_grad()
    def posterior_snapshot(self) -> dict[str, float]:
        """Return lightweight diagnostics for report extraction."""

        total_norm = 0.0
        total_params = 0
        for param in self.model.parameters():
            total_norm += float(param.pow(2).sum().cpu())
            total_params += param.numel()
        return {
            "parameter_l2": total_norm**0.5,
            "parameter_count": float(total_params),
        }

