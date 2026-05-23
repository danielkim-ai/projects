from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import torch
from torch import Tensor

from utils import EpisodeBatch, MIAReport, simulate_membership_inference


class PolicyLike(Protocol):
    def __call__(self, states: Tensor) -> Tensor:
        ...


@dataclass(frozen=True)
class ReturnSummary:
    mean_return: float
    std_return: float
    episode_count: int


class ExpectedReturnEvaluator:
    """Evaluate policy quality on logged trajectories or replay-style rollouts."""

    @staticmethod
    def logged_return(episodes: Sequence[EpisodeBatch]) -> ReturnSummary:
        returns = torch.stack([episode.rewards.sum() for episode in episodes]).float()
        std = returns.std(unbiased=False) if returns.numel() > 1 else torch.tensor(0.0)
        return ReturnSummary(
            mean_return=float(returns.mean().item()),
            std_return=float(std.item()),
            episode_count=len(episodes),
        )

    @staticmethod
    def behavior_clone_action_error(policy: PolicyLike, episodes: Sequence[EpisodeBatch]) -> float:
        errors: list[Tensor] = []
        with torch.no_grad():
            for episode in episodes:
                predicted = policy(episode.states)
                errors.append((predicted - episode.actions).square().mean())
        return float(torch.stack(errors).mean().item())


@dataclass(frozen=True)
class RDPAccount:
    epsilon: float
    delta: float
    order: float
    steps: int


class RDPAccountingTracker:
    """Simple Gaussian RDP tracker mirroring the prototype optimizer interface."""

    def __init__(
        self,
        noise_multiplier: float,
        sampling_rate: float,
        orders: Sequence[float] = (2, 3, 4, 8, 16, 32, 64),
    ) -> None:
        if sampling_rate <= 0.0 or sampling_rate > 1.0:
            raise ValueError("sampling_rate must lie in (0, 1]")
        self.noise_multiplier = noise_multiplier
        self.sampling_rate = sampling_rate
        self.orders = tuple(float(order) for order in orders)
        self.steps = 0

    def update(self, steps: int = 1) -> None:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        self.steps += steps

    def compute_epsilon(self, delta: float) -> RDPAccount:
        if self.noise_multiplier <= 0.0:
            return RDPAccount(float("inf"), delta, self.orders[0], self.steps)
        candidates: list[tuple[float, float]] = []
        for order in self.orders:
            gaussian_rdp = order / (2.0 * self.noise_multiplier**2)
            sampled_rdp = self.sampling_rate**2 * gaussian_rdp
            epsilon = self.steps * sampled_rdp
            epsilon += torch.log(torch.tensor(1.0 / delta)).item() / (order - 1.0)
            candidates.append((epsilon, order))
        epsilon, order = min(candidates, key=lambda item: item[0])
        return RDPAccount(float(epsilon), delta, order, self.steps)


@dataclass(frozen=True)
class MIASuccessSummary:
    report: MIAReport
    success_rate_delta: float


class MIAEvaluator:
    """Quantify and optionally visualize MIA success before and after unlearning."""

    @staticmethod
    def evaluate(
        member_losses_before: Tensor,
        member_losses_after: Tensor,
        nonmember_losses: Tensor,
    ) -> MIASuccessSummary:
        report = simulate_membership_inference(
            member_losses_before=member_losses_before,
            member_losses_after=member_losses_after,
            nonmember_losses=nonmember_losses,
        )
        return MIASuccessSummary(
            report=report,
            success_rate_delta=report.after_accuracy - report.before_accuracy,
        )

    @staticmethod
    def plot(summary: MIASuccessSummary, output_path: str | Path) -> Path:
        import matplotlib.pyplot as plt

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        labels = ["before", "after"]
        values = [summary.report.before_accuracy, summary.report.after_accuracy]
        plt.figure(figsize=(4.5, 3.0))
        plt.bar(labels, values, color=["#4c78a8", "#72b7b2"])
        plt.ylim(0.0, 1.0)
        plt.ylabel("MIA success rate")
        plt.title("Membership Inference Attack")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return path
