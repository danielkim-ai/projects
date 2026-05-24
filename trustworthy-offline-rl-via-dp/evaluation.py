from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

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


class UtilityEvaluator:
    """Score offline utility and Monte Carlo proxy degradation under noise."""

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

    @staticmethod
    def monte_carlo_proxy_return(
        policy: PolicyLike,
        episodes: Sequence[EpisodeBatch],
        action_penalty: float = 0.25,
    ) -> ReturnSummary:
        proxy_returns: list[Tensor] = []
        with torch.no_grad():
            for episode in episodes:
                predicted_actions = policy(episode.states)
                penalty = action_penalty * (predicted_actions - episode.actions).square().sum(dim=-1)
                proxy_returns.append((episode.rewards - penalty).sum())
        returns = torch.stack(proxy_returns).float()
        std = returns.std(unbiased=False) if returns.numel() > 1 else torch.tensor(0.0)
        return ReturnSummary(
            mean_return=float(returns.mean().item()),
            std_return=float(std.item()),
            episode_count=len(episodes),
        )

    @staticmethod
    def delta_j(clean_return: ReturnSummary, noisy_return: ReturnSummary) -> float:
        return clean_return.mean_return - noisy_return.mean_return


ExpectedReturnEvaluator = UtilityEvaluator


@dataclass(frozen=True)
class RDPAccount:
    epsilon: float
    delta: float
    order: float
    steps: int


class PrivacyAuditor:
    """Monitor RDP composition across convergence steps."""

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
        self.history: list[RDPAccount] = []

    def update(self, steps: int = 1) -> None:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        self.steps += steps

    def observe_step(self, delta: float) -> RDPAccount:
        self.update(1)
        account = self.compute_epsilon(delta)
        self.history.append(account)
        return account

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

    def final_report(self, delta: float) -> dict[str, Any]:
        account = self.compute_epsilon(delta)
        return {
            "epsilon": account.epsilon,
            "delta": account.delta,
            "best_order": account.order,
            "steps": account.steps,
            "history": [asdict(item) for item in self.history],
        }


RDPAccountingTracker = PrivacyAuditor


@dataclass(frozen=True)
class MIASuccessSummary:
    report: MIAReport
    success_rate_delta: float


class MIAAnalyzer:
    """Quantify MIA success rate and margin gaps before/after unlearning."""

    @staticmethod
    def analyze(
        member_losses_before: Tensor,
        member_losses_after: Tensor,
        nonmember_losses: Tensor,
    ) -> dict[str, Any]:
        report = simulate_membership_inference(
            member_losses_before=member_losses_before,
            member_losses_after=member_losses_after,
            nonmember_losses=nonmember_losses,
        )
        return {
            "before_accuracy": report.before_accuracy,
            "after_accuracy": report.after_accuracy,
            "success_rate_delta": report.after_accuracy - report.before_accuracy,
            "before_gap": report.before_gap,
            "after_gap": report.after_gap,
            "gap_delta": report.after_gap - report.before_gap,
            "member_loss_before_mean": float(member_losses_before.mean().item()),
            "member_loss_after_mean": float(member_losses_after.mean().item()),
            "nonmember_loss_mean": float(nonmember_losses.mean().item()),
            "margin_distribution": {
                "before": (
                    nonmember_losses[: min(nonmember_losses.numel(), member_losses_before.numel())]
                    - member_losses_before[: min(nonmember_losses.numel(), member_losses_before.numel())]
                ).detach().cpu().tolist(),
                "after": (
                    nonmember_losses[: min(nonmember_losses.numel(), member_losses_after.numel())]
                    - member_losses_after[: min(nonmember_losses.numel(), member_losses_after.numel())]
                ).detach().cpu().tolist(),
            },
        }

    @staticmethod
    def evaluate(
        member_losses_before: Tensor,
        member_losses_after: Tensor,
        nonmember_losses: Tensor,
    ) -> MIASuccessSummary:
        report_dict = MIAAnalyzer.analyze(
            member_losses_before=member_losses_before,
            member_losses_after=member_losses_after,
            nonmember_losses=nonmember_losses,
        )
        report = MIAReport(
            before_accuracy=report_dict["before_accuracy"],
            after_accuracy=report_dict["after_accuracy"],
            before_gap=report_dict["before_gap"],
            after_gap=report_dict["after_gap"],
        )
        return MIASuccessSummary(
            report=report,
            success_rate_delta=report_dict["success_rate_delta"],
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


MIAEvaluator = MIAAnalyzer
