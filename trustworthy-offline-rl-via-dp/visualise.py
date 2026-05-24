from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import SISAShardIndex, TrajectoryDataLoader, TrajectoryDataset
from evaluation import MIAAnalyzer, UtilityEvaluator
from main import apply_influence_unlearning, episode_td_losses, sum_transition_loss
from models import PrivacyAwareCQL
from optimizer import DPStepStats, PhaseClipNormSchedule, TrajectoryDPSGD
from utils import EpisodeBatch, seed_everything


PLOT_DIR = Path("results") / "plots"


@dataclass(frozen=True)
class DynamicMetrics:
    steps: list[int]
    clip_norms: list[float]
    epsilons: list[float]
    raw_norms: list[float]
    clipped_fractions: list[float]
    before_margins: np.ndarray
    after_margins: np.ndarray
    logged_return_mean: float
    logged_return_std: float
    proxy_return_mean: float
    proxy_return_std: float
    delta_j: float
    mia_report: dict[str, object]
    deleted_episode: int
    affected_shards: list[int]


def metrics_to_record(metrics: DynamicMetrics) -> dict[str, Any]:
    return {
        "steps": metrics.steps,
        "clip_norms": metrics.clip_norms,
        "epsilons": metrics.epsilons,
        "raw_norms": metrics.raw_norms,
        "clipped_fractions": metrics.clipped_fractions,
        "before_margins": metrics.before_margins.tolist(),
        "after_margins": metrics.after_margins.tolist(),
        "logged_return": {
            "mean": metrics.logged_return_mean,
            "std": metrics.logged_return_std,
        },
        "proxy_return": {
            "mean": metrics.proxy_return_mean,
            "std": metrics.proxy_return_std,
        },
        "delta_j": metrics.delta_j,
        "mia_report": metrics.mia_report,
        "deleted_episode": metrics.deleted_episode,
        "affected_shards": metrics.affected_shards,
    }


def metrics_from_record(record: dict[str, Any]) -> DynamicMetrics:
    return DynamicMetrics(
        steps=[int(item) for item in record["steps"]],
        clip_norms=[float(item) for item in record["clip_norms"]],
        epsilons=[float(item) for item in record["epsilons"]],
        raw_norms=[float(item) for item in record["raw_norms"]],
        clipped_fractions=[float(item) for item in record["clipped_fractions"]],
        before_margins=np.asarray(record["before_margins"], dtype=float),
        after_margins=np.asarray(record["after_margins"], dtype=float),
        logged_return_mean=float(record["logged_return"]["mean"]),
        logged_return_std=float(record["logged_return"]["std"]),
        proxy_return_mean=float(record["proxy_return"]["mean"]),
        proxy_return_std=float(record["proxy_return"]["std"]),
        delta_j=float(record["delta_j"]),
        mia_report=dict(record["mia_report"]),
        deleted_episode=int(record["deleted_episode"]),
        affected_shards=[int(item) for item in record["affected_shards"]],
    )


def write_metrics(metrics: DynamicMetrics, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics_to_record(metrics), indent=2), encoding="utf-8")
    return path


def read_metrics(path: Path) -> DynamicMetrics:
    return metrics_from_record(json.loads(path.read_text(encoding="utf-8")))


def gaussian_kde(values: np.ndarray, grid: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    if bandwidth is None:
        std = np.std(values, ddof=1) if len(values) > 1 else 1.0
        bandwidth = 1.06 * std * max(len(values), 1) ** (-1 / 5)
        bandwidth = max(float(bandwidth), 0.2)
    scaled = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * scaled**2).sum(axis=1)
    density /= len(values) * bandwidth * np.sqrt(2 * np.pi)
    return density


def _cycle_next(
    iterator: object,
    loader: TrajectoryDataLoader,
) -> tuple[list[EpisodeBatch], object]:
    try:
        return next(iterator), iterator  # type: ignore[arg-type]
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def run_dynamic_pipeline(args: argparse.Namespace) -> DynamicMetrics:
    generator = seed_everything(args.seed)
    dataset = TrajectoryDataset.synthetic(
        episode_count=args.episodes,
        horizon=args.horizon,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        generator=generator,
        shard_count=args.shards,
    )
    episodes = [dataset[index] for index in range(len(dataset))]
    model = PrivacyAwareCQL(args.state_dim, args.action_dim)
    optimizer = TrajectoryDPSGD(
        model.parameters(),
        lr=2e-3,
        clip_schedule=PhaseClipNormSchedule(
            warmup_steps=max(args.steps // 2, 1),
            early_norm=1.3,
            late_norm=0.8,
        ),
        noise_multiplier=0.65,
        sampling_rate=min(args.episode_batch_size / max(len(dataset), 1), 1.0),
    )
    loader = TrajectoryDataLoader(
        dataset=dataset,
        batch_size=args.episode_batch_size,
        shuffle=True,
        generator=seed_everything(args.seed + 1),
    )
    iterator = iter(loader)
    step_stats: list[DPStepStats] = []
    epsilons: list[float] = []

    for _step in range(args.steps):
        sampled, iterator = _cycle_next(iterator, loader)
        losses = [
            sum_transition_loss(model, episode, noise_std=optimizer.noise_multiplier)
            for episode in sampled
        ]
        stats = optimizer.dp_step(losses)
        model.update_target()
        step_stats.append(stats)
        epsilons.append(optimizer.accountant.budget(delta=args.delta).epsilon)

    deleted = dataset.get_episode(args.delete_episode % len(dataset))
    retained_dataset = dataset.without_episode(deleted.episode_id)
    retained = [retained_dataset[index] for index in range(len(retained_dataset))]
    shadow_dataset = TrajectoryDataset.synthetic(
        episode_count=max(4, len(retained) // 2),
        horizon=args.horizon,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        generator=generator,
    )
    shadow_nonmembers = [shadow_dataset[index] for index in range(len(shadow_dataset))]
    before_losses = episode_td_losses(model, [deleted, *retained[: len(shadow_nonmembers) - 1]])
    nonmember_losses = episode_td_losses(model, shadow_nonmembers)
    logged_return = UtilityEvaluator.logged_return(episodes)
    proxy_return = UtilityEvaluator.monte_carlo_proxy_return(model.policy, episodes)
    sharding = SISAShardIndex(dataset=dataset, shard_count=args.shards)
    affected = sorted(sharding.affected_shards([deleted.episode_id]))

    apply_influence_unlearning(model, retained=retained, deleted=deleted)
    after_losses = episode_td_losses(model, [deleted, *retained[: len(shadow_nonmembers) - 1]])
    mia_report = MIAAnalyzer.analyze(before_losses, after_losses, nonmember_losses)
    before_margins = np.asarray(mia_report["margin_distribution"]["before"], dtype=float)  # type: ignore[index]
    after_margins = np.asarray(mia_report["margin_distribution"]["after"], dtype=float)  # type: ignore[index]

    return DynamicMetrics(
        steps=list(range(args.steps)),
        clip_norms=[item.clip_norm for item in step_stats],
        epsilons=epsilons,
        raw_norms=[item.unclipped_norm_mean for item in step_stats],
        clipped_fractions=[item.clipped_fraction for item in step_stats],
        before_margins=before_margins,
        after_margins=after_margins,
        logged_return_mean=logged_return.mean_return,
        logged_return_std=logged_return.std_return,
        proxy_return_mean=proxy_return.mean_return,
        proxy_return_std=proxy_return.std_return,
        delta_j=UtilityEvaluator.delta_j(logged_return, proxy_return),
        mia_report=mia_report,
        deleted_episode=deleted.episode_id,
        affected_shards=affected,
    )


def plot_privacy_utility(metrics: DynamicMetrics) -> Path:
    steps = np.asarray(metrics.steps)
    fig, ax_clip = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    ax_eps = ax_clip.twinx()

    ax_clip.step(steps, metrics.clip_norms, where="post", color="#2f6f9f", linewidth=2.5, label="Clip norm C")
    ax_clip.scatter(steps, metrics.clip_norms, color="#2f6f9f", s=18, zorder=3)
    ax_eps.plot(steps, metrics.epsilons, color="#b4493a", linewidth=2.5, marker="o", label="RDP epsilon")

    ax_clip.set_title("Dynamic Trajectory DP-SGD Privacy-Utility Schedule", pad=12)
    ax_clip.set_xlabel("Training step")
    ax_clip.set_ylabel("Clipping norm C", color="#2f6f9f")
    ax_eps.set_ylabel("Cumulative epsilon", color="#b4493a")
    ax_clip.set_xticks(steps)
    ax_clip.grid(True, alpha=0.25)

    lines = ax_clip.get_lines() + ax_eps.get_lines()
    labels = [line.get_label() for line in lines]
    ax_clip.legend(lines, labels, loc="center right", frameon=False)
    fig.tight_layout()
    path = PLOT_DIR / "plot_privacy_utility.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_unlearning_margin(metrics: DynamicMetrics) -> Path:
    before = metrics.before_margins
    after = metrics.after_margins
    lower = min(before.min(), after.min()) - 0.8
    upper = max(before.max(), after.max()) + 0.8
    grid = np.linspace(lower, upper, 240)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    bins = np.linspace(lower, upper, 14)
    ax.hist(before, bins=bins, density=True, alpha=0.28, color="#4c78a8", label="Before histogram")
    ax.hist(after, bins=bins, density=True, alpha=0.28, color="#f58518", label="After histogram")
    ax.plot(grid, gaussian_kde(before, grid), color="#245a8d", linewidth=2.5, label="Before KDE")
    ax.plot(grid, gaussian_kde(after, grid), color="#c15d00", linewidth=2.5, label="After KDE")
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.2, label="Decision boundary")
    ax.set_title("Dynamic MIA Margin Distribution Before and After Unlearning", pad=12)
    ax.set_xlabel("Margin: non-member loss - member loss")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path = PLOT_DIR / "plot_unlearning_margin.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_utility_tradeoff(metrics: DynamicMetrics) -> Path:
    labels = ["Logged data", "DP policy proxy"]
    means = np.array([metrics.logged_return_mean, metrics.proxy_return_mean])
    stds = np.array([metrics.logged_return_std, metrics.proxy_return_std])

    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=180)
    bars = ax.bar(labels, means, yerr=stds, capsize=8, color=["#59a14f", "#b07aa1"], alpha=0.86)
    ax.axhline(0.0, color="#333333", linewidth=0.9)
    y_mid = float(means.mean())
    ax.annotate(
        rf"$\Delta J = {metrics.delta_j:.2f}$",
        xy=(0.5, y_mid),
        xytext=(0.5, y_mid + 4.0),
        arrowprops={"arrowstyle": "<->", "color": "#333333", "lw": 1.4},
        ha="center",
        va="center",
        fontsize=11,
    )
    for bar, mean in zip(bars, means):
        vertical_alignment = "top" if mean < 0 else "bottom"
        offset = -0.65 if mean < 0 else 0.65
        ax.text(bar.get_x() + bar.get_width() / 2, mean + offset, f"{mean:.2f}", ha="center", va=vertical_alignment)
    ax.set_title("Dynamic Utility Trade-off Under DP Weight Noise", pad=12)
    ax.set_ylabel("Expected return proxy")
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(0.5, ax.get_ylim()[0] + 0.6, "Error bars show episode-level standard deviation.", ha="center", fontsize=9)
    fig.tight_layout()
    path = PLOT_DIR / "plot_utility_tradeoff.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dynamic offline-RL audit simulation and render plots.")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=36)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--episode-batch-size", type=int, default=8)
    parser.add_argument("--delete-episode", type=int, default=11)
    parser.add_argument("--shards", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Load an existing metrics JSON file instead of running a new simulation.",
    )
    parser.add_argument(
        "--write-metrics",
        type=Path,
        default=PLOT_DIR / "latest_visualise_metrics.json",
        help="Path for the metrics JSON produced by a fresh simulation.",
    )
    return parser


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    args = build_parser().parse_args()
    if args.metrics_json is not None:
        metrics = read_metrics(args.metrics_json)
        metrics_path = args.metrics_json
        source = "json"
    else:
        metrics = run_dynamic_pipeline(args)
        metrics_path = write_metrics(metrics, args.write_metrics)
        source = "simulation"
    paths = [
        plot_privacy_utility(metrics),
        plot_unlearning_margin(metrics),
        plot_utility_tradeoff(metrics),
    ]
    summary = {
        "deleted_episode": metrics.deleted_episode,
        "affected_shards": metrics.affected_shards,
        "final_epsilon": metrics.epsilons[-1],
        "delta_j": metrics.delta_j,
        "mia_report": metrics.mia_report,
        "metrics_source": source,
        "metrics_json": str(metrics_path),
    }
    print(f"dynamic_summary={summary}")
    print(f"metrics_json={metrics_path}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
