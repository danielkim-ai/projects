from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

from datasets import RealWorldTrajectoryLoader, SISAShardIndex, TrajectoryDataLoader, TrajectoryDataset
from evaluation import MIAAnalyzer, PrivacyAuditor, UtilityEvaluator
from models import OfflineRLAlgorithm, PrivacyAwareCQL, PrivacyAwareIQL
from optimizer import DPStepStats, PhaseClipNormSchedule, TrajectoryDPSGD
from unlearning import GaussianUpdateNoise, LiSSAInfluenceUnlearner
from utils import EpisodeBatch, seed_everything


METRICS_PATH = Path("results") / "plots" / "latest_visualise_metrics.json"


def sum_transition_loss(model: OfflineRLAlgorithm, episode: EpisodeBatch, noise_std: float) -> Tensor:
    # `trajectory_loss` averages for diagnostics. Multiplying by trajectory length
    # exposes the requested sum of transition contributions before DP clipping.
    terms = model.trajectory_loss(episode, noise_std=noise_std)
    return terms.total * episode.states.shape[0]


def train_dp_cql(
    model: OfflineRLAlgorithm,
    dataset: TrajectoryDataset,
    steps: int,
    episode_batch_size: int,
    seed: int,
    noise_multiplier: float,
    clip_schedule: str,
    early_clip_norm: float,
    late_clip_norm: float,
    static_clip_norm: float,
) -> tuple[TrajectoryDPSGD, PrivacyAuditor, list[DPStepStats], list[float]]:
    generator = seed_everything(seed)
    if clip_schedule == "static":
        schedule = PhaseClipNormSchedule(
            warmup_steps=steps + 1,
            early_norm=static_clip_norm,
            late_norm=static_clip_norm,
        )
    else:
        schedule = PhaseClipNormSchedule(
            warmup_steps=max(steps // 2, 1),
            early_norm=early_clip_norm,
            late_norm=late_clip_norm,
        )
    optimizer = TrajectoryDPSGD(
        model.parameters(),
        lr=2e-3,
        clip_schedule=schedule,
        noise_multiplier=noise_multiplier,
        sampling_rate=min(episode_batch_size / max(len(dataset), 1), 1.0),
    )
    auditor = PrivacyAuditor(
        noise_multiplier=optimizer.noise_multiplier,
        sampling_rate=min(episode_batch_size / max(len(dataset), 1), 1.0),
    )
    loader = TrajectoryDataLoader(
        dataset=dataset,
        batch_size=episode_batch_size,
        shuffle=True,
        generator=generator,
    )
    iterator = iter(loader)
    step_stats: list[DPStepStats] = []
    epsilons: list[float] = []
    for step in range(steps):
        try:
            sampled = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            sampled = next(iterator)
        losses = [
            sum_transition_loss(model, episode, noise_std=optimizer.noise_multiplier)
            for episode in sampled
        ]
        stats = optimizer.dp_step(losses)
        account = auditor.observe_step(delta=1e-5)
        step_stats.append(stats)
        epsilons.append(account.epsilon)
        model.update_target()
        if step in {0, steps - 1}:
            print(
                f"step={step:02d} clip={stats.clip_norm:.2f} "
                f"raw_norm={stats.unclipped_norm_mean:.2f} "
                f"clipped={stats.clipped_fraction:.2f} noise={stats.noise_std:.3f}"
            )
    return optimizer, auditor, step_stats, epsilons


def episode_td_losses(model: OfflineRLAlgorithm, episodes: Sequence[EpisodeBatch]) -> Tensor:
    with torch.enable_grad():
        return model.loss_vector(episodes).detach()


def apply_influence_unlearning(
    model: OfflineRLAlgorithm,
    retained: Sequence[EpisodeBatch],
    deleted: EpisodeBatch,
) -> None:
    def train_loss() -> Tensor:
        return torch.stack(
            [sum_transition_loss(model, episode, noise_std=0.0) for episode in retained]
        ).mean()

    def deleted_loss() -> Tensor:
        return sum_transition_loss(model, deleted, noise_std=0.0)

    unlearner = LiSSAInfluenceUnlearner(
        parameters=model.parameters(),
        train_loss=train_loss,
        deleted_loss=deleted_loss,
        noise=GaussianUpdateNoise(std=2e-4),
    )
    unlearner.forget()


def build_dataset(args: argparse.Namespace, generator: torch.Generator, episode_count: int) -> TrajectoryDataset:
    if args.domain == "synthetic":
        return TrajectoryDataset.synthetic(
            episode_count=episode_count,
            horizon=args.horizon,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            generator=generator,
            shard_count=args.shards,
        )
    loader = RealWorldTrajectoryLoader(
        domain=args.domain,
        data_path=args.data_path,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        horizon=args.horizon,
        episode_count=episode_count,
        shard_count=args.shards,
        generator=generator,
    )
    return loader.load()


def build_model(args: argparse.Namespace) -> OfflineRLAlgorithm:
    if args.algo == "iql":
        return PrivacyAwareIQL(args.state_dim, args.action_dim, hidden_dim=args.hidden_dim)
    return PrivacyAwareCQL(args.state_dim, args.action_dim, hidden_dim=args.hidden_dim)


def metrics_path_for_run(args: argparse.Namespace) -> Path:
    if args.write_metrics != METRICS_PATH:
        return args.write_metrics
    if args.domain == "synthetic" and args.algo == "cql":
        return METRICS_PATH
    return Path("results") / "plots" / f"{args.domain}_{args.algo}_metrics.json"


def write_visualise_metrics(
    path: Path,
    *,
    args: argparse.Namespace,
    step_stats: Sequence[DPStepStats],
    epsilons: Sequence[float],
    deleted_episode: int,
    affected_shards: Sequence[int],
    utility_report: dict[str, Any],
    mia_report: dict[str, Any],
    rdp_budget: dict[str, Any],
    optimizer_rdp_budget: dict[str, Any],
) -> Path:
    margin_distribution = mia_report["margin_distribution"]
    record = {
        "schema_version": 1,
        "source": "main.py",
        "config": {
            "steps": args.steps,
            "episodes": args.episodes,
            "horizon": args.horizon,
            "state_dim": args.state_dim,
            "action_dim": args.action_dim,
            "episode_batch_size": args.episode_batch_size,
            "delete_episode": args.delete_episode,
            "shards": args.shards,
            "seed": args.seed,
            "noise_multiplier": args.noise_multiplier,
            "write_metrics": str(args.write_metrics),
            "hidden_dim": args.hidden_dim,
            "clip_schedule": args.clip_schedule,
            "early_clip_norm": args.early_clip_norm,
            "late_clip_norm": args.late_clip_norm,
            "static_clip_norm": args.static_clip_norm,
            "domain": args.domain,
            "algo": args.algo,
            "data_path": None if args.data_path is None else str(args.data_path),
        },
        "steps": list(range(len(step_stats))),
        "clip_norms": [stats.clip_norm for stats in step_stats],
        "epsilons": [float(epsilon) for epsilon in epsilons],
        "raw_norms": [stats.unclipped_norm_mean for stats in step_stats],
        "clipped_fractions": [stats.clipped_fraction for stats in step_stats],
        "noise_std": [stats.noise_std for stats in step_stats],
        "trajectory_counts": [stats.trajectory_count for stats in step_stats],
        "before_margins": margin_distribution["before"],
        "after_margins": margin_distribution["after"],
        "logged_return": {
            "mean": utility_report["logged_return"]["mean_return"],
            "std": utility_report["logged_return"]["std_return"],
            "episode_count": utility_report["logged_return"]["episode_count"],
        },
        "proxy_return": {
            "mean": utility_report["policy_proxy_return"]["mean_return"],
            "std": utility_report["policy_proxy_return"]["std_return"],
            "episode_count": utility_report["policy_proxy_return"]["episode_count"],
        },
        "delta_j": utility_report["delta_j"],
        "mia_report": mia_report,
        "rdp_budget": rdp_budget,
        "optimizer_rdp_budget": optimizer_rdp_budget,
        "deleted_episode": deleted_episode,
        "affected_shards": list(affected_shards),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def run_demo(args: argparse.Namespace) -> None:
    generator = seed_everything(args.seed)
    metrics_path = metrics_path_for_run(args)
    dataset = build_dataset(args, generator=generator, episode_count=args.episodes)
    episodes = [dataset[index] for index in range(len(dataset))]
    model = build_model(args)
    optimizer, auditor, step_stats, epsilons = train_dp_cql(
        model=model,
        dataset=dataset,
        steps=args.steps,
        episode_batch_size=args.episode_batch_size,
        seed=args.seed + 1,
        noise_multiplier=args.noise_multiplier,
        clip_schedule=args.clip_schedule,
        early_clip_norm=args.early_clip_norm,
        late_clip_norm=args.late_clip_norm,
        static_clip_norm=args.static_clip_norm,
    )
    deleted = dataset.get_episode(args.delete_episode % len(dataset))
    retained_dataset = dataset.without_episode(deleted.episode_id)
    retained = [retained_dataset[index] for index in range(len(retained_dataset))]
    shadow_dataset = build_dataset(
        args,
        generator=generator,
        episode_count=max(4, len(retained) // 2),
    )
    shadow_nonmembers = [shadow_dataset[index] for index in range(len(shadow_dataset))]
    before = episode_td_losses(model, [deleted, *retained[: len(shadow_nonmembers) - 1]])
    nonmembers = episode_td_losses(model, shadow_nonmembers)

    sharding = SISAShardIndex(dataset=dataset, shard_count=args.shards)
    affected = sharding.affected_shards([deleted.episode_id])
    logged_return = UtilityEvaluator.logged_return(episodes)
    proxy_return = UtilityEvaluator.monte_carlo_weight_noisy_proxy_return(
        model.policy,
        episodes,
        weight_noise_std=0.04 * args.noise_multiplier,
        generator=seed_everything(args.seed + 2),
    )
    apply_influence_unlearning(model, retained=retained, deleted=deleted)
    after = episode_td_losses(model, [deleted, *retained[: len(shadow_nonmembers) - 1]])
    report = MIAAnalyzer.analyze(before, after, nonmembers)
    budget = auditor.compute_epsilon(delta=1e-5)
    optimizer_budget = optimizer.accountant.budget(delta=1e-5)
    utility_report = {
        "logged_return": asdict(logged_return),
        "policy_proxy_return": asdict(proxy_return),
        "delta_j": UtilityEvaluator.delta_j(logged_return, proxy_return),
    }
    written_metrics_path = write_visualise_metrics(
        metrics_path,
        args=args,
        step_stats=step_stats,
        epsilons=epsilons,
        deleted_episode=deleted.episode_id,
        affected_shards=sorted(affected),
        utility_report=utility_report,
        mia_report=report,
        rdp_budget=asdict(budget),
        optimizer_rdp_budget=asdict(optimizer_budget),
    )
    print(f"rdp_budget={asdict(budget)}")
    print(f"optimizer_rdp_budget={asdict(optimizer_budget)}")
    print(f"deleted_episode={deleted.episode_id} sisa_retrain_shards={sorted(affected)}")
    print(f"utility_report={utility_report}")
    print(f"mia_report={report}")
    print(f"metrics_json={written_metrics_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trajectory-DP CQL and unlearning simulation for offline RL."
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--domain", choices=("synthetic", "medical", "financial"), default="synthetic")
    parser.add_argument("--algo", choices=("cql", "iql"), default="cql")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=18)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--state-dim", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--episode-batch-size", type=int, default=5)
    parser.add_argument("--delete-episode", type=int, default=3)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-multiplier", type=float, default=0.65)
    parser.add_argument("--write-metrics", type=Path, default=METRICS_PATH)
    parser.add_argument("--clip-schedule", choices=("adaptive", "static"), default="adaptive")
    parser.add_argument("--early-clip-norm", type=float, default=1.3)
    parser.add_argument("--late-clip-norm", type=float, default=0.8)
    parser.add_argument("--static-clip-norm", type=float, default=1.0)
    return parser


if __name__ == "__main__":
    run_demo(build_parser().parse_args())
