from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Sequence

import torch
from torch import Tensor

from datasets import SISAShardIndex, TrajectoryDataLoader, TrajectoryDataset
from evaluation import MIAAnalyzer, PrivacyAuditor, UtilityEvaluator
from models import PrivacyAwareCQL
from optimizer import PhaseClipNormSchedule, TrajectoryDPSGD
from unlearning import GaussianUpdateNoise, LiSSAInfluenceUnlearner
from utils import EpisodeBatch, seed_everything


def sum_transition_loss(model: PrivacyAwareCQL, episode: EpisodeBatch, noise_std: float) -> Tensor:
    # `trajectory_loss` averages for diagnostics. Multiplying by trajectory length
    # exposes the requested sum of transition contributions before DP clipping.
    terms = model.trajectory_loss(episode, noise_std=noise_std)
    return terms.total * episode.states.shape[0]


def train_dp_cql(
    model: PrivacyAwareCQL,
    dataset: TrajectoryDataset,
    steps: int,
    episode_batch_size: int,
    seed: int,
) -> tuple[TrajectoryDPSGD, PrivacyAuditor]:
    generator = seed_everything(seed)
    optimizer = TrajectoryDPSGD(
        model.parameters(),
        lr=2e-3,
        clip_schedule=PhaseClipNormSchedule(
            warmup_steps=max(steps // 2, 1),
            early_norm=1.3,
            late_norm=0.8,
        ),
        noise_multiplier=0.65,
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
        auditor.observe_step(delta=1e-5)
        model.update_target()
        if step in {0, steps - 1}:
            print(
                f"step={step:02d} clip={stats.clip_norm:.2f} "
                f"raw_norm={stats.unclipped_norm_mean:.2f} "
                f"clipped={stats.clipped_fraction:.2f} noise={stats.noise_std:.3f}"
            )
    return optimizer, auditor


def episode_td_losses(model: PrivacyAwareCQL, episodes: Sequence[EpisodeBatch]) -> Tensor:
    with torch.enable_grad():
        return model.loss_vector(episodes).detach()


def apply_influence_unlearning(
    model: PrivacyAwareCQL,
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


def run_demo(args: argparse.Namespace) -> None:
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
    optimizer, auditor = train_dp_cql(
        model=model,
        dataset=dataset,
        steps=args.steps,
        episode_batch_size=args.episode_batch_size,
        seed=args.seed + 1,
    )
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
    before = episode_td_losses(model, [deleted, *retained[: len(shadow_nonmembers) - 1]])
    nonmembers = episode_td_losses(model, shadow_nonmembers)

    sharding = SISAShardIndex(dataset=dataset, shard_count=args.shards)
    affected = sharding.affected_shards([deleted.episode_id])
    logged_return = UtilityEvaluator.logged_return(episodes)
    proxy_return = UtilityEvaluator.monte_carlo_proxy_return(model.policy, episodes)
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
    print(f"rdp_budget={asdict(budget)}")
    print(f"optimizer_rdp_budget={asdict(optimizer_budget)}")
    print(f"deleted_episode={deleted.episode_id} sisa_retrain_shards={sorted(affected)}")
    print(f"utility_report={utility_report}")
    print(f"mia_report={report}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trajectory-DP CQL and unlearning simulation for offline RL."
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=18)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--episode-batch-size", type=int, default=5)
    parser.add_argument("--delete-episode", type=int, default=3)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    return parser


if __name__ == "__main__":
    run_demo(build_parser().parse_args())
