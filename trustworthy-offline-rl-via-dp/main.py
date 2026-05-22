from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Sequence

import torch
from torch import Tensor

from models import PrivacyAwareCQL
from optimizer import PhaseClipNormSchedule, TrajectoryDPSGD
from unlearning import EpisodeShardManager, GaussianUpdateNoise, LiSSAInfluenceUnlearner
from utils import EpisodeBatch, choose_episode_batch, make_synthetic_episodes
from utils import seed_everything, simulate_membership_inference


def sum_transition_loss(model: PrivacyAwareCQL, episode: EpisodeBatch, noise_std: float) -> Tensor:
    # `trajectory_loss` averages for diagnostics. Multiplying by trajectory length
    # exposes the requested sum of transition contributions before DP clipping.
    terms = model.trajectory_loss(episode, noise_std=noise_std)
    return terms.total * episode.states.shape[0]


def train_dp_cql(
    model: PrivacyAwareCQL,
    episodes: Sequence[EpisodeBatch],
    steps: int,
    episode_batch_size: int,
    seed: int,
) -> TrajectoryDPSGD:
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
        sampling_rate=min(episode_batch_size / max(len(episodes), 1), 1.0),
    )
    for step in range(steps):
        sampled = choose_episode_batch(episodes, episode_batch_size, generator)
        losses = [
            sum_transition_loss(model, episode, noise_std=optimizer.noise_multiplier)
            for episode in sampled
        ]
        stats = optimizer.dp_step(losses)
        model.update_target()
        if step in {0, steps - 1}:
            print(
                f"step={step:02d} clip={stats.clip_norm:.2f} "
                f"raw_norm={stats.unclipped_norm_mean:.2f} "
                f"clipped={stats.clipped_fraction:.2f} noise={stats.noise_std:.3f}"
            )
    return optimizer


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
    episodes = make_synthetic_episodes(
        episode_count=args.episodes,
        horizon=args.horizon,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        generator=generator,
    )
    model = PrivacyAwareCQL(args.state_dim, args.action_dim)
    optimizer = train_dp_cql(
        model=model,
        episodes=episodes,
        steps=args.steps,
        episode_batch_size=args.episode_batch_size,
        seed=args.seed + 1,
    )
    deleted = episodes[args.delete_episode % len(episodes)]
    retained = [episode for episode in episodes if episode.episode_id != deleted.episode_id]
    shadow_nonmembers = make_synthetic_episodes(
        episode_count=max(4, len(retained) // 2),
        horizon=args.horizon,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        generator=generator,
    )
    before = episode_td_losses(model, [deleted, *retained[: len(shadow_nonmembers) - 1]])
    nonmembers = episode_td_losses(model, shadow_nonmembers)

    sharding = EpisodeShardManager(shard_count=args.shards)
    sharding.partition(episodes)
    affected = sharding.delete_episode(deleted.episode_id)
    apply_influence_unlearning(model, retained=retained, deleted=deleted)
    after = episode_td_losses(model, [deleted, *retained[: len(shadow_nonmembers) - 1]])
    report = simulate_membership_inference(before, after, nonmembers)
    budget = optimizer.accountant.budget(delta=1e-5)
    print(f"rdp_budget={asdict(budget)}")
    print(f"deleted_episode={deleted.episode_id} sisa_retrain_shards={sorted(affected)}")
    print(f"mia_report={asdict(report)}")


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
