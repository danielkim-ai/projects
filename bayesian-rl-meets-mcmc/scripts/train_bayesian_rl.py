"""Train a Bayesian VAC agent with periodic SGLD recalibration on MuJoCo."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import optim

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_recalibration import HybridRecalibrationConfig, HybridRecalibrator  # noqa: E402
from src.sgld import SGLDConfig  # noqa: E402
from src.vac import VACConfig, VariationalActorCritic  # noqa: E402


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    episode_return: float
    episode_length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bayesian VAC + Preconditioned SGLD trainer.")
    parser.add_argument("--env-id", default="HalfCheetah-v4")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--vi-epochs", type=int, default=4)
    parser.add_argument("--recalibrate-every", type=int, default=5)
    parser.add_argument("--ess-floor", type=float, default=8.0)
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "results" / "tensorboard")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def scale_action(action: np.ndarray, env: Any) -> np.ndarray:
    low = env.action_space.low
    high = env.action_space.high
    return low + 0.5 * (action + 1.0) * (high - low)


def collect_rollout(
    env: Any,
    model: VariationalActorCritic,
    device: torch.device,
    steps: int,
    gamma: float,
    gae_lambda: float,
) -> RolloutBatch:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[float] = []
    values: list[float] = []

    obs, _ = env.reset()
    episode_return = 0.0
    episode_length = 0

    for _ in range(steps):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            value = model.value(obs_tensor).squeeze(0)
            policy_action, _ = model.act(obs_tensor)
        action_np = policy_action.squeeze(0).cpu().numpy()
        next_obs, reward, terminated, truncated, _ = env.step(scale_action(action_np, env))
        done = terminated or truncated

        observations.append(obs)
        actions.append(action_np)
        rewards.append(float(reward))
        dones.append(float(done))
        values.append(float(value.cpu()))
        episode_return += float(reward)
        episode_length += 1

        obs = next_obs
        if done:
            obs, _ = env.reset()

    with torch.no_grad():
        next_value = float(
            model.value(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).cpu()
        )

    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_nonterminal = 1.0 - dones[t]
        next_val = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae

    returns = advantages + np.asarray(values, dtype=np.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)

    return RolloutBatch(
        observations=torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=device),
        actions=torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=device),
        returns=torch.as_tensor(returns, dtype=torch.float32, device=device),
        advantages=torch.as_tensor(advantages, dtype=torch.float32, device=device),
        episode_return=episode_return,
        episode_length=episode_length,
    )


def main() -> None:
    args = parse_args()
    import gymnasium as gym
    from torch.utils.tensorboard import SummaryWriter

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    env = gym.make(args.env_id)
    env.action_space.seed(args.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    model = VariationalActorCritic(
        VACConfig(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=args.hidden_dim)
    ).to(device)
    optimiser = optim.Adam(model.parameters(), lr=args.lr)
    recalibrator = HybridRecalibrator(
        model,
        HybridRecalibrationConfig(
            recalibrate_every_episodes=args.recalibrate_every,
            ess_floor=args.ess_floor,
            sgld=SGLDConfig(step_size=args.lr * 0.1, prior_precision=1.0e-2),
        ),
    )

    writer = SummaryWriter(args.log_dir / args.env_id)
    latest_metrics: dict[str, torch.Tensor] | None = None

    for episode in range(1, args.episodes + 1):
        if episode > 1:
            recalibrator.sgld.load_posterior_sample(model, recalibrator.sample_policy_state())
        batch = collect_rollout(env, model, device, args.rollout_steps, args.gamma, args.gae_lambda)

        for _ in range(args.vi_epochs):
            optimiser.zero_grad(set_to_none=True)
            metrics = model.elbo_loss(batch.observations, batch.actions, batch.returns, batch.advantages)
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimiser.step()
            latest_metrics = metrics

        assert latest_metrics is not None
        kl_value = float(latest_metrics["kl"].cpu())
        beta_value = model.update_beta_kl(kl_value)
        ess_value = recalibrator.update_episode_score(batch.episode_return)

        if recalibrator.should_recalibrate(episode, kl_value, ess_value):
            force_warmup = recalibrator.needs_warmup(ess_value)

            def closure() -> torch.Tensor:
                return model.elbo_loss(
                    batch.observations,
                    batch.actions,
                    batch.returns,
                    batch.advantages,
                )["loss"]

            mcmc_losses = recalibrator.recalibrate(closure, force_warmup=force_warmup)
            writer.add_scalar("mcmc/recalibration_loss", float(np.mean(mcmc_losses)), episode)
            writer.add_scalar("mcmc/forced_warmup", float(force_warmup), episode)

        writer.add_scalar("rollout/return", batch.episode_return, episode)
        writer.add_scalar("rollout/length", batch.episode_length, episode)
        writer.add_scalar("vac/loss", float(latest_metrics["loss"].detach().cpu()), episode)
        writer.add_scalar("vac/actor_loss", float(latest_metrics["actor_loss"].cpu()), episode)
        writer.add_scalar("vac/critic_loss", float(latest_metrics["critic_loss"].cpu()), episode)
        writer.add_scalar("vac/kl", kl_value, episode)
        writer.add_scalar("vac/beta_kl", beta_value, episode)
        writer.add_scalar("mcmc/ess", ess_value, episode)

        print(
            f"episode={episode} return={batch.episode_return:.2f} "
            f"kl={kl_value:.4f} beta={beta_value:.5f} ess={ess_value:.2f}"
        )

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
