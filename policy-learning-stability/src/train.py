"""Training entry point for contextual Bayesian-SAC experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict

import gymnasium as gym

from bayesian_sac import BayesianSACConfig, ContextualBayesianSAC
from reward_scaling import DynamicRewardScalingConfig, DynamicRewardScalingWrapper


def build_environment(environment_id: str) -> DynamicRewardScalingWrapper:
    env = gym.make(environment_id)
    return DynamicRewardScalingWrapper(env, DynamicRewardScalingConfig())


def build_agent(env: gym.Env, context_dim: int) -> ContextualBayesianSAC:
    observation_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    config = BayesianSACConfig(
        observation_dim=observation_dim,
        action_dim=action_dim,
        context_dim=context_dim,
    )
    return ContextualBayesianSAC(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run contextual Bayesian-SAC stability experiments.")
    parser.add_argument("--env", default="HalfCheetah-v4", help="Gymnasium environment identifier.")
    parser.add_argument("--context-dim", type=int, default=8, help="Latent context dimensionality.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = build_environment(args.env)
    agent = build_agent(env, args.context_dim)
    print({"environment": args.env, "agent": asdict(agent.config)})


if __name__ == "__main__":
    main()
