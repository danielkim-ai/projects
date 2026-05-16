"""Ray orchestration for Bayesian-SAC under non-stationary reward scaling."""

from __future__ import annotations

import argparse
from typing import Any

import gymnasium as gym
import numpy as np
import ray
from ray.tune.registry import register_env

from src.agent import ContextualBayesianSAC
from src.env_wrapper import DynamicRewardScaleWrapper


def make_scaled_mujoco(config: dict[str, Any]) -> DynamicRewardScaleWrapper:
    """Registerable factory for HalfCheetah-v4 and Ant-v4 perturbation regimes."""

    env_name = config.get("env_name", "HalfCheetah-v4")
    perturbation = config.get("perturbation_type", "sinusoidal")
    return DynamicRewardScaleWrapper(gym.make(env_name), perturbation_type=perturbation)


@ray.remote
def run_episode(env_name: str, perturbation_type: str, state_dim: int, action_dim: int) -> dict[str, Any]:
    """Collect one lightweight episode and update the BO evidence stream."""

    env = make_scaled_mujoco({"env_name": env_name, "perturbation_type": perturbation_type})
    agent = ContextualBayesianSAC(state_dim=state_dim, action_dim=action_dim)
    observation, _ = env.reset()
    total_reward = 0.0
    alpha_trace: list[float] = []

    for _ in range(1000):
        state = np.asarray(observation, dtype=np.float32)
        action = env.action_space.sample()
        observation, reward, terminated, truncated, _ = env.step(action)
        alpha_trace.append(env.last_scale_state.alpha_t)
        total_reward += float(reward)

        # A production RLlib policy would replace this sampling action with the
        # actor output, then update the critics from a distributed replay buffer.
        if terminated or truncated:
            break

    # At episode end, reward-scale observations become GP evidence. The next
    # episode samples a temperature posterior conditioned on the updated alpha_t.
    agent.update_surrogate(alpha_trace, total_reward)
    next_temperature = agent.sample_optimal_temperature(float(np.mean(alpha_trace)))
    return {"return": total_reward, "temperature": next_temperature, "mean_alpha": float(np.mean(alpha_trace))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bayesian-SAC reward-scaling experiments with Ray.")
    parser.add_argument("--env", choices=["HalfCheetah-v4", "Ant-v4"], default="HalfCheetah-v4")
    parser.add_argument("--perturbation", choices=["sinusoidal", "abrupt_collapse"], default="sinusoidal")
    parser.add_argument("--episodes", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ray.init(ignore_reinit_error=True)
    register_env("scaled_mujoco", make_scaled_mujoco)

    probe_env = make_scaled_mujoco({"env_name": args.env, "perturbation_type": args.perturbation})
    state_dim = int(np.prod(probe_env.observation_space.shape))
    action_dim = int(np.prod(probe_env.action_space.shape))
    probe_env.close()

    futures = [
        run_episode.remote(args.env, args.perturbation, state_dim, action_dim)
        for _ in range(args.episodes)
    ]
    results = ray.get(futures)
    for index, result in enumerate(results, start=1):
        print(f"[Episode {index}] return={result['return']:.3f} alpha={result['mean_alpha']:.3f} temp={result['temperature']:.3f}")


if __name__ == "__main__":
    main()
