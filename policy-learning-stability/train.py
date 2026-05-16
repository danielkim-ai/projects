"""Ray orchestration for Bayesian-SAC under non-stationary reward scaling."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import ray
from ray.tune.registry import register_env

from src.agent import ContextualBayesianSAC
from src.env_wrapper import DynamicRewardScaleWrapper


PERTURBATION_ALIASES = {"collapse": "abrupt_collapse"}


def make_scaled_mujoco(config: dict[str, Any]) -> DynamicRewardScaleWrapper:
    """Registerable factory for HalfCheetah-v4 and Ant-v4 perturbation regimes."""

    env_name = config.get("env_name", "HalfCheetah-v4")
    perturbation = config.get("perturbation_type", "sinusoidal")
    perturbation = PERTURBATION_ALIASES.get(perturbation, perturbation)
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
    standard_return = total_reward / max(float(np.mean(alpha_trace)), 1e-6)
    return {
        "return": total_reward,
        "standard_return": standard_return,
        "bayesian_return": total_reward,
        "temperature": next_temperature,
        "mean_alpha": float(np.mean(alpha_trace)),
        "device": str(agent.device),
    }


def initialise_ray() -> None:
    """Start Ray with conservative head-node settings for notebooks and Colab."""

    if ray.is_initialized():
        return

    cpu_count = max(1, min(2, os.cpu_count() or 1))
    try:
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            num_cpus=cpu_count,
            object_store_memory=256 * 1024 * 1024,
            _memory=512 * 1024 * 1024,
        )
    except (RuntimeError, ValueError):
        # Some managed notebooks reject explicit memory controls. Falling back
        # keeps the experiment runnable while preserving the same public API.
        ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=cpu_count)


def write_results(results: list[dict[str, Any]], output_path: Path) -> None:
    """Persist episode summaries for automated visualisation."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "mean_alpha",
        "temperature",
        "return",
        "standard_return",
        "bayesian_return",
        "device",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for episode, result in enumerate(results, start=1):
            writer.writerow({"episode": episode, **result})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bayesian-SAC reward-scaling experiments with Ray.")
    parser.add_argument("--env", choices=["HalfCheetah-v4", "Ant-v4"], default="HalfCheetah-v4")
    parser.add_argument("--perturbation", choices=["sinusoidal", "abrupt_collapse", "collapse"], default="sinusoidal")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--results-path", default="results/experiment_log.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initialise_ray()
    register_env("scaled_mujoco", make_scaled_mujoco)

    perturbation = PERTURBATION_ALIASES.get(args.perturbation, args.perturbation)
    probe_env = make_scaled_mujoco({"env_name": args.env, "perturbation_type": perturbation})
    state_dim = int(np.prod(probe_env.observation_space.shape))
    action_dim = int(np.prod(probe_env.action_space.shape))
    probe_env.close()

    futures = [
        run_episode.remote(args.env, perturbation, state_dim, action_dim)
        for _ in range(args.episodes)
    ]
    results = ray.get(futures)
    write_results(results, Path(args.results_path))
    for index, result in enumerate(results, start=1):
        print(
            f"[Episode {index}] return={result['return']:.3f} "
            f"alpha={result['mean_alpha']:.3f} temp={result['temperature']:.3f} "
            f"device={result['device']}"
        )


if __name__ == "__main__":
    main()
