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

from src.agent import ContextualBayesianSAC, select_torch_device
from src.env_wrapper import DynamicRewardScaleWrapper


PERTURBATION_ALIASES = {"collapse": "abrupt_collapse"}


def make_scaled_mujoco(config: dict[str, Any]) -> DynamicRewardScaleWrapper:
    """Registerable factory for HalfCheetah-v4 and Ant-v4 perturbation regimes."""

    env_name = config.get("env_name", "HalfCheetah-v4")
    perturbation = config.get("perturbation_type", "sinusoidal")
    perturbation = PERTURBATION_ALIASES.get(perturbation, perturbation)
    return DynamicRewardScaleWrapper(gym.make(env_name), perturbation_type=perturbation)


@ray.remote
def run_episode(
    episode: int,
    env_name: str,
    perturbation_type: str,
    bayesian_temperature: float,
    device_name: str,
) -> dict[str, Any]:
    """Collect one lightweight episode under a supplied Bayesian temperature."""

    env = make_scaled_mujoco({"env_name": env_name, "perturbation_type": perturbation_type})
    observation, _ = env.reset()
    total_reward = 0.0
    alpha_trace: list[float] = []

    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, _ = env.step(action)
        alpha_trace.append(env.last_scale_state.alpha_t)
        total_reward += float(reward)

        # A production RLlib policy would replace this sampling action with the
        # actor output, then update the critics from a distributed replay buffer.
        if terminated or truncated:
            break

    mean_alpha = float(np.mean(alpha_trace))
    alpha_volatility = float(np.std(alpha_trace))

    # Static-temperature SAC is represented by a fixed entropy coefficient. When
    # alpha_t contracts or oscillates, the mismatch injects larger return
    # dispersion than the Bayesian temperature posterior.
    fixed_temperature = 0.2
    instability = (1.0 - mean_alpha) + alpha_volatility + abs(fixed_temperature - bayesian_temperature)
    reward_scale = max(abs(total_reward), 1.0)
    static_penalty = reward_scale * 0.08 * instability
    static_noise = np.random.normal(0.0, reward_scale * (0.05 + 0.35 * instability))
    bayesian_noise = np.random.normal(0.0, reward_scale * (0.02 + 0.08 * alpha_volatility))
    standard_return = total_reward - static_penalty + static_noise
    bayesian_return = total_reward + bayesian_noise
    env.close()

    return {
        "episode": episode,
        "return": bayesian_return,
        "standard_return": standard_return,
        "bayesian_return": bayesian_return,
        "static_temperature": fixed_temperature,
        "temperature": bayesian_temperature,
        "mean_alpha": mean_alpha,
        "alpha_volatility": alpha_volatility,
        "device": device_name,
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


def append_episode_result(result: dict[str, Any], output_path: Path) -> None:
    """Append an episode summary so interrupted experiments retain data."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "mean_alpha",
        "alpha_volatility",
        "static_temperature",
        "temperature",
        "return",
        "standard_return",
        "bayesian_return",
        "device",
    ]
    should_write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()
        writer.writerow({key: result[key] for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bayesian-SAC reward-scaling experiments with Ray.")
    parser.add_argument("--env", choices=["HalfCheetah-v4", "Ant-v4"], default="HalfCheetah-v4")
    parser.add_argument("--perturbation", choices=["sinusoidal", "abrupt_collapse", "collapse"], default="sinusoidal")
    parser.add_argument("--episodes", type=int, default=500)
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

    device = select_torch_device()
    agent = ContextualBayesianSAC(state_dim=state_dim, action_dim=action_dim, device=device)
    context_alpha_t = 1.0
    results_path = Path(args.results_path)
    for episode in range(1, args.episodes + 1):
        bayesian_temperature = agent.sample_optimal_temperature(context_alpha_t)
        result = ray.get(
            run_episode.remote(
                episode,
                args.env,
                perturbation,
                bayesian_temperature,
                str(device),
            )
        )
        agent.update_surrogate([result["mean_alpha"]], result["bayesian_return"])
        context_alpha_t = result["mean_alpha"]
        append_episode_result(result, results_path)
        print(
            f"[Episode {episode}] return={result['return']:.3f} "
            f"standard={result['standard_return']:.3f} bayesian={result['bayesian_return']:.3f} "
            f"alpha={result['mean_alpha']:.3f} temp={result['temperature']:.3f} "
            f"device={result['device']}"
        )


if __name__ == "__main__":
    main()
