"""Analyse Phase 4 privacy ablations for Bayesian RL."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.result_io import safe_tag  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse Phase 4 epsilon sensitivity and privacy-budget consumption.",
    )
    parser.add_argument("--env-id", default="HalfCheetah-v4")
    parser.add_argument("--epsilons", nargs="+", type=float, default=[1.0, 4.0, 16.0])
    parser.add_argument("--tag-template", default="halfcheetah_privacy_eps{epsilon:g}")
    parser.add_argument("--return-scalar", default="rollout/return")
    parser.add_argument("--privacy-scalar", default="privacy/epsilon_spent")
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "results" / "archive" / "tensorboard")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--delta", type=float, default=1.0e-5)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--episodes", type=int, default=None)
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def epsilon_label(epsilon: float) -> str:
    return f"epsilon={epsilon:g}"


def tag_for_epsilon(template: str, epsilon: float) -> str:
    return template.format(epsilon=epsilon)


def read_tensorboard_runs(
    log_dir: Path,
    scalar: str,
    tag: str,
    env_id: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    event_files = list(log_dir.rglob("events.out.tfevents.*"))
    env_fragment = safe_tag(env_id, "environment").lower()
    tag_fragment = tag.lower()
    event_files = [
        path
        for path in event_files
        if env_fragment in str(path).lower() and tag_fragment in str(path).lower()
    ]
    if not event_files:
        return []

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return []

    grouped: dict[Path, list[tuple[int, float]]] = defaultdict(list)
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file.parent))
        accumulator.Reload()
        if scalar not in accumulator.Tags().get("scalars", []):
            continue
        grouped[event_file.parent].extend(
            (event.step, float(event.value)) for event in accumulator.Scalars(scalar)
        )

    runs: list[tuple[np.ndarray, np.ndarray]] = []
    for points in grouped.values():
        if not points:
            continue
        dedup: dict[int, float] = {}
        for step, value in points:
            dedup[int(step)] = value
        steps = np.asarray(sorted(dedup), dtype=float)
        values = np.asarray([dedup[int(step)] for step in steps], dtype=float)
        runs.append((steps, values))
    return runs


def aggregate_runs(runs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(runs) == 1:
        steps, values = runs[0]
        return steps, values, np.zeros_like(values)

    common_steps = sorted(set.intersection(*(set(steps.astype(int)) for steps, _ in runs)))
    if not common_steps:
        max_len = min(len(values) for _, values in runs)
        matrix = np.vstack([values[:max_len] for _, values in runs])
        return np.arange(max_len, dtype=float), matrix.mean(axis=0), matrix.std(axis=0)

    matrix = []
    for steps, values in runs:
        lookup = {int(step): value for step, value in zip(steps, values)}
        matrix.append([lookup[step] for step in common_steps])
    values_matrix = np.asarray(matrix, dtype=float)
    return np.asarray(common_steps, dtype=float), values_matrix.mean(axis=0), values_matrix.std(axis=0)


def archived_stats_paths(results_dir: Path, env_id: str, tag: str) -> list[Path]:
    env_dir = results_dir / "archive" / safe_tag(env_id, "environment")
    if not env_dir.exists():
        return []
    return sorted(env_dir.glob(f"stats_*{safe_tag(tag, 'run')}*.json"))


def read_stats_series(
    results_dir: Path,
    env_id: str,
    tag: str,
    field: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    runs: list[tuple[np.ndarray, np.ndarray]] = []
    for path in archived_stats_paths(results_dir, env_id, tag):
        stats = json.loads(path.read_text(encoding="utf-8-sig"))
        summaries = stats.get("episodes_summary", [])
        if not summaries:
            continue
        steps = np.asarray([item["episode"] for item in summaries], dtype=float)
        values = np.asarray([item.get(field, np.nan) for item in summaries], dtype=float)
        mask = np.isfinite(values)
        if np.any(mask):
            runs.append((steps[mask], values[mask]))
    return runs


def gaussian_moment_accountant_curve(
    epsilon: float,
    episodes: int,
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    steps = np.arange(1, episodes + 1, dtype=float)
    noise_multiplier = math.sqrt(2.0 * math.log(1.25 / delta)) * math.sqrt(episodes) / max(
        epsilon,
        1.0e-8,
    )
    per_step = math.sqrt(2.0 * math.log(1.25 / delta)) / noise_multiplier
    return steps, np.sqrt(steps) * per_step


def load_return_curve(
    args: argparse.Namespace,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    tag = tag_for_epsilon(args.tag_template, epsilon)
    runs = read_tensorboard_runs(args.log_dir, args.return_scalar, tag, args.env_id)
    if not runs:
        runs = read_stats_series(args.results_dir, args.env_id, tag, "return")
    if not runs:
        return None
    return aggregate_runs(runs)


def load_privacy_curve(
    args: argparse.Namespace,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tag = tag_for_epsilon(args.tag_template, epsilon)
    runs = read_tensorboard_runs(args.log_dir, args.privacy_scalar, tag, args.env_id)
    if not runs:
        runs = read_stats_series(args.results_dir, args.env_id, tag, "epsilon_spent")
    if runs:
        return aggregate_runs(runs)
    episodes = args.episodes or 200
    steps, spent = gaussian_moment_accountant_curve(epsilon, episodes, args.delta)
    return steps, spent, np.zeros_like(spent)


def plot_epsilon_sensitivity(
    curves: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path,
    env_id: str,
) -> None:
    plt.figure(figsize=(8.8, 5.4))
    colours = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(curves), 1)))
    for colour, epsilon in zip(colours, sorted(curves)):
        steps, mean, std = curves[epsilon]
        label = epsilon_label(epsilon)
        plt.plot(steps, mean, linewidth=2.2, color=colour, label=label)
        if np.any(std > 0):
            plt.fill_between(steps, mean - std, mean + std, color=colour, alpha=0.16)
    plt.xlabel("Training Steps")
    plt.ylabel("Expected Return")
    plt.title(f"Epsilon Sensitivity under DP-SGLD - {env_id}")
    plt.grid(alpha=0.24)
    plt.legend(title="Privacy Budget", frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_privacy_loss(
    curves: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path,
    env_id: str,
) -> None:
    plt.figure(figsize=(8.8, 5.4))
    colours = plt.cm.plasma(np.linspace(0.18, 0.82, max(len(curves), 1)))
    for colour, epsilon in zip(colours, sorted(curves)):
        steps, mean, std = curves[epsilon]
        label = epsilon_label(epsilon)
        plt.plot(steps, mean, linewidth=2.2, color=colour, label=label)
        if np.any(std > 0):
            plt.fill_between(steps, mean - std, mean + std, color=colour, alpha=0.16)
    plt.xlabel("Episodes")
    plt.ylabel(r"$\epsilon$ Spent")
    plt.title(f"Privacy Budget Consumption over Training - {env_id}")
    plt.grid(alpha=0.24)
    plt.legend(title="Target Budget", frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    stamp = timestamp()
    output_dir = args.results_dir / "plots" / "phase4" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    return_curves: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    privacy_curves: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for epsilon in args.epsilons:
        return_curve = load_return_curve(args, epsilon)
        if return_curve is not None:
            return_curves[epsilon] = return_curve
        privacy_curves[epsilon] = load_privacy_curve(args, epsilon)

    if not return_curves:
        raise FileNotFoundError(
            "No return curves were found. Run the Phase 4 HalfCheetah epsilon ablation first.",
        )

    sensitivity_path = output_dir / f"epsilon_sensitivity_{safe_tag(args.env_id, 'environment')}_{stamp}.png"
    privacy_path = output_dir / f"privacy_loss_{safe_tag(args.env_id, 'environment')}_{stamp}.png"
    plot_epsilon_sensitivity(return_curves, sensitivity_path, args.env_id)
    plot_privacy_loss(privacy_curves, privacy_path, args.env_id)

    latest_sensitivity = output_dir / "latest_epsilon_sensitivity.png"
    latest_privacy = output_dir / "latest_privacy_loss.png"
    shutil.copy2(sensitivity_path, latest_sensitivity)
    shutil.copy2(privacy_path, latest_privacy)

    print(f"Wrote {sensitivity_path}")
    print(f"Wrote {privacy_path}")
    print(f"Wrote {latest_sensitivity}")
    print(f"Wrote {latest_privacy}")


if __name__ == "__main__":
    main()
