"""Visualise TensorBoard logs or latest diagnostic stats with archive-safe plots."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.result_io import safe_tag  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise Bayesian RL TensorBoard logs.")
    parser.add_argument("--phase", required=True, help="Phase name, for example phase1 or phase2.")
    parser.add_argument("--tag", required=True, help="Experiment tag, for example seed7.")
    parser.add_argument("--scalar", default="rollout/return", help="TensorBoard scalar to visualise.")
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "results" / "archive" / "tensorboard")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--env-id", default=None, help="Optional environment identifier for environment-specific logs.")
    args = parser.parse_args()
    if args.phase.lower() in {"phase2", "phase3"} and not args.env_id:
        parser.error("--env-id is required for Phase 2 and Phase 3 visualisation.")
    return args


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def plot_filename(phase: str, env_id: str | None, tag: str, stamp: str, descriptor: str) -> str:
    """Create an archive-safe plot filename with phase, environment, tag, and time."""

    phase_name = safe_tag(phase, "phase")
    env_name = safe_tag(env_id or "all-envs", "environment")
    tag_name = safe_tag(tag, "run")
    descriptor_name = safe_tag(descriptor, "plot")
    return f"{phase_name}_{env_name}_{tag_name}_{stamp}_{descriptor_name}.png"


def output_paths(results_dir: Path, phase: str, tag: str, env_id: str | None = None) -> tuple[Path, Path, str]:
    stamp = timestamp()
    phase_name = safe_tag(phase, "phase")
    run_tag = f"{safe_tag(tag, 'run')}_{stamp}"
    phase_dir = results_dir / "plots" / phase_name
    if env_id:
        phase_dir = phase_dir / safe_tag(env_id, "environment")
    output_dir = phase_dir / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = phase_dir / "latest_comparison.png"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    return output_dir, latest_path, stamp


def read_tensorboard_runs(log_dir: Path, scalar: str, tag: str, env_id: str | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    event_files = list(log_dir.rglob("events.out.tfevents.*"))
    if env_id:
        event_files = [path for path in event_files if safe_tag(env_id, "environment").lower() in str(path).lower()]
    if tag:
        event_files = [path for path in event_files if tag.lower() in str(path).lower()]
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
        dedup = {}
        for step, value in points:
            dedup[step] = value
        steps = np.asarray(sorted(dedup), dtype=float)
        values = np.asarray([dedup[int(step)] for step in steps], dtype=float)
        runs.append((steps, values))
    return runs


def read_latest_stats(results_dir: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    stats_path = results_dir / "latest_stats.json"
    if not stats_path.exists():
        return []
    stats = json.loads(stats_path.read_text(encoding="utf-8-sig"))
    series = stats.get("series", {})
    if "episode" in series and "sgld_return" in series:
        return [
            (
                np.asarray(series["episode"], dtype=float),
                np.asarray(series["sgld_return"], dtype=float),
            )
        ]
    summaries = stats.get("episodes_summary", [])
    if summaries:
        return [
            (
                np.asarray([item["episode"] for item in summaries], dtype=float),
                np.asarray([item["return"] for item in summaries], dtype=float),
            )
        ]
    return []


def read_tensorboard_hypers(log_dir: Path, tag: str, env_id: str | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    hypers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ("gamma", "alpha"):
        runs = read_tensorboard_runs(log_dir, f"hyperparameter/{name}", tag, env_id)
        if runs:
            steps, mean, _ = aggregate_runs(runs)
            hypers[name] = (steps, mean)
    return hypers


def read_latest_hypers(results_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    stats_path = results_dir / "latest_stats.json"
    if not stats_path.exists():
        return {}
    stats = json.loads(stats_path.read_text(encoding="utf-8-sig"))
    samples = stats.get("hyperparameter_samples") or []
    if not samples:
        summaries = stats.get("episodes_summary", [])
        samples = [item for item in summaries if "gamma" in item and "alpha" in item]
    if not samples:
        return {}
    episodes = np.asarray([item["episode"] for item in samples], dtype=float)
    return {
        "gamma": (episodes, np.asarray([item["gamma"] for item in samples], dtype=float)),
        "alpha": (episodes, np.asarray([item["alpha"] for item in samples], dtype=float)),
    }


def aggregate_runs(runs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(runs) == 1:
        steps, values = runs[0]
        return steps, values, np.zeros_like(values)

    common_steps = sorted(set.intersection(*(set(steps.astype(int)) for steps, _ in runs)))
    if not common_steps:
        max_len = min(len(values) for _, values in runs)
        common_steps = list(range(max_len))
        matrix = np.vstack([values[:max_len] for _, values in runs])
        return np.asarray(common_steps, dtype=float), matrix.mean(axis=0), matrix.std(axis=0)

    matrix = []
    for steps, values in runs:
        step_to_value = {int(step): value for step, value in zip(steps, values)}
        matrix.append([step_to_value[step] for step in common_steps])
    values_matrix = np.asarray(matrix, dtype=float)
    return (
        np.asarray(common_steps, dtype=float),
        values_matrix.mean(axis=0),
        values_matrix.std(axis=0),
    )


def plot_comparison(steps: np.ndarray, mean: np.ndarray, std: np.ndarray, output_path: Path, phase: str) -> None:
    plt.figure(figsize=(8.5, 5.2))
    plt.plot(steps, mean, color="#1f4e79", linewidth=2.2, label="Mean expected return")
    if np.any(std > 0):
        plt.fill_between(steps, mean - std, mean + std, color="#1f4e79", alpha=0.18, label="Standard deviation")
    plt.xlabel("Optimisation Steps")
    plt.ylabel("Expected Return")
    subtitle = "\nPhase 2: Diagnostic Run" if phase.lower() == "phase2" else ""
    plt.title(f"Bayesian RL Comparison - {phase}{subtitle}")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_hyperparameters(hypers: dict[str, tuple[np.ndarray, np.ndarray]], output_path: Path, phase: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    colours = {"gamma": "#1f4e79", "alpha": "#7b3294"}
    labels = {"gamma": "Discount Posterior Gamma", "alpha": "Entropy Posterior Alpha"}

    for row, name in enumerate(("gamma", "alpha")):
        if name not in hypers:
            axes[row, 0].axis("off")
            axes[row, 1].axis("off")
            continue
        steps, values = hypers[name]
        axes[row, 0].plot(steps, values, color=colours[name], linewidth=2.0)
        axes[row, 0].set_xlabel("Optimisation Steps")
        axes[row, 0].set_ylabel(labels[name])
        axes[row, 0].grid(alpha=0.25)
        axes[row, 1].hist(values, bins=min(20, max(5, len(values))), color=colours[name], alpha=0.72)
        axes[row, 1].set_xlabel(labels[name])
        axes[row, 1].set_ylabel("Posterior Frequency")
        axes[row, 1].grid(alpha=0.2)

    subtitle = "\nPhase 2: Diagnostic Run" if phase.lower() == "phase2" else ""
    fig.suptitle(f"Hyperparameter Posterior Estimation - {phase}{subtitle}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir, latest_path, stamp = output_paths(args.results_dir, args.phase, args.tag, args.env_id)
    output_path = output_dir / plot_filename(
        args.phase,
        args.env_id,
        args.tag,
        stamp,
        "comparison_return",
    )

    runs = read_tensorboard_runs(args.log_dir, args.scalar, args.tag, args.env_id)
    source = "tensorboard"
    if not runs:
        runs = read_latest_stats(args.results_dir)
        source = "latest_stats"
    if not runs:
        raise FileNotFoundError("No TensorBoard scalar logs or latest_stats.json return series were found.")

    steps, mean, std = aggregate_runs(runs)
    plot_comparison(steps, mean, std, output_path, args.phase)
    shutil.copy2(output_path, latest_path)

    hypers = read_tensorboard_hypers(args.log_dir, args.tag, args.env_id)
    if not hypers:
        hypers = read_latest_hypers(args.results_dir)
    if hypers:
        hyper_path = output_dir / plot_filename(
            args.phase,
            args.env_id,
            args.tag,
            stamp,
            "hyperparameter_posterior",
        )
        latest_hyper_path = latest_path.with_name("latest_hyperparameters.png")
        plot_hyperparameters(hypers, hyper_path, args.phase)
        shutil.copy2(hyper_path, latest_hyper_path)
        print(f"Wrote {hyper_path}")
        print(f"Wrote {latest_hyper_path}")

    print(f"Source: {source}")
    print(f"Wrote {output_path}")
    print(f"Wrote {latest_path}")


if __name__ == "__main__":
    main()
