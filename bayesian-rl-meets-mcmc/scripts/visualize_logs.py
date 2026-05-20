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
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def output_paths(results_dir: Path, phase: str, tag: str) -> tuple[Path, Path, str]:
    stamp = timestamp()
    phase_name = safe_tag(phase, "phase")
    run_tag = f"{safe_tag(tag, 'run')}_{stamp}"
    output_dir = results_dir / "plots" / phase_name / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = results_dir / "plots" / phase_name / "latest_comparison.png"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    return output_dir, latest_path, stamp


def read_tensorboard_runs(log_dir: Path, scalar: str, tag: str) -> list[tuple[np.ndarray, np.ndarray]]:
    event_files = list(log_dir.rglob("events.out.tfevents.*"))
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
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
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
    plt.title(f"Bayesian RL Comparison - {phase}")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir, latest_path, stamp = output_paths(args.results_dir, args.phase, args.tag)
    output_path = output_dir / f"comparison_return_{stamp}.png"

    runs = read_tensorboard_runs(args.log_dir, args.scalar, args.tag)
    source = "tensorboard"
    if not runs:
        runs = read_latest_stats(args.results_dir)
        source = "latest_stats"
    if not runs:
        raise FileNotFoundError("No TensorBoard scalar logs or latest_stats.json return series were found.")

    steps, mean, std = aggregate_runs(runs)
    plot_comparison(steps, mean, std, output_path, args.phase)
    shutil.copy2(output_path, latest_path)

    print(f"Source: {source}")
    print(f"Wrote {output_path}")
    print(f"Wrote {latest_path}")


if __name__ == "__main__":
    main()

