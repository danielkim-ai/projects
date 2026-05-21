"""Analyse Phase 4 privacy ablations for Bayesian RL."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
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
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "results" / "logs" / "phase4")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--delta", type=float, default=1.0e-5)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--episodes", type=int, default=None)
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def plot_filename(env_id: str, descriptor: str) -> str:
    """Create an official Phase 4 privacy analysis filename."""

    env_name = safe_tag(env_id, "environment").replace("-", "").lower()
    descriptor_name = safe_tag(descriptor, "plot").lower()
    return f"phase4_{env_name}_{descriptor_name}.png"


def epsilon_label(epsilon: float) -> str:
    return f"epsilon={epsilon:g}"


def tag_for_epsilon(template: str, epsilon: float) -> str:
    return template.format(epsilon=epsilon)


def normalise_match_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def epsilon_tokens(epsilon: float) -> set[str]:
    raw = f"{epsilon:g}"
    decimal = f"{epsilon:.1f}"
    values = {raw, decimal, raw.replace(".", ""), decimal.replace(".", "")}
    tokens: set[str] = set()
    for value in values:
        tokens.update(
            {
                normalise_match_text(f"eps{value}"),
                normalise_match_text(f"epsilon{value}"),
                normalise_match_text(f"e{value}"),
            }
        )
    return tokens


def environment_tokens(env_id: str) -> set[str]:
    safe_env = safe_tag(env_id, "environment")
    first = safe_env.split("-")[0]
    return {
        normalise_match_text(safe_env),
        normalise_match_text(first),
    }


def resolve_input_path(path: Path) -> list[Path]:
    """Support absolute paths and paths relative to cwd or the project root."""

    if path.is_absolute():
        return [path.resolve(strict=False)]
    return [
        (Path.cwd() / path).resolve(strict=False),
        (PROJECT_ROOT / path).resolve(strict=False),
    ]


def list_existing_directories(bases: list[Path]) -> list[Path]:
    """Return directories below the candidate bases for debugging."""

    directories: list[Path] = []
    for base in bases:
        if not base.exists():
            continue
        for root, dirnames, _ in os.walk(base):
            root_path = Path(root).resolve(strict=False)
            directories.append(root_path)
            for dirname in dirnames:
                directories.append((root_path / dirname).resolve(strict=False))
    return sorted(dict.fromkeys(directories))


def fuzzy_matches_phase4(candidate: Path, env_id: str, epsilon: float, tag: str) -> bool:
    """Return whether a Phase 4 folder plausibly belongs to the requested ablation."""

    text = normalise_match_text(str(candidate))
    tag_text = normalise_match_text(tag)
    has_epsilon = any(token in text for token in epsilon_tokens(epsilon))
    has_environment = any(token in text for token in environment_tokens(env_id))
    return tag_text in text or (has_environment and has_epsilon)


def tensorboard_search_roots(
    log_dir: Path,
    tag: str,
    env_id: str,
    epsilon: float,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Return TensorBoard roots, attempted paths, and existing Phase 4 folders."""

    attempted: list[Path] = []
    roots: list[Path] = []
    candidate_bases = resolve_input_path(log_dir)
    candidate_bases.extend(
        [
            PROJECT_ROOT / "results" / "logs" / "phase4",
            PROJECT_ROOT / "results" / "archive" / "tensorboard",
        ]
    )
    candidate_bases = list(dict.fromkeys(base.resolve(strict=False) for base in candidate_bases))
    tag_fragment = safe_tag(tag, "run")
    for base in candidate_bases:
        for candidate in (base, base / tag_fragment, base / tag):
            resolved = candidate.resolve(strict=False)
            if resolved in attempted:
                continue
            attempted.append(resolved)
            if resolved.exists():
                roots.append(resolved)
    existing_directories = list_existing_directories(candidate_bases)
    for directory in existing_directories:
        if fuzzy_matches_phase4(directory, env_id, epsilon, tag):
            roots.append(directory)
    return sorted(dict.fromkeys(roots)), attempted, existing_directories


def read_tensorboard_runs(
    log_dir: Path,
    scalar: str,
    tag: str,
    env_id: str,
    epsilon: float,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[Path], list[Path]]:
    roots, attempted, existing_directories = tensorboard_search_roots(log_dir, tag, env_id, epsilon)
    event_files: list[Path] = []
    for root in roots:
        event_files.extend(root.rglob("events.out.tfevents.*"))
    event_files = [
        path
        for path in event_files
        if fuzzy_matches_phase4(path.parent, env_id, epsilon, tag)
    ]
    if not event_files:
        return [], attempted, existing_directories

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return [], attempted, existing_directories

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
    return runs, attempted, existing_directories


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


def archived_stats_paths(results_dir: Path, env_id: str, tag: str) -> tuple[list[Path], list[Path]]:
    attempted: list[Path] = []
    matches: list[Path] = []
    for base in resolve_input_path(results_dir):
        env_dir = base / "archive" / safe_tag(env_id, "environment")
        attempted.append(env_dir.resolve(strict=False))
        if env_dir.exists():
            matches.extend(sorted(env_dir.glob(f"stats_*{safe_tag(tag, 'run')}*.json")))
    return matches, attempted


def read_stats_series(
    results_dir: Path,
    env_id: str,
    tag: str,
    field: str,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[Path]]:
    runs: list[tuple[np.ndarray, np.ndarray]] = []
    paths, attempted = archived_stats_paths(results_dir, env_id, tag)
    for path in paths:
        stats = json.loads(path.read_text(encoding="utf-8-sig"))
        summaries = stats.get("episodes_summary", [])
        if not summaries:
            continue
        steps = np.asarray([item["episode"] for item in summaries], dtype=float)
        values = np.asarray([item.get(field, np.nan) for item in summaries], dtype=float)
        mask = np.isfinite(values)
        if np.any(mask):
            runs.append((steps[mask], values[mask]))
    return runs, attempted


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
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray] | None, list[Path], list[Path]]:
    tag = tag_for_epsilon(args.tag_template, epsilon)
    runs, attempted, existing_dirs = read_tensorboard_runs(
        args.log_dir,
        args.return_scalar,
        tag,
        args.env_id,
        epsilon,
    )
    if not runs:
        runs, stats_attempted = read_stats_series(args.results_dir, args.env_id, tag, "return")
        attempted.extend(stats_attempted)
    if not runs:
        return None, attempted, existing_dirs
    return aggregate_runs(runs), attempted, existing_dirs


def load_privacy_curve(
    args: argparse.Namespace,
    epsilon: float,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], list[Path], list[Path]]:
    tag = tag_for_epsilon(args.tag_template, epsilon)
    runs, attempted, existing_dirs = read_tensorboard_runs(
        args.log_dir,
        args.privacy_scalar,
        tag,
        args.env_id,
        epsilon,
    )
    if not runs:
        runs, stats_attempted = read_stats_series(args.results_dir, args.env_id, tag, "epsilon_spent")
        attempted.extend(stats_attempted)
    if runs:
        return aggregate_runs(runs), attempted, existing_dirs
    episodes = args.episodes or 200
    steps, spent = gaussian_moment_accountant_curve(epsilon, episodes, args.delta)
    return (steps, spent, np.zeros_like(spent)), attempted, existing_dirs


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
    attempted_paths: list[Path] = []
    existing_directories: list[Path] = []
    for epsilon in args.epsilons:
        return_curve, return_attempted, return_dirs = load_return_curve(args, epsilon)
        attempted_paths.extend(return_attempted)
        existing_directories.extend(return_dirs)
        if return_curve is not None:
            return_curves[epsilon] = return_curve
        privacy_curve, privacy_attempted, privacy_dirs = load_privacy_curve(args, epsilon)
        attempted_paths.extend(privacy_attempted)
        existing_directories.extend(privacy_dirs)
        privacy_curves[epsilon] = privacy_curve

    if not return_curves:
        unique_attempts = "\n".join(f"- {path}" for path in dict.fromkeys(attempted_paths))
        existing = "\n".join(f"- {path}" for path in dict.fromkeys(existing_directories))
        existing_message = existing or "- No folders currently exist below the searched log roots."
        raise FileNotFoundError(
            "No return curves were found. The analyser attempted these TensorBoard paths:\n"
            f"{unique_attempts}\n"
            "Existing folders below the searched log roots:\n"
            f"{existing_message}\n"
            "Run the Phase 4 HalfCheetah epsilon ablation first, or pass --log-dir with an absolute path.",
        )

    sensitivity_path = output_dir / plot_filename(
        args.env_id,
        "epsilon_sensitivity",
    )
    privacy_path = output_dir / plot_filename(
        args.env_id,
        "privacy_budget",
    )
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
