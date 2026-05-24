from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PLOT_DIR = Path("results") / "plots"
DEFAULT_METRICS = PLOT_DIR / "latest_visualise_metrics.json"


@dataclass(frozen=True)
class ExperimentMetrics:
    steps: list[int]
    clip_norms: list[float]
    epsilons: list[float]
    raw_norms: list[float]
    clipped_fractions: list[float]
    before_margins: np.ndarray
    after_margins: np.ndarray
    logged_return_mean: float
    logged_return_std: float
    proxy_return_mean: float
    proxy_return_std: float
    delta_j: float
    mia_report: dict[str, Any]
    deleted_episode: int
    affected_shards: list[int]
    source_path: Path


def _float_list(record: dict[str, Any], key: str) -> list[float]:
    if key not in record:
        raise KeyError(f"metrics JSON is missing required key: {key}")
    return [float(item) for item in record[key]]


def _int_list(record: dict[str, Any], key: str) -> list[int]:
    if key not in record:
        raise KeyError(f"metrics JSON is missing required key: {key}")
    return [int(item) for item in record[key]]


def metrics_from_record(record: dict[str, Any], source_path: Path) -> ExperimentMetrics:
    logged_return = record["logged_return"]
    proxy_return = record["proxy_return"]
    return ExperimentMetrics(
        steps=_int_list(record, "steps"),
        clip_norms=_float_list(record, "clip_norms"),
        epsilons=_float_list(record, "epsilons"),
        raw_norms=_float_list(record, "raw_norms"),
        clipped_fractions=_float_list(record, "clipped_fractions"),
        before_margins=np.asarray(record["before_margins"], dtype=float),
        after_margins=np.asarray(record["after_margins"], dtype=float),
        logged_return_mean=float(logged_return["mean"]),
        logged_return_std=float(logged_return["std"]),
        proxy_return_mean=float(proxy_return["mean"]),
        proxy_return_std=float(proxy_return["std"]),
        delta_j=float(record["delta_j"]),
        mia_report=dict(record["mia_report"]),
        deleted_episode=int(record["deleted_episode"]),
        affected_shards=_int_list(record, "affected_shards"),
        source_path=source_path,
    )


def read_metrics(path: Path) -> ExperimentMetrics:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run main.py first so it can write the latest "
            "experiment metrics JSON."
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    return metrics_from_record(record, path)


def gaussian_kde(values: np.ndarray, grid: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    if values.size == 0:
        raise ValueError("margin arrays must contain at least one value")
    if bandwidth is None:
        std = np.std(values, ddof=1) if len(values) > 1 else 1.0
        bandwidth = 1.06 * std * max(len(values), 1) ** (-1 / 5)
        bandwidth = max(float(bandwidth), 0.2)
    scaled = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * scaled**2).sum(axis=1)
    density /= len(values) * bandwidth * np.sqrt(2 * np.pi)
    return density


def plot_privacy_utility(metrics: ExperimentMetrics, output_dir: Path) -> Path:
    steps = np.asarray(metrics.steps)
    fig, ax_clip = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    ax_eps = ax_clip.twinx()

    ax_clip.step(
        steps,
        metrics.clip_norms,
        where="post",
        color="#2f6f9f",
        linewidth=2.5,
        label="Clip norm C",
    )
    ax_clip.scatter(steps, metrics.clip_norms, color="#2f6f9f", s=18, zorder=3)
    ax_eps.plot(
        steps,
        metrics.epsilons,
        color="#b4493a",
        linewidth=2.5,
        marker="o",
        label="RDP epsilon",
    )

    ax_clip.set_title("Trajectory DP-SGD Privacy-Utility Schedule", pad=12)
    ax_clip.set_xlabel("Training step")
    ax_clip.set_ylabel("Clipping norm C", color="#2f6f9f")
    ax_eps.set_ylabel("Cumulative epsilon", color="#b4493a")
    ax_clip.set_xticks(steps)
    ax_clip.grid(True, alpha=0.25)

    lines = ax_clip.get_lines() + ax_eps.get_lines()
    labels = [line.get_label() for line in lines]
    ax_clip.legend(lines, labels, loc="center right", frameon=False)
    fig.tight_layout()
    path = output_dir / "plot_privacy_utility.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_unlearning_margin(metrics: ExperimentMetrics, output_dir: Path) -> Path:
    before = metrics.before_margins
    after = metrics.after_margins
    lower = min(before.min(), after.min()) - 0.8
    upper = max(before.max(), after.max()) + 0.8
    grid = np.linspace(lower, upper, 240)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    bins = np.linspace(lower, upper, 14)
    ax.hist(before, bins=bins, density=True, alpha=0.28, color="#4c78a8", label="Before histogram")
    ax.hist(after, bins=bins, density=True, alpha=0.28, color="#f58518", label="After histogram")
    ax.plot(grid, gaussian_kde(before, grid), color="#245a8d", linewidth=2.5, label="Before KDE")
    ax.plot(grid, gaussian_kde(after, grid), color="#c15d00", linewidth=2.5, label="After KDE")
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.2, label="Decision boundary")
    ax.set_title("MIA Margin Distribution Before and After Unlearning", pad=12)
    ax.set_xlabel("Margin: non-member loss - member loss")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path = output_dir / "plot_unlearning_margin.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_utility_tradeoff(metrics: ExperimentMetrics, output_dir: Path) -> Path:
    labels = ["Logged data", "DP policy proxy"]
    means = np.array([metrics.logged_return_mean, metrics.proxy_return_mean])
    stds = np.array([metrics.logged_return_std, metrics.proxy_return_std])

    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=180)
    bars = ax.bar(labels, means, yerr=stds, capsize=8, color=["#59a14f", "#b07aa1"], alpha=0.86)
    ax.axhline(0.0, color="#333333", linewidth=0.9)
    y_mid = float(means.mean())
    ax.annotate(
        rf"$\Delta J = {metrics.delta_j:.2f}$",
        xy=(0.5, y_mid),
        xytext=(0.5, y_mid + 4.0),
        arrowprops={"arrowstyle": "<->", "color": "#333333", "lw": 1.4},
        ha="center",
        va="center",
        fontsize=11,
    )
    for bar, mean in zip(bars, means):
        vertical_alignment = "top" if mean < 0 else "bottom"
        offset = -0.65 if mean < 0 else 0.65
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + offset,
            f"{mean:.2f}",
            ha="center",
            va=vertical_alignment,
        )
    ax.set_title("Utility Trade-off Under DP Weight Noise", pad=12)
    ax.set_ylabel("Expected return proxy")
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(
        0.5,
        ax.get_ylim()[0] + 0.6,
        "Error bars show episode-level standard deviation.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    path = output_dir / "plot_utility_tradeoff.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render audit plots from an experiment metrics JSON written by main.py."
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=DEFAULT_METRICS,
        help="Experiment metrics JSON to render. Defaults to results/plots/latest_visualise_metrics.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLOT_DIR,
        help="Directory where PNG plots will be written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_metrics(args.metrics_json)
    paths = [
        plot_privacy_utility(metrics, args.output_dir),
        plot_unlearning_margin(metrics, args.output_dir),
        plot_utility_tradeoff(metrics, args.output_dir),
    ]
    summary = {
        "metrics_source": str(metrics.source_path),
        "deleted_episode": metrics.deleted_episode,
        "affected_shards": metrics.affected_shards,
        "final_epsilon": metrics.epsilons[-1],
        "delta_j": metrics.delta_j,
        "before_margin_variance": float(np.var(metrics.before_margins)),
        "after_margin_variance": float(np.var(metrics.after_margins)),
    }
    print(f"visualisation_summary={summary}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
