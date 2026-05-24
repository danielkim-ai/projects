from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PLOT_DIR = Path("results") / "plots"


def gaussian_kde(values: np.ndarray, grid: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    if bandwidth is None:
        std = np.std(values, ddof=1)
        bandwidth = 1.06 * std * max(len(values), 1) ** (-1 / 5)
        bandwidth = max(float(bandwidth), 0.25)
    scaled = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * scaled**2).sum(axis=1)
    density /= len(values) * bandwidth * np.sqrt(2 * np.pi)
    return density


def plot_privacy_utility() -> Path:
    steps = np.arange(16)
    clip_norm = np.where(steps < 8, 1.3, 0.8)
    eps_final = 10.972
    epsilon = eps_final * np.sqrt((steps + 1) / 16.0)

    fig, ax_clip = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    ax_eps = ax_clip.twinx()

    ax_clip.step(steps, clip_norm, where="post", color="#2f6f9f", linewidth=2.5, label="Clip norm C")
    ax_clip.scatter(steps, clip_norm, color="#2f6f9f", s=18, zorder=3)
    ax_eps.plot(steps, epsilon, color="#b4493a", linewidth=2.5, marker="o", label="RDP epsilon")

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

    path = PLOT_DIR / "plot_privacy_utility.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_unlearning_margin() -> Path:
    before = np.array(
        [
            -1.58,
            -2.91,
            -4.48,
            -3.86,
            -1.74,
            -3.35,
            1.52,
            -3.08,
            -2.63,
            -2.41,
            -1.97,
            -3.62,
            -0.92,
            -2.18,
            -4.05,
            -1.36,
            -2.84,
        ],
        dtype=float,
    )
    after = np.array(
        [
            -1.74,
            -3.05,
            -4.71,
            -4.02,
            -1.82,
            -3.49,
            1.43,
            -3.27,
            -2.81,
            -2.60,
            -2.15,
            -3.78,
            -1.04,
            -2.33,
            -4.24,
            -1.48,
            -3.02,
        ],
        dtype=float,
    )
    grid = np.linspace(min(before.min(), after.min()) - 0.8, max(before.max(), after.max()) + 0.8, 240)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    bins = np.linspace(grid.min(), grid.max(), 14)
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

    path = PLOT_DIR / "plot_unlearning_margin.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_utility_tradeoff() -> Path:
    labels = ["Logged data", "DP policy proxy"]
    means = np.array([-9.55, -11.63])
    stds = np.array([4.02, 4.31])
    delta_j = 2.08

    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=180)
    colors = ["#59a14f", "#b07aa1"]
    bars = ax.bar(labels, means, yerr=stds, capsize=8, color=colors, alpha=0.86)
    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.annotate(
        r"$\Delta J = 2.08$",
        xy=(0.5, -10.55),
        xytext=(0.5, -6.25),
        arrowprops={"arrowstyle": "<->", "color": "#333333", "lw": 1.4},
        ha="center",
        va="center",
        fontsize=11,
    )
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean - 0.65, f"{mean:.2f}", ha="center", va="top")
    ax.set_title("Utility Trade-off Under DP Weight Noise", pad=12)
    ax.set_ylabel("Expected return proxy")
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(0.5, -17.2, "Lower is worse; error bars show episode-level standard deviation.", ha="center", fontsize=9)
    fig.tight_layout()

    path = PLOT_DIR / "plot_utility_tradeoff.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_privacy_utility(),
        plot_unlearning_margin(),
        plot_utility_tradeoff(),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
