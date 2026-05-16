"""Automated visualisation for Bayesian-SAC reward-scaling experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REQUIRED_TIMELINE_COLUMNS = {"episode", "mean_alpha", "temperature"}
REQUIRED_RETURN_COLUMNS = {"agent", "return"}


def read_experiment_log(path: Path) -> pd.DataFrame:
    """Read a CSV or JSON experiment log into a normalised dataframe."""

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return pd.DataFrame(payload)
    raise ValueError("Experiment logs must be supplied as CSV or JSON files.")


def plot_temperature_timeline(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Plot reward scaling alpha_t against the learned temperature tau_t."""

    missing = REQUIRED_TIMELINE_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing timeline columns: {', '.join(sorted(missing))}")

    output_path = output_dir / "reward_scale_temperature_timeline.png"
    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(frame["episode"], frame["mean_alpha"], marker="o", label=r"Reward scale $\alpha_t$")
    axis.plot(frame["episode"], frame["temperature"], marker="s", label=r"Learned temperature $\tau_t$")
    axis.set_xlabel("Episode")
    axis.set_ylabel("Value")
    axis.set_title("Reward Scaling vs Bayesian Temperature Adaptation")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_return_variance(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Compare return dispersion between Standard SAC and Bayesian SAC."""

    if "bayesian_return" in frame.columns and "standard_return" in frame.columns:
        comparison = frame.melt(
            value_vars=["standard_return", "bayesian_return"],
            var_name="agent",
            value_name="return",
        )
        comparison["agent"] = comparison["agent"].map(
            {"standard_return": "Standard SAC", "bayesian_return": "Bayesian SAC"}
        )
    else:
        missing = REQUIRED_RETURN_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing return-comparison columns: {', '.join(sorted(missing))}")
        comparison = frame.copy()

    output_path = output_dir / "return_variance_comparison.png"
    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=comparison, x="agent", y="return", ax=axis, palette="Set2")
    sns.stripplot(data=comparison, x="agent", y="return", ax=axis, color="black", alpha=0.45)
    axis.set_xlabel("")
    axis.set_ylabel("Episode Return")
    axis.set_title("Return Variance: Standard SAC vs Bayesian SAC")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create publication-ready plots from experiment logs.")
    parser.add_argument("--input", default="results/experiment_log.csv", help="Path to a CSV or JSON experiment log.")
    parser.add_argument("--output-dir", default="results", help="Directory in which PNG figures are written.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = read_experiment_log(Path(args.input))
    timeline_path = plot_temperature_timeline(frame, output_dir)
    variance_path = plot_return_variance(frame, output_dir)
    print(f"[Visualisation] Saved {timeline_path}")
    print(f"[Visualisation] Saved {variance_path}")


if __name__ == "__main__":
    main()
