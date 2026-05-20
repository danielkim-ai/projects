"""Prepare the Low-data MuJoCo SGLD report artefacts.

The current Phase 1 path writes Antigravity-readable placeholders from a
deterministic diagnostic simulation. Replace the synthetic return traces with
real MuJoCo rollouts once the training loop is implemented.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import (  # noqa: E402
    ReportMetrics,
    bayesian_regret,
    effective_sample_size,
    expected_calibration_error,
    pac_bayes_bound,
)
from src.result_io import run_id, update_latest_copy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Low-data MuJoCo Phase 1 report artefacts.")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-tag", default="phase1_low_data")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = run_id(args.save_tag, phase="phase1", seed=args.seed)

    episodes = np.arange(1, args.episodes + 1)
    vi_returns = 120.0 + 8.0 * np.log1p(episodes) + rng.normal(0.0, 3.0, size=args.episodes)
    sgld_returns = vi_returns + 5.0 * (1.0 - np.exp(-episodes / 4.0)) + rng.normal(0.0, 1.5, size=args.episodes)
    oracle_returns = np.full(args.episodes, 155.0)
    posterior_chain = sgld_returns - sgld_returns.mean()

    empirical_loss = float(np.maximum(0.0, oracle_returns - sgld_returns).mean() / oracle_returns.mean())
    kl_qp = 18.0
    ess = effective_sample_size(posterior_chain)
    metrics = ReportMetrics(
        pac_bayes_bound=pac_bayes_bound(empirical_loss=empirical_loss, kl_qp=kl_qp, n=args.episodes),
        regret=bayesian_regret(oracle_returns, sgld_returns),
        calibration_error=expected_calibration_error(
            confidences=np.clip(sgld_returns / oracle_returns, 0.0, 1.0),
            outcomes=np.ones(args.episodes),
        ),
        effective_sample_size=ess,
    )

    stats = {
        "experiment": "low-data-mujoco-sgld-phase1",
        "experiment_id": experiment_id,
        "status": "ready_for_real_rollout_integration",
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "metrics": metrics.to_json(),
        "series": {
            "episode": episodes.tolist(),
            "vi_return": vi_returns.round(4).tolist(),
            "sgld_return": sgld_returns.round(4).tolist(),
            "oracle_return": oracle_returns.round(4).tolist(),
        },
        "report_contract": {
            "required_files": ["latest_stats.json", f"plot_{experiment_id}.png"],
            "consumer": "Antigravity",
            "notes": "Synthetic diagnostics must be replaced by MuJoCo rollout data in Phase 2.",
        },
    }

    stats_path = args.results_dir / f"stats_{experiment_id}.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    latest_stats_path = update_latest_copy(stats_path, "latest_stats.json")

    plt.figure(figsize=(8, 5))
    plt.plot(episodes, vi_returns, marker="o", label="VAC / VI")
    plt.plot(episodes, sgld_returns, marker="s", label="SGLD recalibrated")
    plt.plot(episodes, oracle_returns, linestyle="--", label="Oracle reference")
    plt.xlabel("Low-data episode")
    plt.ylabel("Return")
    plt.title("MCMC vs VI trade-off under low-data MuJoCo protocol")
    plt.legend()
    plt.tight_layout()
    plot_path = args.results_dir / f"plot_{experiment_id}.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()
    latest_plot_path = update_latest_copy(plot_path, "latest_mcmc_vs_vi_tradeoff.png")

    print(f"Wrote {stats_path}")
    print(f"Wrote {latest_stats_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {latest_plot_path}")


if __name__ == "__main__":
    main()

