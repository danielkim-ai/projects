# Research & Engineering Projects Archive

A curated, centralised repository capturing foundational and empirical implementations spanning Reinforcement Learning, Applied Statistics, and Sequential Decision-Making under Uncertainty.

## Core Research Directory
- `/policy-learning-stability`: Robust policy gradients under non-stationary reward scaling regimes. (Completed)
- `/bayesian-rl-meets-mcmc`: Bayesian reinforcement learning with MCMC-VI posterior estimation for sample-efficient uncertainty quantification. Phase 1-4 complete, including Privacy-Preserving Analysis and the final Differential Privacy Integration milestone.

## Experimental Results Gallery

The Bayesian RL project archives visualisation artefacts under `bayesian-rl-meets-mcmc/results/plots/` using the official `{phase}_{env_id}_{type}.png` naming convention.

### Phase 1 Diagnostic

![Phase 1 HalfCheetah diagnostic](bayesian-rl-meets-mcmc/results/plots/phase1/phase1_halfcheetahv4_diagnostic.png)

### Phase 3 Posterior

| Environment | Posterior | Performance |
| --- | --- | --- |
| HalfCheetah-v4 | ![Phase 3 HalfCheetah posterior](bayesian-rl-meets-mcmc/results/plots/phase3/phase3_halfcheetahv4_posterior.png) | ![Phase 3 HalfCheetah performance](bayesian-rl-meets-mcmc/results/plots/phase3/phase3_halfcheetahv4_performance.png) |
| Ant-v4 | ![Phase 3 Ant posterior](bayesian-rl-meets-mcmc/results/plots/phase3/phase3_antv4_posterior.png) | ![Phase 3 Ant performance](bayesian-rl-meets-mcmc/results/plots/phase3/phase3_antv4_performance.png) |
| Hopper-v4 | ![Phase 3 Hopper posterior](bayesian-rl-meets-mcmc/results/plots/phase3/phase3_hopperv4_posterior.png) | ![Phase 3 Hopper performance](bayesian-rl-meets-mcmc/results/plots/phase3/phase3_hopperv4_performance.png) |
| Humanoid-v4 | ![Phase 3 Humanoid posterior](bayesian-rl-meets-mcmc/results/plots/phase3/phase3_humanoidv4_posterior.png) | ![Phase 3 Humanoid performance](bayesian-rl-meets-mcmc/results/plots/phase3/phase3_humanoidv4_performance.png) |

### Differential Privacy Analysis

![Phase 4 HalfCheetah epsilon sensitivity](bayesian-rl-meets-mcmc/results/plots/phase4/analysis/phase4_halfcheetahv4_epsilon_sensitivity.png)

![Phase 4 HalfCheetah privacy budget](bayesian-rl-meets-mcmc/results/plots/phase4/analysis/phase4_halfcheetahv4_privacy_budget.png)

### Phase 4 Privacy-Preserving Performance

| Environment | Posterior | Performance |
| --- | --- | --- |
| HalfCheetah-v4 | ![Phase 4 HalfCheetah posterior](bayesian-rl-meets-mcmc/results/plots/phase4/phase4_halfcheetahv4_posterior.png) | ![Phase 4 HalfCheetah performance](bayesian-rl-meets-mcmc/results/plots/phase4/phase4_halfcheetahv4_performance.png) |
| Ant-v4 | ![Phase 4 Ant posterior](bayesian-rl-meets-mcmc/results/plots/phase4/phase4_antv4_posterior.png) | ![Phase 4 Ant performance](bayesian-rl-meets-mcmc/results/plots/phase4/phase4_antv4_performance.png) |
| Hopper-v4 | ![Phase 4 Hopper posterior](bayesian-rl-meets-mcmc/results/plots/phase4/phase4_hopperv4_posterior.png) | ![Phase 4 Hopper performance](bayesian-rl-meets-mcmc/results/plots/phase4/phase4_hopperv4_performance.png) |
| Humanoid-v4 | ![Phase 4 Humanoid posterior](bayesian-rl-meets-mcmc/results/plots/phase4/phase4_humanoidv4_posterior.png) | ![Phase 4 Humanoid performance](bayesian-rl-meets-mcmc/results/plots/phase4/phase4_humanoidv4_performance.png) |
