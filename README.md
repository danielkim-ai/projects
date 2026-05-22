# Research & Engineering Projects Archive

A curated, centralised repository capturing foundational and empirical implementations spanning Reinforcement Learning, Applied Statistics, and Sequential Decision-Making under Uncertainty.

## Core Research Directory
- `/policy-learning-stability`: Robust policy gradients under non-stationary reward scaling regimes. (Completed)
- `/bayesian-rl-meets-mcmc`: Bayesian reinforcement learning with MCMC-VI posterior estimation for sample-efficient uncertainty quantification. Phase 1-4 complete, including Privacy-Preserving Analysis and the final Differential Privacy Integration milestone.
- `/trustworthy-offline-rl-via-dp`: Trajectory-level DP-SGD, privacy-aware offline CQL, LiSSA unlearning, SISA episode shards, and MIA verification for regulated offline RL logs.

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

## Stability & Regret Analysis

Frequentist epsilon-greedy exploration can accumulate linear regret, $O(T)$, when point-estimated value functions remain miscalibrated in data-scarce regimes. The Bayesian RL project instead uses posterior sampling: Thompson-style policies are drawn from the learned posterior, giving the standard sub-linear Bayesian regret target $\tilde{O}(\sqrt{dT})$.

![Phase 2 HalfCheetah stability and regret comparison](bayesian-rl-meets-mcmc/results/plots/phase2/phase2_halfcheetahv4_performance.png)

### Phase 4 Privacy-Preserving Performance

| Environment | Posterior | Performance |
| --- | --- | --- |
| HalfCheetah-v4 | ![Phase 4 HalfCheetah posterior](bayesian-rl-meets-mcmc/results/plots/phase4/phase4_halfcheetahv4_posterior.png) | ![Phase 4 HalfCheetah performance](bayesian-rl-meets-mcmc/results/plots/phase4/phase4_halfcheetahv4_performance.png) |
| Ant-v4 | ![Phase 4 Ant posterior](bayesian-rl-meets-mcmc/results/plots/phase4/phase4_antv4_posterior.png) | ![Phase 4 Ant performance](bayesian-rl-meets-mcmc/results/plots/phase4/phase4_antv4_performance.png) |
| Hopper-v4 | ![Phase 4 Hopper posterior](bayesian-rl-meets-mcmc/results/plots/phase4/phase4_hopperv4_posterior.png) | ![Phase 4 Hopper performance](bayesian-rl-meets-mcmc/results/plots/phase4/phase4_hopperv4_performance.png) |
| Humanoid-v4 | ![Phase 4 Humanoid posterior](bayesian-rl-meets-mcmc/results/plots/phase4/phase4_humanoidv4_posterior.png) | ![Phase 4 Humanoid performance](bayesian-rl-meets-mcmc/results/plots/phase4/phase4_humanoidv4_performance.png) |

## Download Technical Brief (LaTeX)

The full archive file is available at `bayesian-rl-meets-mcmc/docs/technical_brief.tex`.

```latex
\documentclass[conference]{IEEEtran}
\usepackage{amsmath,amssymb,graphicx,booktabs,hyperref}
\title{Bayesian RL Meets MCMC: Posterior Estimation for Sample-Efficient and Privacy-Preserving Control}
\author{\IEEEauthorblockN{Research Engineering Brief}}
\begin{document}
\maketitle
\begin{abstract}
We summarise a four-phase Bayesian reinforcement learning project validated on HalfCheetah-v4, Ant-v4, Hopper-v4, and Humanoid-v4.
\end{abstract}
\section{Method}
The method combines variational actor-critic updates with SGLD posterior recalibration, hyperparameter posterior estimation, and differential privacy analysis.
\section{Regret}
Epsilon-greedy exploration may incur $O(T)$ regret under severe data scarcity, whereas Bayesian Thompson Sampling targets $\tilde{O}(\sqrt{dT})$ Bayesian regret.
\section{Conclusion}
Posterior estimation improves the stability and auditability of reinforcement learning under scarce data and privacy constraints.
\end{document}
```
