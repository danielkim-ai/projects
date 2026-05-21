# Bayesian RL Meets MCMC

## Research Focus

This project investigates **Bayesian RL Meets MCMC -- Sample Efficiency via Posterior Estimation**, with particular emphasis on the epistemic fragility of point estimates in reinforcement learning under severe data scarcity.

Status: **Phase 1-4 complete, including Privacy-Preserving Analysis**. The project has been validated across major MuJoCo benchmarks, including HalfCheetah-v4, Ant-v4, Hopper-v4, and Humanoid-v4, with **Differential Privacy Integration** serving as the final research milestone.

### Problem

Standard policy optimisation methods such as PPO and SAC commonly rely on point-estimated policy and value parameters. In low-data regimes, this practice can induce brittle exploration, poorly calibrated value estimates, and overconfident policy updates. The central concern is not merely variance in returns, but the misrepresentation of epistemic uncertainty as though it were aleatoric noise or optimisation error.

### Solution

We study a hybrid **MCMC-Variational Inference (VI)** framework for calibrated uncertainty quantification in actor-critic reinforcement learning. The intended programme is to combine tractable variational objectives with posterior sampling methods that better preserve uncertainty over policies, critics, and selected hyperparameters.

### Core Methods

- **MCMC-Augmented Policy Optimisation:** Using SGLD/SGHMC for approximate posterior sampling over policy and critic parameters, thereby replacing single-point updates with posterior-aware optimisation dynamics.
- **Variational Actor-Critic (VAC):** Marginalising over parameter uncertainty through an evidence lower bound (ELBO), using objectives of the form

$$\mathcal{L}_{\text{ELBO}}(\phi) = \mathbb{E}_{q_{\phi}(\theta)} \left[ \log p(\mathcal{D} \mid \theta) \right] - \text{KL}(q_{\phi}(\theta) \,\|\, p(\theta))$$

- **Bayesian Hyperparameter Inference:** Treating discount, entropy, and learning-rate parameters such as $\gamma$, $\alpha$, and $\eta$ as random variables rather than fixed constants:

$$p(\theta, \gamma, \alpha, \eta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta, \gamma, \alpha, \eta) p(\theta) p(\gamma) p(\alpha) p(\eta)$$

## Phase 1 Implementation

```text
bayesian-rl-meets-mcmc/
  configs/   Hyperparameter prior specifications and experiment templates.
  docs/      Technical notes, theoretical derivations, and internal references.
  results/   Latest report artefacts at root; timestamped runs in archive/.
  scripts/   Execution entry points for future training and evaluation workflows.
  requirements.txt and environment.yaml live at the project root.
  src/       Phase 1 SGLD, VAC, hybrid recalibration, and metric utilities.
```

## Current Status

This repository currently contains the project scaffold, Phase 1 diagnostic pipeline, Phase 2 MuJoCo training loop, Phase 3 hyperparameter posterior sampling, Phase 4 privacy-preserving DP-SGLD hooks, privacy analysis tooling, documentation, and environment setup material. Phases 1-4 have been validated across HalfCheetah-v4, Ant-v4, Hopper-v4, and Humanoid-v4.

The first experiment target is **Low-data MuJoCo**. The current report contract is implemented through `scripts/run_low_data_mujoco.py`. Antigravity should always read only the latest files in the `results/` root:

- `results/latest_stats.json`, including PAC-Bayes bound, regret, calibration error, and effective sample size fields.
- `results/latest_mcmc_vs_vi_tradeoff.png`, a visual comparison surface for VAC/VI and SGLD-recalibrated traces.

Timestamped experiment artefacts are retained under `results/archive/`.
Matplotlib visualisations are archived under `results/plots/{phase}/{tag_timestamp}/`, or `results/plots/{phase}/{env_id}/{tag_timestamp}/` when an environment is specified. Antigravity should read the latest plot at the corresponding phase or environment level, for example `results/plots/phase1/latest_comparison.png` or `results/plots/phase2/HalfCheetah-v4/latest_comparison.png`.

The Phase 1 diagnostic script retains deterministic low-data traces for reproducible report validation, while Phases 2 and 3 use the MuJoCo training loop and TensorBoard log pipeline.

## Algorithmic Skeleton

- `src/sgld.py`: Stochastic Gradient Langevin Dynamics updates for approximate posterior sampling at SGD-like cost.
- `src/vac.py`: Mean-field variational actor-critic modules and an ELBO-style objective,

$$\mathcal{L}(\phi) = \mathbb{E}_{q_{\phi}(\theta)} \left[ \sum_t r_t \right] - \beta \cdot D_{\text{KL}}(q_{\phi}(\theta) \,\|\, p(\theta))$$

- `src/hybrid_recalibration.py`: A hybrid schedule in which VAC supplies online variational updates and SGLD periodically recalibrates the posterior approximation, especially when KL or effective sample size diagnostics suggest mode collapse.
- `src/metrics.py`: Report-facing PAC-Bayes, regret, calibration, and effective sample size utilities.

## Initial Research Questions

- Can posterior sampling over policy and critic parameters reduce sample complexity relative to point-estimated PPO or SAC under restricted data budgets?
- How should SGLD or SGHMC noise be calibrated when the likelihood term is induced by bootstrapped temporal-difference targets?
- When does the variational approximation understate posterior uncertainty, and can intermittent MCMC correction mitigate that failure mode?
- Which hyperparameters, among $\gamma$, $\alpha$, and $\eta$, most materially affect Bayesian regret and posterior calibration?
- How does the posterior over policy parameters evolve as rollout data arrive online, and what concept drift signals should trigger posterior discounting?
- Can PAC-Bayes bounds be made tight enough to guide continuous-control policy selection, rather than merely supplying a retrospective certificate?

## Quick Start

From the project directory, use the following command templates. They are written with the Colab `!` prefix so they can be copied directly into a notebook cell after installation.

Note: Always specify `--env-id` for environment-dependent training and visualisation to ensure environment-specific optimisation.

### Phase 1 Diagnostic

```bash
!python scripts/run_low_data_mujoco.py --env-id HalfCheetah-v4 --episodes 12 --seed 7 --save-tag phase1_seed7
!python scripts/visualize_logs.py --phase phase1 --tag seed7 --env-id HalfCheetah-v4
```

The command writes timestamped result files under `results/archive/` and refreshes root-level `results/latest_stats.json` and `results/latest_mcmc_vs_vi_tradeoff.png`, allowing the report pipeline to verify PAC-Bayes, regret, calibration, and MCMC-VI trade-off fields before full MuJoCo rollout integration.

### Phase 2 Performance

```bash
!python scripts/train_bayesian_rl.py --env-id HalfCheetah-v4 --episodes 50 --save-tag halfcheetah_phase2
!python scripts/visualize_logs.py --phase phase2 --tag halfcheetah_phase2 --env-id HalfCheetah-v4
```

When `--episodes` is omitted, Phase 2 uses environment-aware diagnostic defaults: Hopper `80`, Ant `150`, HalfCheetah `50`, and Humanoid `200`. Environment-specific stats are archived under `results/archive/{env_id}/`.

To inspect TensorBoard logs after a training run:

```bash
!tensorboard --logdir results/archive/tensorboard
```

### Phase 3 Automated Hyperparameter Optimisation

Phase 3 extends the posterior estimation programme from policy parameters to RL hyperparameters. SGLD posterior estimation over $\gamma$ and $\alpha$ produced stable performance even in the high-dimensional Humanoid-v4 setting, supporting the thesis that Bayesian hyperparameter optimisation can reduce brittle manual tuning under scarce rollout data.

```bash
!python scripts/train_bayesian_rl.py --env-id HalfCheetah-v4 --episodes 200 --phase phase3 --tag hyper_study --sample_hypers true
!python scripts/visualize_logs.py --phase phase3 --tag hyper_study --env-id HalfCheetah-v4
```

When $\gamma$ and $\alpha$ samples are available, the visualiser also writes hyperparameter posterior trajectories and histograms to `results/plots/{phase}/{tag_timestamp}/` and refreshes `results/plots/{phase}/latest_hyperparameters.png`.

## Next Steps

### Phase 4 Privacy-Preserving Reinforcement Learning

The final milestone is **Differential Privacy Integration**: combining Differential Privacy (DP) with SGLD so that posterior sampling noise and Gaussian privacy noise jointly support safer reinforcement learning without avoidable performance degradation. The Phase 4 analysis compares privacy strengths while tracking return, calibration, regret, and cumulative privacy cost.

DP-SGLD experiments activate gradient clipping, Gaussian privacy noise, and TensorBoard privacy-budget traces. Use Ant and Humanoid to inspect how stronger privacy constraints affect expected return, calibration, and posterior recalibration:

```bash
!python scripts/train_bayesian_rl.py --env-id Ant-v4 --episodes 150 --phase phase4 --tag ant_privacy_eps8 --privacy true --epsilon 8.0
!python scripts/train_bayesian_rl.py --env-id Humanoid-v4 --episodes 200 --phase phase4 --tag humanoid_privacy_eps4 --privacy true --epsilon 4.0
!python scripts/visualize_logs.py --phase phase4 --tag ant_privacy_eps8 --env-id Ant-v4
```

The cumulative privacy cost is logged as `privacy/epsilon_spent`, with the target budget saved in `results/latest_stats.json` under the `privacy` field.

For a HalfCheetah-v4 epsilon sensitivity ablation, run the same privacy-enabled training protocol at three privacy strengths and then generate the analysis figures:

```bash
# Step 1: strong privacy, larger expected utility cost.
!python scripts/train_bayesian_rl.py --env-id HalfCheetah-v4 --episodes 50 --phase phase4 --tag halfcheetah_privacy_eps1 --privacy true --epsilon 1.0

# Step 2: moderate privacy baseline.
!python scripts/train_bayesian_rl.py --env-id HalfCheetah-v4 --episodes 50 --phase phase4 --tag halfcheetah_privacy_eps4 --privacy true --epsilon 4.0

# Step 3: weaker privacy, smaller expected utility cost.
!python scripts/train_bayesian_rl.py --env-id HalfCheetah-v4 --episodes 50 --phase phase4 --tag halfcheetah_privacy_eps16 --privacy true --epsilon 16.0

# Step 4: analyse epsilon sensitivity and privacy loss.
!python scripts/analyse_privacy.py --env-id HalfCheetah-v4 --epsilons 1.0 4.0 16.0
```

Phase 4 TensorBoard logs are written to `results/logs/phase4/{tag}/`, for example `results/logs/phase4/halfcheetah_privacy_eps4/`. The analyser supports absolute paths and paths relative to either the shell working directory or the project root via `--log-dir`.

The analysis script writes `epsilon_sensitivity_*.png` and `privacy_loss_*.png` to `results/plots/phase4/analysis/`, alongside `latest_epsilon_sensitivity.png` and `latest_privacy_loss.png` for report automation.

## Experimental Results Gallery

The repository archives plot artefacts under `results/plots/{phase}/` using the official `{phase}_{env_id}_{type}.png` naming convention.

### Phase 1 Diagnostic

![Phase 1 HalfCheetah diagnostic comparison](results/plots/phase1/phase1_halfcheetahv4_diagnostic.png)

### Phase 3 Posterior Estimation

| Environment | Posterior | Performance |
| --- | --- | --- |
| HalfCheetah-v4 | ![Phase 3 HalfCheetah posterior](results/plots/phase3/phase3_halfcheetahv4_posterior.png) | ![Phase 3 HalfCheetah performance](results/plots/phase3/phase3_halfcheetahv4_performance.png) |
| Ant-v4 | ![Phase 3 Ant posterior](results/plots/phase3/phase3_antv4_posterior.png) | ![Phase 3 Ant performance](results/plots/phase3/phase3_antv4_performance.png) |
| Hopper-v4 | ![Phase 3 Hopper posterior](results/plots/phase3/phase3_hopperv4_posterior.png) | ![Phase 3 Hopper performance](results/plots/phase3/phase3_hopperv4_performance.png) |
| Humanoid-v4 | ![Phase 3 Humanoid posterior](results/plots/phase3/phase3_humanoidv4_posterior.png) | ![Phase 3 Humanoid performance](results/plots/phase3/phase3_humanoidv4_performance.png) |

### Differential Privacy Analysis

| Environment | Epsilon Sensitivity | Privacy Budget Consumption |
| --- | --- | --- |
| HalfCheetah-v4 | ![Phase 4 HalfCheetah epsilon sensitivity](results/plots/phase4/analysis/phase4_halfcheetahv4_epsilon_sensitivity.png) | ![Phase 4 HalfCheetah privacy budget](results/plots/phase4/analysis/phase4_halfcheetahv4_privacy_budget.png) |

### Phase 4 Privacy-Preserving Performance

| Environment | Posterior | Performance |
| --- | --- | --- |
| HalfCheetah-v4 | ![Phase 4 HalfCheetah posterior](results/plots/phase4/phase4_halfcheetahv4_posterior.png) | ![Phase 4 HalfCheetah performance](results/plots/phase4/phase4_halfcheetahv4_performance.png) |
| Ant-v4 | ![Phase 4 Ant posterior](results/plots/phase4/phase4_antv4_posterior.png) | ![Phase 4 Ant performance](results/plots/phase4/phase4_antv4_performance.png) |
| Hopper-v4 | ![Phase 4 Hopper posterior](results/plots/phase4/phase4_hopperv4_posterior.png) | ![Phase 4 Hopper performance](results/plots/phase4/phase4_hopperv4_performance.png) |
| Humanoid-v4 | ![Phase 4 Humanoid posterior](results/plots/phase4/phase4_humanoidv4_posterior.png) | ![Phase 4 Humanoid performance](results/plots/phase4/phase4_humanoidv4_performance.png) |

### Troubleshooting Phase 4 Paths

If the analyser cannot find TensorBoard logs, first inspect the folders under `results/logs/phase4/`. The analyser uses fuzzy matching, so tags such as `halfcheetah_privacy_eps4`, `HalfCheetah_epsilon4_seed7`, or `privacy_eps4_halfcheetah` are all acceptable provided the environment and epsilon cues are present.

When running from a different working directory, pass the log root explicitly:

```bash
!python scripts/analyse_privacy.py --env-id HalfCheetah-v4 --epsilons 1.0 4.0 16.0 --log-dir /absolute/path/to/results/logs/phase4
```

If no matching data are found, the analyser prints both the paths it attempted and the existing folders below the searched log roots.

For Google Colab, mount Google Drive, create the working notebook directory if needed, clone the repository there, and move into the actual project directory before installing dependencies:

```python
from google.colab import drive
drive.mount("/content/drive")

# Create and enter the working directory.
!mkdir -p "/content/drive/MyDrive/Colab Notebooks"
%cd /content/drive/MyDrive/Colab\ Notebooks/

# Clone the repository and enter the Bayesian RL project.
!git clone https://github.com/danielkim-ai/projects.git
%cd projects/bayesian-rl-meets-mcmc
!pip install -r requirements.txt
!python scripts/run_low_data_mujoco.py --save-tag colab_phase1
```
