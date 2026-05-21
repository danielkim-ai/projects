# Bayesian RL Meets MCMC

## Research Focus

This project investigates **Bayesian RL Meets MCMC -- Sample Efficiency via Posterior Estimation**, with particular emphasis on the epistemic fragility of point estimates in reinforcement learning under severe data scarcity.

Status: **Successfully validated across major MuJoCo benchmarks (Phase 1-3)**, including HalfCheetah-v4, Ant-v4, Hopper-v4, and Humanoid-v4.

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

This repository currently contains the project scaffold, Phase 1 diagnostic pipeline, Phase 2 MuJoCo training loop, Phase 3 hyperparameter posterior sampling, Phase 4 privacy-preserving DP-SGLD hooks, documentation, and environment setup material. Phases 1-3 have been validated across HalfCheetah-v4, Ant-v4, Hopper-v4, and Humanoid-v4.

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

The next research step is to combine Differential Privacy (DP) with SGLD so that posterior sampling noise and Gaussian privacy noise jointly support safer reinforcement learning without avoidable performance degradation. The Phase 4 implementation path will compare privacy strengths on Ant-v4 and Humanoid-v4 while tracking return, calibration, regret, and cumulative privacy cost.

DP-SGLD experiments activate gradient clipping, Gaussian privacy noise, and TensorBoard privacy-budget traces. Use Ant and Humanoid to inspect how stronger privacy constraints affect expected return, calibration, and posterior recalibration:

```bash
!python scripts/train_bayesian_rl.py --env-id Ant-v4 --episodes 150 --phase phase4 --tag ant_privacy_eps8 --privacy true --epsilon 8.0
!python scripts/train_bayesian_rl.py --env-id Humanoid-v4 --episodes 200 --phase phase4 --tag humanoid_privacy_eps4 --privacy true --epsilon 4.0
!python scripts/visualize_logs.py --phase phase4 --tag ant_privacy_eps8 --env-id Ant-v4
```

The cumulative privacy cost is logged as `privacy/epsilon_spent`, with the target budget saved in `results/latest_stats.json` under the `privacy` field.

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
