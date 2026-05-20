# Bayesian RL Meets MCMC

## Research Focus

This project investigates **Bayesian RL Meets MCMC -- Sample Efficiency via Posterior Estimation**, with particular emphasis on the epistemic fragility of point estimates in reinforcement learning under severe data scarcity.

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

This repository currently contains the project scaffold, Phase 1 algorithmic skeletons, documentation, and environment setup material. Full MuJoCo rollout collection, production training loops, posterior diagnostics, and empirical benchmark claims are intentionally deferred.

The first experiment target is **Low-data MuJoCo**. The current report contract is implemented through `scripts/run_low_data_mujoco.py`. Antigravity should always read only the latest files in the `results/` root:

- `results/latest_stats.json`, including PAC-Bayes bound, regret, calibration error, and effective sample size fields.
- `results/latest_mcmc_vs_vi_tradeoff.png`, a visual comparison surface for VAC/VI and SGLD-recalibrated traces.

Timestamped experiment artefacts are retained under `results/archive/`.
Matplotlib visualisations are archived under `results/plots/{phase}/{tag_timestamp}/`, or `results/plots/{phase}/{env_id}/{tag_timestamp}/` when an environment is specified. Antigravity should read the latest plot at the corresponding phase or environment level, for example `results/plots/phase1/latest_comparison.png` or `results/plots/phase2/HalfCheetah-v4/latest_comparison.png`.

The script presently uses deterministic diagnostic traces so that the report pipeline can be validated before the real MuJoCo training loop is introduced.

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

From the repository root, run the Phase 1 low-data MuJoCo diagnostic artefact generator:

```bash
cd bayesian-rl-meets-mcmc
python scripts/run_low_data_mujoco.py --episodes 12 --seed 7 --save-tag phase1_seed7
```

The command writes timestamped result files under `results/archive/` and refreshes root-level `results/latest_stats.json` and `results/latest_mcmc_vs_vi_tradeoff.png`, allowing the report pipeline to verify PAC-Bayes, regret, calibration, and MCMC-VI trade-off fields before full MuJoCo rollout integration.

To launch the MuJoCo training path with TensorBoard logging:

```bash
cd bayesian-rl-meets-mcmc
python scripts/train_bayesian_rl.py --env-id HalfCheetah-v4 --save-tag halfcheetah_phase2
tensorboard --logdir results/archive/tensorboard
```

When `--episodes` is omitted, Phase 2 uses environment-aware diagnostic defaults: Hopper `80`, Ant `150`, HalfCheetah `50`, and Humanoid `200`. Environment-specific stats are archived under `results/archive/{env_id}/`.

To render report-ready plots from TensorBoard logs, or from `latest_stats.json` when TensorBoard logs are unavailable:

```bash
python scripts/visualize_logs.py --phase phase1 --tag seed7
python scripts/visualize_logs.py --phase phase2 --tag halfcheetah_phase2 --env-id HalfCheetah-v4
```

For Phase 3 hyperparameter posterior estimation:

```bash
python scripts/train_bayesian_rl.py --phase phase3 --tag hyper_study --sample_hypers true
python scripts/visualize_logs.py --phase phase3 --tag hyper_study
```

When $\gamma$ and $\alpha$ samples are available, the visualiser also writes hyperparameter posterior trajectories and histograms to `results/plots/{phase}/{tag_timestamp}/` and refreshes `results/plots/{phase}/latest_hyperparameters.png`.

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
