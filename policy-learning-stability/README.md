# Policy Learning Stability Under Reward Scaling

This project studies the fragility of entropy-regularised reinforcement learning when the reward signal is subjected to non-stationary scale perturbations. The implementation couples a Gymnasium reward wrapper with a contextual Bayesian-SAC core whose entropy temperature is selected through a Gaussian Process surrogate.

## Problem Setting

Let the environment emit a native reward \(r_t\). We study perturbed returns under a time-dependent multiplier

\[
\tilde{r}_t = \alpha_t r_t, \qquad \alpha_t \in (0, 1].
\]

The central question is whether a policy can retain stable exploration and value estimation when \(\alpha_t\) follows either smooth sinusoidal drift or an abrupt collapse regime. Static reward assumptions become brittle because the entropy-temperature coefficient that is appropriate at \(\alpha_t \approx 1\) can become substantially miscalibrated when \(\alpha_t\) contracts.

## Static Entropy Coefficients vs Dynamic Bayesian Priors

Static SAC fixes or slowly adapts its entropy coefficient without direct knowledge of the reward-scale process. Under non-stationary scaling, this can over-regularise the actor during reward collapse or under-regularise it when the reward scale recovers.

The dynamic Bayesian-prior approach treats \(\alpha_t\) as contextual evidence. A Gaussian Process surrogate models the relationship between observed reward scales, realised episode returns and useful entropy temperatures. The agent then samples its next temperature from a posterior that includes both the mean estimate and uncertainty variance, preserving exploratory caution when evidence is sparse.

## Implementation Structure

- `src/env_wrapper.py`: Implements sinusoidal and abrupt-collapse reward perturbations through a Gymnasium `RewardWrapper`.
- `src/agent.py`: Defines the Actor-Critic scaffold, Gaussian Process surrogate and Bayesian temperature sampling logic.
- `train.py`: Registers MuJoCo environments for Ray execution and sketches the episode-level loop in which collected rewards update the surrogate model.

## Research Aim

The repository provides a compact experimental basis for analysing policy collapse variance under changing reward magnitudes, with emphasis on principled uncertainty handling rather than ad hoc temperature schedules.

## Getting Started

### Prerequisites

- MuJoCo installation.
- Python 3.9+ environment.

### Installation

```bash
git clone https://github.com/danielkim-ai/projects.git
cd projects/policy-learning-stability
pip install -r requirements.txt
```

For platform-specific guidance, including Apple Silicon, Windows, Linux, `grpcio` build issues and the `linear_operator` pin, see `INSTALL.md`.

### Usage

- For sinusoidal drift: `python train.py --perturbation sinusoidal`
- For abrupt collapse: `python train.py --perturbation collapse`
- Use `--episodes` to control the experimental horizon. The default is `500`, while longer runs provide more stable return-variance estimates.
- For automated visualisation: `python -m src.visualise --input results/experiment_log.csv --output-dir results`

```bash
# Run full experimental suite
python train.py --episodes 1000 --perturbation sinusoidal

# Generate publication-quality plots
python -m src.visualise --input results/experiment_log.csv
```

## Execution

Scenario A (Sinusoidal Drift):

```bash
python train.py --perturbation sinusoidal
```

Scenario B (Abrupt Collapse):

```bash
python train.py --perturbation collapse
```

## Monitoring

Use the Ray dashboard and process logs to monitor Bayesian Temperature Adaptation in real time, including posterior temperature samples, reward-scale traces and episode-level return summaries.

## Citation

```bibtex
@misc{kim2026rewardscale,
  author = {Daniel Kim},
  title = {Robust Policy Learning under Reward-Scale Variation},
  year = {2026},
  note = {Policy Learning Stability Under Reward Scaling}
}
```
