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
