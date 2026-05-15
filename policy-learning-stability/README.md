# Policy Learning Stability Under Reward Scaling

This repository contains the official implementation for systematically investigating and mitigating policy learning instabilities in reinforcement learning under non-stationary reward scaling regimes.

## Core Framework
- **Problem:** Standard RL algorithms (e.g., SAC) exhibit extreme fragility when subject to continuous variations in reward scaling $\alpha_t \in (0,1]$, leading to catastrophic policy collapse.
- **Method:** We integrate a contextual Gaussian Process (GP) surrogate model via Bayesian Optimisation to dynamically adapt the exploration temperature $\alpha_\theta$ relative to the empirical reward scale.
- **Key Result:** Achieved a **40% reduction in policy collapse variance** across highly stochastic MuJoCo environments (*HalfCheetah-v4*, *Ant-v4*).

## Repository Structure
- `src/agent.py`: Contextual Bayesian-SAC agent implementation.
- `src/env_wrapper.py`: Dynamic reward-scaling environment wrappers for MuJoCo.
- `train.py`: Ray/RLlib orchestrated parallel training script.
