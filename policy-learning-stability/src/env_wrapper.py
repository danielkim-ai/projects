import gymnasium as gym
import numpy as np

class DynamicRewardScaleWrapper(gym.RewardWrapper):
    """
    Wraps a continuous control environment to introduce non-stationary 
    reward scaling perturbations alpha_t simulating adversarial or stochastic shifts.
    """
    def __init__(self, env: gym.Env, perturbation_type: str = "sinusoidal"):
        super().__init__(env)
        self.perturbation_type = perturbation_type
        self.timestep = 0

    def reward(self, reward: float) -> float:
        self.timestep += 1
        if self.perturbation_type == "sinusoidal":
            # Dynamic scaling factor alpha_t between 0.1 and 1.0
            alpha_t = 0.55 + 0.45 * np.sin(self.timestep / 100.0)
        elif self.perturbation_type == "collapse":
            # Simulates sudden structural reward degradation
            alpha_t = 0.1 if (200 < self.timestep < 400) else 1.0
        else:
            alpha_t = 1.0
            
        return float(reward * alpha_t)
