import numpy as np
import gymnasium as gym


class RandomPolicy:

    def __init__(self, action_space: gym.spaces.Discrete, seed: int = 0):
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(0, self.action_space.n))

    def reset(self):
        pass


class HeuristicPolicy:

    def select_action(self, obs: np.ndarray) -> int:
        x, y, vx, vy, theta, theta_dot, leg_l, leg_r = obs

        if theta > 0.2:
            return 3 
        if theta < -0.2:
            return 1 

        if vy < -0.3:
            return 2

        if vx > 0.3:
            return 1
        if vx < -0.3:
            return 3

        return 0

    def reset(self):
        pass