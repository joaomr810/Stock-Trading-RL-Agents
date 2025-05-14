import numpy as np

class RandomAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        
    def policy(self, state):
        return np.random.choice(self.n_actions)