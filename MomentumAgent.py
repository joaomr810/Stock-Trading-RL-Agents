import numpy as np

class MomentumAgent:
    def __init__(self, n_actions, num_companies, window=3):
        self.n_actions = n_actions
        self.num_companies = num_companies
        self.window = window
        self.history = []

    def policy(self, state):
        prices = state[self.num_companies:2*self.num_companies]
        self.history.append(prices)

        if len(self.history) < self.window + 1:
            return np.random.choice(self.n_actions)

        # Calculate momentum
        momentum = momentum = self.history[-1] - self.history[-self.window-1]
        actions = []

        for m in momentum:
            if m > 0:
                actions.append(2)  # Buy
            elif m < 0:
                actions.append(0)  # Sell
            else:
                actions.append(1)  # Hold

        # Convert to index
        action_index = 0
        for i, a in enumerate(actions):
            action_index += a * (3 ** (self.num_companies - i - 1))

        return action_index
