import numpy as np
from collections import defaultdict

class MonteCarloAgent:
    """
    Monte Carlo Agent for stock trading, using first-visit updates
    """

    def __init__(self, n_actions, epsilon=0.5, gamma=1.0):
        """
        Initializes the Monte Carlo Agent

        Args:
            n_actions (int): Total number of possible actions in the environment
            epsilon (float): Exploration rate (probability of random action)
            gamma (float): Discount factor for future rewards
        """

        self.Q = defaultdict(lambda: np.zeros(n_actions))       # Initialize Q-table with n_actions zeros
        self.returns = defaultdict(list)                        # Store returns for each (state, action) pair
        self.n_actions = n_actions                                  
        self.epsilon = epsilon                                 
        self.gamma = gamma                                    

        # Environment-specific parameters (to be set during training)
        self.total_companies = None     # Number of companies being traded
        self.budget = None              # Initial trading budget

        self.episode_rewards = []       # Tracks rewards for each episode

    def get_state_key(self, state):
        """
        Converts the continuous state vector into a discrete key for the Q-table.
        State vector format: [stock_counts, market_data, money], where market_data 
        contains [close, return] for each company

        Args:
            state (np.ndarray): Continuous state vector from the environment

        Returns:
            tuple: Discretized state representation for Q-table
        """

        num_companies = self.total_companies

        # Extract components from the state vector
        stock_counts = state[:num_companies]        # Number of stocks held per company
        market_data = state[num_companies:-1]       # Extract the close and return values        
        money = state[-1]                           # Available cash

        # Bin definitions for each variable
        log_close_bins = np.linspace(-2.5, 6.5, 30)
        return_bins = np.linspace(-0.3, 0.3, 20)
        money_bins = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0])

        discrete = []

        discrete.extend([int(x) for x in stock_counts])

        for i in range(num_companies):
            close = market_data[i * 2]
            ret = market_data[i * 2 + 1]

            log_close = np.log(max(close, 1e-4))                        # Transforming into log-close price
            close_bin = np.digitize(log_close, log_close_bins)          # Discretize log(close) into bins defined
            close_bin = max(1, min(close_bin, len(log_close_bins)))     # Values below the lowest bin go to 1, values above the highest go to the last bin

            ret_bin = np.digitize(ret, return_bins)                     # Discretize return
            ret_bin = max(1, min(ret_bin, len(return_bins)))            # Ensure valid bin index

            discrete.append(close_bin)
            discrete.append(ret_bin)

        money_scaled = money / self.budget                              # Normalize money by initial budget
        money_bin = np.digitize(money_scaled, money_bins)               # Discretized normalized money
        money_bin = max(1, min(money_bin, len(money_bins)))             # Ensure valid bin index

        discrete.append(money_bin)

        return tuple(discrete)
    
    def policy(self, state):
        """
        Implements epsilon-greedy policy for action selection. 
        With probability epsilon: random action (exploration),
        otherwise: best known action (exploitation)

        Args:
            state (np.ndarray): Current state of the environment

        Returns:
            int: Index of the chosen action 
        """

        key = self.get_state_key(state)
        if np.random.rand() < self.epsilon:             # Exploration
            return np.random.choice(self.n_actions)
        else:                                           # Exploitation
            return np.argmax(self.Q[key])
        
    def train(self, env, episodes=1000):
        """
        Trains the agent using first-visit Monte Carlo updates.
        For each episode:
        1. Generates a complete episode following current policy
        2. Updates Q-values using first-visit Monte Carlo method
        3. Tracks episode rewards for monitoring

        Args:
            env (market): The stock trading environment
            episodes (int): Number of episodes to train       
        """

        # Initialize environment parameters if not already set
        if self.total_companies is None:
            self.total_companies = env.total_companies    
            self.budget = env.budget

        self.episode_rewards = []

        for ep in range(episodes):
            self.epsilon = max(0.05, self.epsilon / (1 + ep / 100))    # Decaying epsilon overtime until 0.05

            episode = []            # To store tuples: (state, action, reward)
            state = env.start()
            total_reward = 0
            done = False

            # Generate complete episode
            while not done:
                action = self.policy(state)
                next_state, reward, done = env.new_day(action)
                episode.append((self.get_state_key(state), action, reward))
                state = next_state
                total_reward += reward
            
            self.episode_rewards.append(total_reward)

            # First-visit Monte Carlo update
            G = 0
            visited = set()
            for t in reversed(range(len(episode))):
                state_key, action, reward = episode[t]
                G = self.gamma * G + reward
                if (state_key, action) not in visited:
                    visited.add((state_key, action))
                    self.returns[(state_key, action)].append(G)
                    self.Q[state_key][action] = np.mean(self.returns[(state_key, action)])