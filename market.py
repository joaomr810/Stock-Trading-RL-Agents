import numpy as np
import pandas as pd
import itertools

def get_stock_data(companies, start_date='2018-04-01'):
    """
    Loads and processes stock data for multiple companies

    Args:
        companies (list): List of company symbols to load data
        start_date (str): Start date for data collection (YYYY-MM-DD)

    Returns:
        np.ndarray: Array containing Close and Return values for all companies
    """

    output = pd.DataFrame(columns=['Date'])
    is_first = True
    for company in companies:
        df = pd.read_csv(f'dataset/stocks/{company}.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[df['Date'] >= pd.to_datetime(start_date)]
        df = df.sort_values('Date')

        df['Return'] = df['Close'].pct_change()
        df = df[['Date', 'Close', 'Return']].dropna()

        df = df.rename(columns={
            'Close': f'{company}_Close',
            'Return': f'{company}_Return'
        })

        if is_first:
            output = df.copy()
            is_first = False
        else:
            output = output.merge(df, on='Date')

    output = output.drop(columns=['Date'])
    return output.values

class market:
    """
    Stock trading environment that simulates buying and selling stocks over time.
    Supports multiple companies and tracks portfolio value changes
    """
    def __init__(self, companies, budget=1e4):
        """
        Sets up the trading environment

        Args:
            companies (list): List of company symbols to trade
            budget (float): Starting money for stock trading
        """
        self.data = get_stock_data(companies)
        self.budget = budget
        self.total_days = self.data.shape[0]
        self.total_companies = len(companies)

        # Define action space for each company: 0 = sell, 1 = hold, 2 = buy
        self.index_actions = np.arange(3**self.total_companies)
        self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.total_companies)))

        # State includes: stock counts + market info (Close and Return) + money available
        self.state_size = self.total_companies + self.data.shape[1] + 1 
        self.start()

    def get_episode_value(self):
        """
        Returns the current total value of the portfolio (stocks + cash)
        """
        return self._get_eval()
    
    def start(self):
        """
        Resets the environment to initial state

        Returns:
            np.ndarray: Initial state vector containing stock counts, market data, and cash
        """

        self.today = 0
        self.stock_counts = np.zeros(self.total_companies)
        self.stock_price = self.data[self.today]
        self.money_available = self.budget
        return self._get_state()
    
    def new_day(self, action):
        """
        Advances to next trading day and applies trading action

        Args:
            action (int): Index of the action to take (from action_list)
        
        Returns:
            tuple: (new_state, reward, done)
        """

        previous_val = self._get_eval()
        self.today += 1
        self.stock_price = self.data[self.today]
        self._exchange(action)
        current_val = self._get_eval()
        reward = current_val - previous_val
        done = self.today == (self.total_days - 1)
        return self._get_state(), reward, done
    
    def _exchange(self, action):
        """
        Executes buying and selling of stocks based on the action

        Args:
            action (int): Index of the action to take (from action_list)
        """

        actions = self.action_list[action]
        close_prices = self.stock_price[::2]

        # Selling stocks:
        for i, a in enumerate(actions):
            if a == 0 and self.stock_counts[i] > 0:
                self.money_available += close_prices[i] * self.stock_counts[i]
                self.stock_counts[i] = 0

        # Buy 1 unit of each stock marked as 'buy', until availble money is exhausted
        buy_list = [i for i, a in enumerate(actions) if a == 2]
        while True:
            bought_any = False
            for i in buy_list:
                if self.money_available >= close_prices[i]:
                    self.money_available -= close_prices[i]
                    self.stock_counts[i] += 1
                    bought_any = True
            if not bought_any:
                break
    
    def _get_eval(self):
        """
        Calculates current portfolio value

        Returns:
            float: Total value of stocks and cash
        """

        close_prices = self.stock_price[::2]
        return self.stock_counts.dot(close_prices) + self.money_available

    def _get_state(self):
        """
        Creates the current state vector

        Returns:
            np.ndarray: Vector containing stock counts, market data, and cash
        """

        state = np.zeros(self.state_size)
        state[:self.total_companies] = self.stock_counts
        state[self.total_companies:-1] = self.stock_price
        state[-1] = self.money_available
        return state