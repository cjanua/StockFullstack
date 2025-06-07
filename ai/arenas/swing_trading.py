import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import A2C

from stable_baselines3.common.vec_env import SubprocVecEnv
import multiprocessing as mp

from ai.arenas.swing_trading import SwingTradingEnv
# from ai.models.a3c import A3CTradingAgent


class SwingTradingEnv(gym.Env):
    def __init__(self, data, initial_capital=100000, transaction_cost=0.001):
        super().__init__()
        
        self.data = data
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # Action space: continuous position allocation [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation space: market features + portfolio state
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(60, 25), dtype=np.float32  # 60 timesteps, 25 features
        )
        
        # Market microstructure simulation
        self.bid_ask_spread = 0.001
        self.market_impact = 0.0005
        self.slippage_factor = 0.0002
        
    def step(self, action):
        # Execute trade with realistic costs
        execution_price, costs = self.simulate_execution(
            action[0], 
            self.data.iloc[self.current_step]['close']
        )
        
        # Update portfolio
        self.execute_portfolio_update(action[0], execution_price, costs)
        
        # Calculate sophisticated reward
        reward = self.calculate_reward()
        
        # Generate next observation
        obs = self.get_observation()
        
        # Check termination conditions
        done = self.is_episode_complete()
        
        return obs, reward, done, False, self.get_info()
    
    def calculate_reward(self):
        """Multi-objective reward function optimizing risk-adjusted returns"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        # Portfolio return component
        portfolio_return = (
            self.portfolio_history[-1] - self.portfolio_history[-2]
        ) / self.portfolio_history[-2]
        
        # Risk-adjusted component (rolling Sharpe ratio)
        if len(self.portfolio_history) > 30:
            recent_returns = np.diff(self.portfolio_history[-30:]) / self.portfolio_history[-31:-1]
            sharpe = np.mean(recent_returns) / np.std(recent_returns) if np.std(recent_returns) > 0 else 0
            risk_adjustment = 0.1 * sharpe
        else:
            risk_adjustment = 0
        
        # Transaction cost penalty
        trading_penalty = -0.001 * (self.trades_count / max(self.current_step, 1))
        
        return (portfolio_return + risk_adjustment + trading_penalty) * 100
    
    def simulate_execution(self, action, market_price):
        """Realistic trade execution simulation"""
        # Bid-ask spread
        if action > 0:  # Buying
            execution_price = market_price * (1 + self.bid_ask_spread/2)
        else:  # Selling
            execution_price = market_price * (1 - self.bid_ask_spread/2)
        
        # Market impact (square root law)
        trade_size = abs(action * self.portfolio_value)
        impact = self.market_impact * np.sqrt(trade_size / self.average_volume)
        execution_price *= (1 + impact) if action > 0 else (1 - impact)
        
        # Random slippage
        slippage = np.random.normal(0, self.slippage_factor)
        execution_price *= (1 + slippage)
        
        # Total transaction costs
        costs = trade_size * self.transaction_cost
        
        return execution_price, costs
    

def create_vectorized_env(data_splits, n_envs=8):
    """Create parallel trading environments for faster training"""
    
    def make_env(rank):
        def _init():
            env_data = data_splits[rank]
            env = SwingTradingEnv(env_data)
            env.seed(1000 + rank)
            return env
        return _init
    
    env_fns = [make_env(i) for i in range(n_envs)]
    return SubprocVecEnv(env_fns)

# Training with vectorized environments
def train_agent_parallel():
    # Split data across different time periods
    data_splits = split_data_by_periods(stock_data, n_periods=8)
    vec_env = create_vectorized_env(data_splits)
    
    # Train A3C agent
    model = A2C(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=512,
        gamma=0.99,
        verbose=1
    )
    
    model.learn(total_timesteps=500000)
    return model