# ai/config/settings.py
import os
from dataclasses import dataclass

@dataclass
class TradingConfig:
    # Alpaca API credentials
    ALPACA_API_KEY: str = os.environ.get('ALPACA_API_KEY')
    ALPACA_SECRET_KEY: str = os.environ.get('ALPACA_SECRET_KEY')
    ALPACA_PAPER: bool = True
    
    # Model parameters
    LSTM_INPUT_SIZE: int = 30
    LSTM_HIDDEN_SIZE: int = 128
    LSTM_NUM_LAYERS: int = 1
    SEQUENCE_LENGTH: int = 60
    LEARNING_RATE: float = 3e-4
    
    # Trading parameters
    INITIAL_CAPITAL: float = 100000
    MAX_POSITION_SIZE: float = 0.95
    TRANSACTION_COST: float = 0.002
    RISK_PER_TRADE: float = 0.02

    CACHE_DIR: str = '.cache'
    CACHE_LIFESPAN_HOURS: int = 12
    
    # Data parameters
    SYMBOLS: list = None
    LOOKBACK_DAYS: int = 252 * 2  # 2 years
    
    def __post_init__(self):
        if self.SYMBOLS is None:
            self.SYMBOLS = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Initialize configuration
config = TradingConfig()