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
    LEARNING_RATE: float = 0.001
    
    # Trading parameters
    INITIAL_CAPITAL: float = 100000
    MAX_POSITION_SIZE: float = 0.75
    TRANSACTION_COST: float = 0.002
    RISK_PER_TRADE: float = 0.025

    CACHE_DIR: str = '.cache'
    CACHE_LIFESPAN_HOURS: int = 12
    
    # Data parameters
    SYMBOLS: list = None
    LOOKBACK_DAYS: int = 252 * 2  # 2 years
    
    def __post_init__(self):
        if self.SYMBOLS is None:
            self.SYMBOLS = [
                'SPY',    # S&P 500
                'QQQ',    # Nasdaq
                'IWM',    # Small caps
                'AAPL',   # Mega cap tech
                'MSFT',   # Mega cap tech
                'GOOGL',  # Mega cap tech
                'AMZN',   # Mega cap tech
                'TSLA',   # High volatility
                'NVDA',   # AI/Semiconductor
                'META',   # Social media
                'XLF',    # Financial sector
                'XLK',    # Technology sector
                'XLE',    # Energy sector
                'GLD',    # Gold
                'TLT',    # Long-term bonds
            ]


# Initialize configuration
config = TradingConfig()