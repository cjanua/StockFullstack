# ai/config/settings.py
import os
from dataclasses import dataclass


@dataclass
class TradingConfig:
    # Alpaca API credentials
    ALPACA_API_KEY: str = os.environ.get('ALPACA_API_KEY')
    ALPACA_SECRET_KEY: str = os.environ.get('ALPACA_SECRET_KEY')
    ALPACA_PAPER: bool = True
    MARKET="IEX" # - 15 min delay (realtime/horizon) or "SIP" for realtime

    # Model parameters
    LSTM_INPUT_SIZE: int = 30
    LSTM_HIDDEN_SIZE: int = 128
    LSTM_NUM_LAYERS: int = 1
    SEQUENCE_LENGTH: int = 60
    LEARNING_RATE: float = 0.0005
    NUM_EPOCHS: int = 250

    USE_ENSEMBLE: bool = False

    # Trading parameters
    INITIAL_CAPITAL: float = 100000
    MAX_POSITION_SIZE: float = 0.25
    TRANSACTION_COST: float = 0.002
    RISK_PER_TRADE: float = 0.015

    CACHE_DIR: str = '.cache'
    CACHE_LIFESPAN_HOURS: int = 12

    # Data parameters
    SYMBOLS: list = None
    LOOKBACK_DAYS: int = 252 * 4  # 4 years

    def __post_init__(self):
        if self.SYMBOLS is None:
            self.SYMBOLS = [
                # Liquid ETFs
                'SPY',    # S&P 500
                # 'QQQ',    # Nasdaq
                # # 'IWM',    # Small caps
                'SPXS',     # Inverse S&P 500
                'TQQQ',  # 3x NASDAQ
                'SQQQ',  # Inverse TQQQ
                'SOXL',  # 3x Semi-Conductors
                'FNGU',  # 3x FAANG-like
                'TSLA',  # High volatility
                'NVDA',  # AI/Semiconductor
                'AMD',  # Social media
                'UVXY',  # 1.5 VIX - Volatility


                # # 'XLF',    # Financial sector
                # # 'XLK',    # Technology sector
                # # 'XLE',    # Energy sector

                # # Commodity ETFs
                # 'GLD',    # Gold
                # # 'TLT',    # Long-term bonds
                # 'USO',    # US Oil
                # # 'URNM',   # Uranium
                # # 'SLV',    # Silver

                # # International ETFs
                # 'EFA',    # Developed markets
                # 'EEM',    # Emerging markets
                # # 'VWO',    # Emerging markets (Vanguard)
                # # 'EWJ',    # Japan
                # # 'EWZ',    # Brazil
                # # 'EWC',    # Canada
                # # 'ARGT',   # Argentina
                # # 'EWW',    # Mexico
                # # 'EWH',    # Hong Kong
                # # 'EWT',    # Taiwan
                # # 'EWY',    # South Korea

                # # Mega caps
                # # 'AAPL',   # Mega cap tech
                # # 'MSFT',   # Mega cap tech
                # # 'GOOGL',  # Mega cap tech
                # # 'AMZN',   # Mega cap tech
                # 'TSLA',   # High volatility
                # # 'NVDA',   # AI/Semiconductor
                # 'META',   # Social media

                # # Dividend ETFs
                # # 'VIG',    # Dividend growth
                # # 'DVY',    # High dividend yield
                # 'VTI',    # Total stock market
                # 'SCHD',   # Quality dividend ETF
            ]


# Initialize configuration
config = TradingConfig()
