# alpaca/sdk/clients.py
from typing import Optional
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.common.exceptions import APIError

from backend.alpaca.core import AlpacaConfig
from backend.alpaca.core import logger

class AlpacaClientManager:
    """Centralized management of Alpaca API clients"""
    
    _trading_client: Optional[TradingClient] = None
    _historical_client: Optional[StockHistoricalDataClient] = None
    
    @classmethod
    def get_trading_client(cls) -> TradingClient:
        """Get or create a trading client"""
        if not cls._trading_client:
            try:
                key, secret, _ = AlpacaConfig.get_credentials()
                cls._trading_client = TradingClient(key, secret, paper=False)
                
                # Verify client connection
                cls._trading_client.get_account()
                logger.info("Trading client created and verified")
            except (ValueError, APIError) as e:
                logger.error(f"Error creating trading client: {e}")
                raise
        
        return cls._trading_client
    
    @classmethod
    def get_historical_client(cls) -> StockHistoricalDataClient:
        """Get or create a historical data client"""
        if not cls._historical_client:
            try:
                key, secret, _ = AlpacaConfig.get_credentials()
                cls._historical_client = StockHistoricalDataClient(key, secret)
                logger.info("Historical data client created")
            except (ValueError, Exception) as e:
                logger.error(f"Error creating historical data client: {e}")
                raise
        
        return cls._historical_client
    
    @classmethod
    def reset(cls):
        """Reset all clients (useful for testing)"""
        cls._trading_client = None
        cls._historical_client = None