# backend/alpaca/sdk/clients.py
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.common.exceptions import APIError
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

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


class AlpacaDataConnector:
    """
    A high-level connector to fetch financial data using the Alpaca API.
    This class is what the AI part of your application will interact with.
    """
    def __init__(self, config):
        """
        Initializes the data connector with the historical data client.
        
        Args:
            config: A configuration object (like your TradingConfig dataclass).
        """
        self.client = AlpacaClientManager.get_historical_client()
        self.config = config

    async def get_historical_data(self, symbols: list, lookback_days: int) -> Dict[str, pd.DataFrame]:
        """
        Fetches and processes historical daily bar data for a list of symbols.
        
        Args:
            symbols: A list of stock tickers.
            lookback_days: The number of days of historical data to fetch.
            
        Returns:
            A dictionary where keys are the symbols and values are DataFrames
            of their historical data.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        all_data = {}

        async def fetch_symbol(symbol: str):
            """An inner async function to fetch data for a single symbol."""
            try:
                request_params = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    start=start_date,
                    end=end_date,
                    adjustment='raw' # Use raw data
                )
                
                # Use asyncio.to_thread to run the synchronous SDK call in a non-blocking way
                bars = await asyncio.to_thread(self.client.get_stock_bars, request_params)
                
                if bars:
                    df = bars.df

                    if isinstance(df.index, pd.MultiIndex):
                        df = df.reset_index(level='symbol', drop=True)
                    df.index = df.index.tz_convert('UTC').tz_localize(None)

                    return symbol, df
                return symbol, pd.DataFrame()

            except Exception as e:
                logger.error(f"Failed to fetch historical data for {symbol}: {e}")
                return symbol, pd.DataFrame()

        # Create and run all fetching tasks in parallel
        tasks = [fetch_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)

        # Populate the final dictionary
        for symbol, data in results:
            if not data.empty:
                all_data[symbol] = data
        
        return all_data

