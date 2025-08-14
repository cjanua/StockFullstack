# backend/alpaca/sdk/clients.py
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.common.exceptions import APIError
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from ai.utils import print_integrity_check
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
        self.cache_dir = Path(self.config.CACHE_DIR) / 'alpaca'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.request_semaphore = asyncio.Semaphore(3)  # Max 3 concurrent requests
        self.request_delay = 0.5  # 500ms delay between requests
        self.retry_delay = 2.0    # 2 second delay for retries
        self.max_retries = 3      # Maximum retry attempts


    def _get_cache_path(self, *args) -> Path:
        """Creates a unique filename for a given cache request."""

        parts = []
        for arg in args:
            parts.append(f"{arg}")

        filename = "_".join(sorted(parts)) + ".csv"

        return self.cache_dir / filename


    def _is_cache_valid(self, path: Path) -> bool:
        """Checks if a cache file exists and is not too old."""
        if not path.exists():
            return False

        file_mod_time = datetime.fromtimestamp(path.stat().st_mtime)
        if (datetime.now() - file_mod_time).total_seconds() > self.config.CACHE_LIFESPAN_HOURS * 3600:
            return False # Cache is stale

        return True

    async def get_historical_data(self, symbols: list, lookback_days: int, market: str = "IEX") -> Dict[str, pd.DataFrame]:
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
        failed_symbols = []


        async def fetch_symbol(symbol: str, retry_count: int = 0) -> None:
            """An inner async function to fetch data for a single symbol."""
            async with self.request_semaphore:
                cache_path = self._get_cache_path(market, symbol, lookback_days)

                # Check cache first
                if self._is_cache_valid(cache_path):
                    print(f"CACHE HIT: '{symbol}' from {cache_path}")
                    try:
                        df = pd.read_csv(cache_path, index_col='timestamp', parse_dates=True)

                        # Force essential columns to be numeric after loading
                        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                        for col in numeric_cols:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')

                        df.dropna(subset=numeric_cols, inplace=True)
                        print_integrity_check(df, f"Cached Alpaca Data for {symbol}")

                        all_data[symbol] = df
                        return
                    except Exception as e:
                        logger.warning(f"Cache read failed for {symbol}: {e}")
                        # Continue to API fetch if cache fails

                # Add delay before API request
                if retry_count > 0:
                    delay = self.retry_delay * (2 ** retry_count)  # Exponential backoff
                    print(f"Retrying {symbol} in {delay}s (attempt {retry_count + 1})")
                    await asyncio.sleep(delay)
                else:
                    await asyncio.sleep(self.request_delay)

                try:
                    request_params = StockBarsRequest(
                        symbol_or_symbols=symbol,
                        timeframe=TimeFrame.Hour,
                        start=start_date,
                        end=end_date,
                        adjustment='raw',
                        feed=market.lower(),
                    )

                    print(f"API REQUEST: Fetching {symbol} (attempt {retry_count + 1})")

                    # Use asyncio.to_thread for the synchronous SDK call
                    bars = await asyncio.to_thread(self.client.get_stock_bars, request_params)

                    if bars:
                        df = bars.df

                        if isinstance(df.index, pd.MultiIndex):
                            df = df.reset_index(level='symbol', drop=True)

                        df.index = df.index.tz_convert('UTC').normalize()
                        df.index = df.index.tz_localize(None)

                        # Save to cache
                        df.to_csv(cache_path)
                        print(f"API SUCCESS: Saved '{symbol}' to cache")
                        print_integrity_check(df, f"Fresh Alpaca Data for {symbol}")
                        all_data[symbol] = df
                    else:
                        logger.warning(f"No data returned for {symbol}")
                        failed_symbols.append(symbol)

                except Exception as e:
                    error_msg = str(e).lower()

                    # Handle rate limiting specifically
                    if "too many requests" in error_msg or "rate limit" in error_msg:
                        if retry_count < self.max_retries:
                            logger.warning(f"Rate limited on {symbol}, retrying...")
                            await fetch_symbol(symbol, retry_count + 1)
                            return
                        else:
                            logger.error(f"Max retries exceeded for {symbol} due to rate limiting")
                    else:
                        logger.error(f"Failed to fetch data for {symbol}: {e}")

                    failed_symbols.append(symbol)

        # SEQUENTIAL PROCESSING instead of parallel to avoid rate limits
        print(f"Fetching data for {len(symbols)} symbols sequentially...")
        for i, symbol in enumerate(symbols):
            print(f"Processing {symbol} ({i+1}/{len(symbols)})")
            await fetch_symbol(symbol)

            # Add a small delay between symbols to be extra safe
            if i < len(symbols) - 1:  # Don't delay after the last symbol
                await asyncio.sleep(0.2)

        # Report results
        successful_symbols = list(all_data.keys())
        print(f"\n✅ Successfully fetched: {len(successful_symbols)} symbols")
        print(f"❌ Failed to fetch: {len(failed_symbols)} symbols")

        if failed_symbols:
            print(f"Failed symbols: {failed_symbols}")

        return all_data

