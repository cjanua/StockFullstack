# alpaca/loaders.py

from datetime import datetime, timedelta
from typing import List, Any, Callable
import os
from alpaca.trading.client import TradingClient
from alpaca.common.exceptions import APIError
from alpaca.trading.models import Position, Asset, TradeAccount, Watchlist
from alpaca.trading.requests import GetPortfolioHistoryRequest
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from result import Ok, Result, Err
from .serializers import (
    serialize_account, serialize_asset, serialize_position, 
    serialize_portfolio_history, serialize_watchlist
)

from backend.alpaca.core.util import logger
from backend.alpaca.core.config import AlpacaConfig
import pandas as pd

import redis
import json
import traceback
from io import StringIO

ALPACA_KEY, ALPACA_SECRET, _ = AlpacaConfig().get_credentials()

# Initialize Redis client with the host from environment variables
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)

def cache_result(key: str, ttl: int, func: Callable[[], Any]) -> Result[Any, str]:
    """Helper function to cache results"""
    cached_data = redis_client.get(key)
    if cached_data:
        logger.info(f"Returning cached data for key: {key}")
        return Ok(json.loads(cached_data))

    try:
        result = func()
        serialized_result = json.dumps(result)
        redis_client.setex(key, ttl, serialized_result)
        return Ok(result)
    except Exception as e:
        logger.error(f"Error in {func.__name__}: {type(e).__name__}: {str(e)}")
        return Err(str(e))

def get_trading_client() -> Result[TradingClient, str]:
    """Get authenticated trading client with enhanced error handling"""
    if not ALPACA_KEY or not ALPACA_SECRET:
        return Err("Alpaca API credentials not found in environment")
    
    try:
        client = TradingClient(
            ALPACA_KEY,
            ALPACA_SECRET,
            paper=False,
        )
        try:
            _ = client.get_account()
            logger.info("Trading client created and verified")
            return Ok(client)
        except APIError as e:
            logger.error(f"API Error: {str(e)}")
            return Err(str(e))
    except Exception as e:
        logger.error(f"Error in get_trading_client: {type(e).__name__}: {str(e)}")
        return Err(str(e))

def get_portfolio_history(days: int, timeframe: str) -> Result[Any, str]:
    """Get portfolio history"""
    cache_key = f'portfolio_history_{days}_{timeframe}'
    return cache_result(cache_key, 3600, lambda: _fetch_portfolio_history(days, timeframe))

def _fetch_portfolio_history(days: int, timeframe: str) -> Any:
    client_res = get_trading_client()
    if not client_res.is_ok():
        raise Exception(client_res.err_value)
    client = client_res.ok_value

    req = GetPortfolioHistoryRequest(
        period=f'{days}D',
        timeframe=f'{timeframe}',
    )
    history = client.get_portfolio_history(req)
    logger.info("Successfully retrieved history")
    return serialize_portfolio_history(history)

def get_history(days: int = 365):
    """Get historical data for all symbols in portfolio plus benchmarks with caching"""
    cache_key = f'history_{days}'
    
    # Try to get from cache first
    cached_data = redis_client.get(cache_key)
    if cached_data:
        logger.info(f"Returning cached historical data for {days} days")
        # Deserialize the cached DataFrame
        try:
            df = pd.read_json(StringIO(cached_data))
            return Ok(df)
        except Exception as e:
            logger.warning(f"Error deserializing cached data: {e}. Fetching fresh data.")
            # Continue to fetch fresh data if deserialization fails
    
    try:
        # Original code to fetch data
        logger.info(f"Getting fresh historical data for {days} days")
        
        positions_result = get_positions()
        if positions_result.is_err():
            return Err(f"Error fetching positions: {positions_result.err_value}")
        
        positions = positions_result.ok_value
        symbols = [p['symbol'] for p in positions]
        
        symbols.extend(['SPY', 'QQQ', 'IWM', 'GLD'])
        symbols = list(set(symbols))
        
        end_date = datetime.now() - timedelta(minutes=15, seconds=5)
        start_date = end_date - timedelta(days=days)
        
        data_client_res = get_historical_data_client()
        if data_client_res.is_err():
            return Err(data_client_res.err_value)
        
        data_client = data_client_res.ok_value
        
        symbol_data = {}
        
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        
        bars_response = data_client.get_stock_bars(request_params)
        
        for symbol in symbols:
            if symbol in bars_response.data:
                symbol_bars = bars_response.data[symbol]
                
                bars_data = []
                for bar in symbol_bars:
                    bars_data.append({
                        'timestamp': bar.timestamp,
                        'close': bar.close
                    })
                
                if bars_data:
                    df = pd.DataFrame(bars_data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    symbol_data[symbol] = df['close']
                    print(f"Successfully retrieved {len(bars_data)} bars for {symbol}")
            else:
                print(f"No data available for {symbol}")
        
        if not symbol_data:
            return Err("No historical data retrieved for any symbol")
        
        # Combine all symbol data into a single DataFrame
        prices_df = pd.DataFrame(symbol_data)
        
        # Ensure the DataFrame is structured correctly
        if prices_df.empty:
            logger.error("No data available in the DataFrame.")
            return Err("No data available in the DataFrame.")
        
        # Fill missing values
        prices_df = prices_df.ffill().bfill()
        
        # Cache the result for 24 hours (since historical data doesn't change)
        if not prices_df.empty:
            try:
                # Serialize and cache the DataFrame
                redis_client.setex(cache_key, 86400, prices_df.to_json())
            except Exception as e:
                logger.error(f"Failed to cache historical data: {e}")
        
        return Ok(prices_df)
    
    except Exception as e:
        logger.error(f"Error in get_history: {str(e)}\n{traceback.format_exc()}")
        return Err(f"Error in get_history: {str(e)}")

def get_watchlists() -> Result[List[Watchlist], str]:
    """Get watchlists"""
    return cache_result('watchlists', 3600, _fetch_watchlists)

def _fetch_watchlists() -> List[Watchlist]:
    client_res = get_trading_client()
    if not client_res.is_ok():
        raise Exception(client_res.err_value)
    client = client_res.ok_value

    watchlists = client.get_watchlists()
    logger.info("Successfully retrieved watchlists")
    return [serialize_watchlist(w) for w in watchlists]

def get_account() -> Result[TradeAccount, str]:
    """Get account information"""
    # Reduce cache TTL from 3600 to 30 seconds
    return cache_result('account', 30, _fetch_account)

def _fetch_account() -> TradeAccount:
    client_res = get_trading_client()
    if not client_res.is_ok():
        raise Exception(client_res.err_value)
    client = client_res.ok_value

    account = client.get_account()
    logger.info("Successfully retrieved account information")
    return serialize_account(account)

def get_positions() -> Result[List[Position], str]:
    """Get positions information"""
    # Reduce cache TTL from 3600 to 30 seconds
    return cache_result('positions', 30, _fetch_positions)

def _fetch_positions() -> List[Position]:
    client_res = get_trading_client()
    if not client_res.is_ok():
        raise Exception(client_res.err_value)
    client = client_res.ok_value

    positions = client.get_all_positions()
    logger.info("Successfully retrieved positions")
    return [serialize_position(p) for p in positions]

def get_assets() -> Result[List[Asset], str]:
    """Get list of assets"""
    return cache_result('assets', 3600, _fetch_assets)

def _fetch_assets() -> List[Asset]:
    client_res = get_trading_client()
    if not client_res.is_ok():
        raise Exception(client_res.err_value)
    client = client_res.ok_value

    assets = client.get_all_assets()
    logger.info("Successfully retrieved assets")
    return [serialize_asset(a) for a in assets]

def query_asset(symbol: str) -> Result[List[Asset], str]:
    """Query asset by symbol"""
    return cache_result(f'asset_{symbol}', 3600, lambda: _fetch_asset(symbol))

def _fetch_asset(query: str) -> List[Asset]:
    assets_res = get_assets()
    if assets_res.is_err():
        raise Exception(assets_res.err_value)
    
    assets: List[Asset] = assets_res.ok_value

    query_lower = query.lower()
    matching_assets = [
        asset for asset in assets
        if query_lower in asset['symbol'].lower() or query_lower in asset['name'].lower()
    ]
    
    if not matching_assets:
        raise Exception(f"No assets found matching query '{query}'")
    
    return matching_assets

def get_historical_data_client():
    """Get a dedicated client for historical data"""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        
        if not ALPACA_KEY or not ALPACA_SECRET:
            return Err("Alpaca API credentials not found in environment")
        
        data_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
        logger.info("Historical data client created")
        return Ok(data_client)
    except Exception as e:
        logger.error(f"Error creating historical data client: {type(e).__name__}: {str(e)}")
        return Err(str(e))

# Add a function to clear the Redis cache
def clear_portfolio_cache() -> None:
    """Clear portfolio-related cached data from Redis"""
    try:
        # Delete the specific cached data we need fresh
        redis_client.delete('positions')
        redis_client.delete('account')
        redis_client.delete('portfolio_history*')
        
        # Don't clear history cache by default since it's expensive to rebuild
        # Use clear_history_cache() separately when needed
        
        logger.info("Portfolio cache cleared successfully")
    except Exception as e:
        logger.error(f"Failed to clear cache: {str(e)}")

# Add this function to clear history cache
def clear_history_cache(days=None):
    """Clear historical data cache"""
    try:
        if days is None:
            # Clear all historical data caches
            keys = redis_client.keys("history_*")
            if keys:
                redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} historical data cache entries")
        else:
            # Clear specific days cache
            cache_key = f'history_{days}'
            redis_client.delete(cache_key)
            logger.info(f"Cleared historical data cache for {days} days")
            
    except Exception as e:
        logger.error(f"Failed to clear history cache: {e}")