import logging
from typing import List, Any, Callable
from dotenv import load_dotenv
import os
from alpaca.trading.client import TradingClient
from alpaca.common.exceptions import APIError
from alpaca.trading.models import Position, Asset, PortfolioHistory, TradeAccount, Watchlist
from alpaca.trading.requests import GetPortfolioHistoryRequest

from result import Ok, Result, Err
import requests
from backend.alpaca.serializers import serialize_account, serialize_asset, serialize_position, serialize_portfolio_history
from backend.alpaca.serializers.Watchlist import serialize_watchlist
from util import logger
from config import APCA
import redis
import json

ALPACA_KEY, ALPACA_SECRET, _ = APCA

# Initialize Redis client
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

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
    return cache_result('account', 3600, _fetch_account)

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
    return cache_result('positions', 3600, _fetch_positions)

def _fetch_positions() -> List[Position]:
    client_res = get_trading_client()
    if not client_res.is_ok():
        raise Exception(client_res.err_value)
    client = client_res.ok_value

    positions = client.get_positions()
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

    assets = client.get_assets()
    logger.info("Successfully retrieved assets")
    return [serialize_asset(a) for a in assets]