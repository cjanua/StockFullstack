import logging
from typing import List
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

def get_trading_client() -> Result[TradingClient, str]:
    """Get authenticated trading client with enhanced error handling"""
    if not ALPACA_KEY or not ALPACA_SECRET:
        return Err("Alpaca API credentials not found in environment")
    
    try:
        # Create client with additional options
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
            return Err(f"API Error: {str(e)}")
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return Err(f"Connection test failed: {str(e)}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return Err(f"Request failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return Err(str(e))
    
def get_account() -> Result[TradeAccount, str]:
    """Get account information"""
    cached_account = redis_client.get('account')
    if cached_account:
        logger.info("Returning cached account information")
        return Ok(json.loads(cached_account))

    client_res = get_trading_client()
    if not client_res.is_ok():
        return Err(client_res.err_value)

    client = client_res.ok_value

    try:
        account = client.get_account()
        logger.info("Successfully retrieved account")
        serialized_account = serialize_account(account)
        redis_client.setex('account', 3600, json.dumps(serialized_account))  # Cache for 1 hour
        return Ok(serialized_account)
    except Exception as e:
        logger.error(f"Error in get_account: {type(e).__name__}: {str(e)}")
        return Err(str(e))

def get_positions() -> Result[List[Position], str]:
    """Get account positions"""
    cached_positions = redis_client.get('positions')
    if cached_positions:
        logger.info("Returning cached positions")
        return Ok(json.loads(cached_positions))

    client_res = get_trading_client()
    if not client_res.is_ok():
        return Err(client_res.err_value)
    client = client_res.ok_value

    try:
        positions = client.get_all_positions()
        logger.info("Successfully retrieved positions")
        serialized_positions = [serialize_position(p) for p in positions]
        redis_client.setex('positions', 3600, json.dumps(serialized_positions))  # Cache for 1 hour
        return Ok(serialized_positions)
    except Exception as e:
        logger.error(f"Error in get_positions: {type(e).__name__}: {str(e)}")
        return Err(str(e))
    
def get_assets() -> Result[List[Asset], str]:
    """Get account assets"""
    cached_assets = redis_client.get('assets')
    if cached_assets:
        logger.info("Returning cached assets")
        return Ok(json.loads(cached_assets))

    client_res = get_trading_client()
    if not client_res.is_ok():
        return Err(client_res.err_value)
    client = client_res.ok_value

    try:
        assets = client.get_all_assets()
        logger.info("Successfully retrieved assets")
        serialized_assets = [serialize_asset(a) for a in assets]
        redis_client.setex('assets', 3600, json.dumps(serialized_assets))  # Cache for 1 hour
        return Ok(serialized_assets)
    except Exception as e:
        logger.error(f"Error in get_account: {type(e).__name__}: {str(e)}")
        return Err(str(e))

def get_portfolio_history(days: int = 7, timeframe: str = "1D") -> Result[PortfolioHistory, str]:
    """Get account history"""
    cache_key = f"portfolio_history_{days}_{timeframe}"
    cached_history = redis_client.get(cache_key)
    if cached_history:
        logger.info("Returning cached portfolio history")
        return Ok(json.loads(cached_history))

    client_res = get_trading_client()
    if not client_res.is_ok():
        return Err(client_res.err_value)
    client = client_res.ok_value

    try:
        req: GetPortfolioHistoryRequest = GetPortfolioHistoryRequest(
            period=f'{days}D',
            timeframe=f'{timeframe}',
        )
        history = client.get_portfolio_history(req)
        logger.info("Successfully retrieved history")
        serialized_history = serialize_portfolio_history(history)
        redis_client.setex(cache_key, 3600, json.dumps(serialized_history))  # Cache for 1 hour
        return Ok(serialized_history)
    except Exception as e:
        logger.error(f"Error in get_account: {type(e).__name__}: {str(e)}")
        return Err(str(e))

def get_watchlists() -> Result[List[Watchlist], str]:
    """Get watchlists"""
    cached_watchlists = redis_client.get('watchlists')
    if cached_watchlists:
        logger.info("Returning cached watchlists")
        return Ok(json.loads(cached_watchlists))

    client_res = get_trading_client()
    if not client_res.is_ok():
        return Err(client_res.err_value)
    client = client_res.ok_value

    try:
        watchlists: List[Watchlist] = client.get_watchlists()
        logger.info("Successfully retrieved watchlists")
        serialized_watchlists = [serialize_watchlist(w) for w in watchlists]
        redis_client.setex('watchlists', 3600, json.dumps(serialized_watchlists))  # Cache for 1 hour
        return Ok(serialized_watchlists)
    except Exception as e:
        logger.error(f"Error in get_watchlists: {type(e).__name__}: {str(e)}")
        return Err(str(e))