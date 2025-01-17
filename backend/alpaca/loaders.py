import logging
from typing import List
from dotenv import load_dotenv
import os
from alpaca.trading.client import TradingClient
from alpaca.common.exceptions import APIError
from alpaca.trading.models import Position, Asset

from result import Ok, Result, Err
import requests
from backend.alpaca.serializers.Account import serialize_account
from backend.alpaca.serializers.Asset import serialize_asset
from backend.alpaca.serializers.Position import serialize_position
from util import logger
from config import APCA

ALPACA_KEY, ALPACA_SECRET, _ = APCA

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
    
def get_account() -> Result[dict, str]:
    """Get account information"""

    client_res = get_trading_client()
    if not client_res.is_ok():
        return Err(client_res.err_value)

    client = client_res.ok_value

    try:
        account = client.get_account()
        logger.info("Successfully retrieved account")
        return Ok(serialize_account(account))
    except Exception as e:
        logger.error(f"Error in get_account: {type(e).__name__}: {str(e)}")
        return Err(str(e))

def get_positions() -> Result[List[Position], str]:
    """Get account information"""
    # from serializers.Account import serialize_account

    client_res = get_trading_client()
    if not client_res.is_ok():
        return Err(client_res.err_value)

    client = client_res.ok_value

    try:
        positions = client.get_all_positions()
        logger.info("Successfully retrieved positions")
        return Ok([serialize_position(p) for p in positions])
    except Exception as e:
        logger.error(f"Error in get_account: {type(e).__name__}: {str(e)}")
        return Err(str(e))
    
def get_assets() -> Result[List[Asset], str]:
    """Get account information"""
    # from serializers.Account import serialize_account

    client_res = get_trading_client()
    if not client_res.is_ok():
        return Err(client_res.err_value)

    client = client_res.ok_value

    try:
        assets = client.get_all_assets()
        logger.info("Successfully retrieved assets")
        print(assets)
        return Ok([serialize_asset(a) for a in assets])
    except Exception as e:
        logger.error(f"Error in get_account: {type(e).__name__}: {str(e)}")
        return Err(str(e))