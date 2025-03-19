# alpaca/handlers.py

from typing import Dict, Any, Callable
from backend.alpaca.cli.commands import CommandContext, CommandRegistry
from result import is_ok
from backend.alpaca.sdk.loaders import (
    get_account, query_asset, get_positions, get_assets, 
    get_portfolio_history, get_watchlists
)
import argparse
registry = CommandRegistry()

def handle_response(func: Callable[[], Any]) -> Dict[str, Any]:
    """Helper function to handle API responses"""
    res = func()
    if is_ok(res):
        return res.ok_value
    return {
        "error": True,
        "message": res.err_value if hasattr(res, 'err_value') else "Unknown error"
    }

# Command handlers
@registry.register('trading', 'account')
def get_trading_account(*args) -> Dict[str, Any]:
    """Get account information"""
    return handle_response(get_account)

@registry.register('trading', 'account', 'positions')
def get_trading_positions(*args) -> Dict[str, Any]:
    """Get positions information"""
    return handle_response(get_positions)

@registry.register('trading', 'assets')
def get_trading_assets(*args) -> Dict[str, Any]:
    """Get list of assets"""
    return handle_response(get_assets)

@registry.register('trading', 'account', 'history')
def get_account_history(args: argparse.Namespace, ctx: CommandContext) -> Dict[str, Any]:
    """Get account history"""
    days = args.days or 7
    timeframe = args.timeframe or '1D'
    if timeframe.endswith('Min'):
        if days >= 7 and timeframe == '1Min':
            return {
                "error": True,
                "message": "1Min timeframe is limited to less than 7 days of history",
                "code": "1Min7DayErr"
            }
        if days >= 30 and timeframe.endswith('Min'):
            return {
                "error": True,
                "message": "Any min timeframe is limited to less than 30 days of history",
                "code": "Min30DayErr"
            }
    
    return handle_response(lambda: get_portfolio_history(days, timeframe))

@registry.register('trading', 'account', 'watchlists')
def get_account_watchlists(*args) -> Dict[str, Any]:
    """Get account watchlists"""
    return handle_response(get_watchlists)

@registry.register('trading', 'assets', 'search')
def search_assets(args: argparse.Namespace, ctx: CommandContext) -> Dict[str, Any]:
    """Search assets"""
    query = args.query
    if not query:
        return {
            "error": True,
            "message": "Missing query parameter",
            "code": "MissingQuery"
        }
    return handle_response(lambda: query_asset(query))