from typing import Dict, Any
from result import is_ok
from loaders import get_account, get_positions, get_assets, get_portfolio_history, get_watchlists
from commands import CommandRegistry, CommandContext
import argparse
from datetime import datetime, timedelta

registry = CommandRegistry()

# Command handlers
@registry.register('trading', 'account')
def get_trading_account(*kwrgs) -> Dict[str, Any]:
    """Get account information"""
    res = get_account()
    if is_ok(res):
        return res.ok_value
    return {
        "error": True,
        "message": res.err_value if hasattr(res, 'err_value') else "Unknown error"
    }

@registry.register('trading', 'account', 'positions')
def get_trading_positions(*kwrgs) -> Dict[str, Any]:
    """Get positions information"""
    res = get_positions()
    if is_ok(res):
        return res.ok_value
    return {
        "error": True,
        "message": res.err_value if hasattr(res, 'err_value') else "Unknown error"
    }

@registry.register('trading', 'assets')
def get_trading_assets(*kwrgs) -> Dict[str, Any]:
    """Get list of assets"""
    res = get_assets()
    if is_ok(res):
        return res.ok_value
    return {
        "error": True,
        "message": res.err_value if hasattr(res, 'err_value') else "Unknown error"
    }

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
    
    res = get_portfolio_history(days, timeframe)
    if is_ok(res):
        return res.ok_value
    return {
        "error": True,
        "message": res.err_value if hasattr(res, 'err_value') else "Unknown error"
    }

@registry.register('trading', 'account', 'watchlists')
def get_account_watchlists(args: argparse.Namespace, ctx: CommandContext) -> Dict[str, Any]:
    """Get account history"""
    
    res = get_watchlists()
    if is_ok(res):
        return res.ok_value
    return {
        "error": True,
        "message": res.err_value if hasattr(res, 'err_value') else "Unknown error"
    }

# @registry.register('data', 'stocks', 'bars')
# def get_stock_bars(args: argparse.Namespace, ctx: CommandContext) -> Dict[str, Any]:
#     """Get stock bar data"""
#     return {
#         'symbol': args.symbol,
#         'interval': args.interval,
#         'bars': [
#             {
#                 'timestamp': (datetime.now() - timedelta(minutes=i)).isoformat(),
#                 'open': 100 + i,
#                 'high': 101 + i,
#                 'low': 99 + i,
#                 'close': 100.5 + i,
#                 'volume': 1000 * i
#             } for i in range(args.limit or 10)
#         ]
#     }