#!../../venv/bin/python3
import argparse
from logging import DEBUG
from typing import Dict, Any
from enum import Enum
from result import is_ok
from commands import CommandContext, CommandRegistry
from util import format_output, logger

from loaders import get_account, get_positions, get_assets


class Interval(str, Enum):
    MINUTE = '1m'
    FIVE_MINUTES = '5m'
    HOUR = '1h'
    DAY = '1d'


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

@registry.register('trading', 'positions')
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
# @registry.register('trading', 'account', 'history')
# def get_account_history(args: argparse.Namespace, ctx: CommandContext) -> Dict[str, Any]:
#     """Get account history"""
#     days = args.days or 7
#     return {
#         'account_id': 'demo-123',
#         'transactions': [
#             {
#                 'date': (datetime.now() - timedelta(days=i)).isoformat(),
#                 'type': 'DEPOSIT',
#                 'amount': 1000.00
#             } for i in range(days)
#         ]
#     }

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

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description='API-style command line interface',
        usage='%(prog)s <domain>/<resource>[/<action>] [options]'
    )
    
    # Global options
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--format', choices=['json', 'csv', 'table'], default='json',
                      help='Output format')
    parser.add_argument('--pretty', action='store_true', help='Pretty print output')
    
    # Command pattern argument
    parser.add_argument('command', help='Command pattern (e.g., trading/account)')
    
    # Command-specific arguments
    # Historical Data
    parser.add_argument('--symbol', help='Stock symbol for bars data')
    parser.add_argument('--interval', type=Interval, choices=list(Interval),
                      help='Bar interval')
    parser.add_argument('--limit', type=int, help='Number of results to return')
    parser.add_argument('--days', type=int, help='Number of days of history')

    # Auth
    parser.add_argument('--token', type=int, help='Authentication token')
    
    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up context
    ctx = CommandContext(
        debug=args.debug,
        format=args.format,
        pretty=args.pretty
    )
    
    if ctx.debug:
        logger.setLevel(DEBUG)
        logger.debug(f"Arguments: {args}")
    
    try:
       # Parse command pattern
        parts = args.command.split('/')
        if len(parts) == 2:
            domain, resource = parts
            action = 'get'  # default action
        elif len(parts) == 3:
            domain, resource, action = parts
        else:
            raise ValueError("Invalid command pattern")
        
        # Get command handler
        handler = registry.get_handler(domain, resource, action)
        if not handler:
            logger.error(f"Unknown command: {args.command}")
            return 1
        
        # Execute handler
        result = handler(args, ctx)
        
        # Format and print output
        print(format_output(result, ctx))
        return 0
        
    except ValueError:
        logger.error("Invalid command pattern. Use: domain/resource/[action]")
        return 1
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        if ctx.debug:
            logger.exception("Detailed error information:")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())