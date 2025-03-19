#!/usr/bin/env /home/wsluser/Stocks/venv/bin/python3
# alpaca/apca.py
import argparse
from logging import DEBUG
from enum import Enum
from backend.alpaca.cli.commands import CommandContext
from backend.alpaca.core.util import format_output, logger
from backend.alpaca.handlers.handlers import registry  # Import the registry from handlers



class Interval(str, Enum):
    MINUTE = '1m'
    FIVE_MINUTES = '5m'
    HOUR = '1h'
    DAY = '1d'

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description='API-style command line interface',
        usage='%(prog)s <domain>/<resource>[/<action>] [options]'
    )
    
    # Global options
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--format', choices=['json', 'csv', 'table'], default='json', help='Output format')
    parser.add_argument('--pretty', action='store_true', help='Pretty print output')
    
    # Command pattern argument
    parser.add_argument('command', help='Command pattern (e.g., trading/account)')
    
    # Command-specific arguments
    # Historical Data
    parser.add_argument('--symbol', help='Stock symbol for bars data')
    parser.add_argument('--interval', type=Interval, choices=list(Interval), help='Bar interval')
    parser.add_argument('--limit', type=int, help='Number of results to return')

    parser.add_argument('--days', type=int, help='Number of days into history')
    parser.add_argument('--timeframe', type=str, help='Resolution of historical data')

    # Auth
    parser.add_argument('--token', type=int, help='Authentication token')
    parser.add_argument('--query', type=str, help='A Search query')
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