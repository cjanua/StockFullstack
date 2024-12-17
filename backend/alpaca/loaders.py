import logging
from dotenv import load_dotenv
import os

from alpaca.trading.client import TradingClient
from result import Ok, Result
load_dotenv()

ALPACA_KEY = os.getenv('ALPACA_KEY')
ALPACA_SECRET = os.getenv('ALPACA_SECRET')

trading_client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=False)

def get_account() -> Result[dict, str]:
    """Get account information"""
    from serializers.Account import serialize_account

    account = trading_client.get_account()
    return Ok(serialize_account(account))