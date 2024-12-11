import logging
from dotenv import load_dotenv
import os

from alpaca.trading.client import TradingClient
from result import Ok, Result
load_dotenv()

APCA_KEY = os.getenv('APCA_KEY')
APCA_SECRET = os.getenv('APCA_SECRET')

trading_client = TradingClient(APCA_KEY, APCA_SECRET, paper=False)

def get_account() -> Result[dict, str]:
    """Get account information"""
    from serializers.Account import serialize_account

    account = trading_client.get_account()
    return Ok(serialize_account(account))