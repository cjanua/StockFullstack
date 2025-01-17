from backend.alpaca.config import APCA
from util import logger
import requests
from alpaca.trading.client import TradingClient

ALPACA_KEY, ALPACA_SECRET, ALPACA_URL_ = APCA

def test_trading_api(client: TradingClient):
    """Test trading api"""
    # First try direct request to verify credentials
    session = requests.Session()
    headers = {
        'APCA-API-KEY-ID': ALPACA_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET,
        'Accept': 'application/json'
    }
    response = session.get('https://api.alpaca.markets/v2/account', headers=headers)
    
    if response.status_code != 200:
        logger.error(f"Direct API test failed: {response.status_code}:\n {response.text}")
        return Err(f"API test failed: {response.status_code} - {response.text}")