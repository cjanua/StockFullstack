import logging
from dotenv import load_dotenv
import os
from alpaca.trading.client import TradingClient
from alpaca.common.exceptions import APIError
from result import Ok, Result, Err
import requests


logger = logging.getLogger(__name__)
load_dotenv()

ALPACA_KEY = os.environ.get('ALPACA_KEY')
ALPACA_SECRET = os.environ.get('ALPACA_SECRET')
ALPACA_URL = os.environ.get('ALPACA_URL')

def get_trading_client() -> Result[TradingClient, str]:
    """Get authenticated trading client with enhanced error handling"""
    if not ALPACA_KEY or not ALPACA_SECRET:
        return Err("Alpaca API credentials not found in environment")
    
    try:
        # First try direct request to verify credentials
        session = requests.Session()
        headers = {
            'APCA-API-KEY-ID': ALPACA_KEY,
            'APCA-API-SECRET-KEY': ALPACA_SECRET,
            'Accept': 'application/json'
        }
        
        logger.info("Testing direct API access...")
        response = session.get('https://api.alpaca.markets/v2/account', headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Direct API test failed: {response.status_code}")
            logger.error(f"Response body: {response.text}")
            return Err(f"API test failed: {response.status_code} - {response.text}")
        

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
    from serializers.Account import serialize_account

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