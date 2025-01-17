import os
from dotenv import load_dotenv




load_dotenv()

ALPACA_KEY = os.environ.get('ALPACA_KEY')
ALPACA_SECRET = os.environ.get('ALPACA_SECRET')
ALPACA_URL = os.environ.get('ALPACA_URL')

APCA = (ALPACA_KEY, ALPACA_SECRET, ALPACA_URL)