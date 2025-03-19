# alpaca/core/config.py
import os
from typing import Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AlpacaConfig:
    """Centralized configuration management for Alpaca services"""
    
    # API Credentials
    ALPACA_KEY: str = os.getenv('ALPACA_KEY', '')
    ALPACA_SECRET: str = os.getenv('ALPACA_SECRET', '')
    ALPACA_URL: str = os.getenv('ALPACA_URL', 'https://paper-api.alpaca.markets')
    
    # Redis Configuration
    REDIS_HOST: str = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT: int = int(os.getenv('REDIS_PORT', 6379))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def get_credentials(cls) -> Tuple[str, str, str]:
        """Return Alpaca API credentials"""
        if not (cls.ALPACA_KEY and cls.ALPACA_SECRET):
            raise ValueError("Missing Alpaca API credentials")
        return (cls.ALPACA_KEY, cls.ALPACA_SECRET, cls.ALPACA_URL)

    @classmethod
    def validate(cls):
        """Validate configuration"""
        errors = []
        if not cls.ALPACA_KEY:
            errors.append("ALPACA_KEY is not set")
        if not cls.ALPACA_SECRET:
            errors.append("ALPACA_SECRET is not set")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")

# Expose global configuration
config = AlpacaConfig()