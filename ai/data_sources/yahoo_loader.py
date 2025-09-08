"""
Yahoo Finance Data Loader for Training Data
Provides high-quality historical data for model training while using Alpaca for live testing
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Optional
import time

logger = logging.getLogger(__name__)

class YahooFinanceLoader:
    """Yahoo Finance data loader for training data"""
    
    def __init__(self, cache_dir: str = '.cache/yahoo'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_lifespan_hours = 24  # Cache for 24 hours
        
    def _get_cache_path(self, symbol: str, lookback_days: int) -> Path:
        """Get cache file path for symbol and timeframe"""
        return self.cache_dir / f"{symbol}_{lookback_days}d.csv"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file is valid and not expired"""
        if not cache_path.exists():
            return False
            
        # Check if file is too old
        file_age = time.time() - cache_path.stat().st_mtime
        max_age = self.cache_lifespan_hours * 3600
        
        return file_age < max_age
    
    def get_historical_data(
        self, 
        symbols: List[str], 
        lookback_days: int = 252 * 2,
        period: str = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data from Yahoo Finance
        
        Args:
            symbols: List of stock symbols
            lookback_days: Number of days to look back (default 2 years)
            period: Yahoo Finance period string (overrides lookback_days)
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        
        results = {}
        failed_symbols = []
        
        for symbol in symbols:
            try:
                cache_path = self._get_cache_path(symbol, lookback_days)
                
                # Check cache first
                if self._is_cache_valid(cache_path):
                    logger.info(f"Loading {symbol} from cache: {cache_path}")
                    df = pd.read_csv(cache_path, index_col='Date', parse_dates=True)
                    results[symbol] = df
                    continue
                
                logger.info(f"Fetching {symbol} from Yahoo Finance...")
                
                # Create ticker object
                ticker = yf.Ticker(symbol)
                
                if period:
                    # Use period string (e.g., "2y", "5y", "max")
                    hist = ticker.history(period=period)
                else:
                    # Use specific date range
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=lookback_days)
                    hist = ticker.history(start=start_date, end=end_date)
                
                if hist.empty:
                    logger.warning(f"No data returned for {symbol}")
                    failed_symbols.append(symbol)
                    continue
                
                # Standardize column names to match Alpaca format
                hist = hist.rename(columns={
                    'Open': 'open',
                    'High': 'high', 
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # Keep only essential columns
                essential_cols = ['open', 'high', 'low', 'close', 'volume']
                hist = hist[essential_cols]
                
                # Remove timezone info to match Alpaca format
                if hist.index.tz is not None:
                    hist.index = hist.index.tz_localize(None)
                
                # Save to cache
                hist.to_csv(cache_path)
                logger.info(f"Cached {len(hist)} rows for {symbol}")
                
                results[symbol] = hist
                
                # Small delay to be respectful to Yahoo Finance
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                failed_symbols.append(symbol)
        
        logger.info(f"Successfully fetched: {len(results)} symbols")
        if failed_symbols:
            logger.warning(f"Failed to fetch: {failed_symbols}")
        
        return results
    
    def get_market_data(self, lookback_days: int = 252 * 2) -> pd.DataFrame:
        """Get S&P 500 market data for market context features"""
        spy_data = self.get_historical_data(['SPY'], lookback_days)
        if 'SPY' in spy_data:
            return spy_data['SPY']
        else:
            raise ValueError("Could not fetch SPY market data")
    
    def compare_with_alpaca(self, symbol: str, lookback_days: int = 30):
        """
        Compare Yahoo Finance data with Alpaca for validation
        Useful for debugging data quality differences
        """
        try:
            # Get Yahoo data
            yahoo_data = self.get_historical_data([symbol], lookback_days)
            if symbol not in yahoo_data:
                return None, "Yahoo data not available"
            
            # Get Alpaca data  
            from stock_fullstack.common.sdk.clients import AlpacaDataConnector
            import asyncio
            
            async def get_alpaca_data():
                from ai.config.settings import TradingConfig
                config = TradingConfig()
                client = AlpacaDataConnector(config)
                return await client.get_historical_data([symbol], lookback_days)
            
            alpaca_data = asyncio.run(get_alpaca_data())
            if symbol not in alpaca_data:
                return None, "Alpaca data not available"
            
            yahoo_df = yahoo_data[symbol]
            alpaca_df = alpaca_data[symbol]
            
            # Find common dates
            common_dates = yahoo_df.index.intersection(alpaca_df.index)
            if len(common_dates) == 0:
                return None, "No common dates found"
            
            # Compare on common dates
            yahoo_common = yahoo_df.loc[common_dates]
            alpaca_common = alpaca_df.loc[common_dates]
            
            comparison = {
                'symbol': symbol,
                'yahoo_rows': len(yahoo_df),
                'alpaca_rows': len(alpaca_df),
                'common_dates': len(common_dates),
                'yahoo_date_range': f"{yahoo_df.index.min()} to {yahoo_df.index.max()}",
                'alpaca_date_range': f"{alpaca_df.index.min()} to {alpaca_df.index.max()}",
                'close_price_correlation': yahoo_common['close'].corr(alpaca_common['close']),
                'volume_correlation': yahoo_common['volume'].corr(alpaca_common['volume']),
                'avg_close_diff_pct': ((yahoo_common['close'] - alpaca_common['close']) / alpaca_common['close'] * 100).abs().mean()
            }
            
            return comparison, None
            
        except Exception as e:
            return None, str(e)

# Example usage and validation
if __name__ == '__main__':
    # Test the Yahoo Finance loader
    loader = YahooFinanceLoader()
    
    # Test basic functionality
    print("Testing Yahoo Finance data loader...")
    test_symbols = ['AAPL', 'SPY', 'QQQ']
    data = loader.get_historical_data(test_symbols, lookback_days=30)
    
    for symbol, df in data.items():
        print(f"{symbol}: {len(df)} rows, {df.index.min()} to {df.index.max()}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Latest close: ${df['close'].iloc[-1]:.2f}")
    
    # Compare with Alpaca (if available)
    if 'AAPL' in data:
        print("\nComparing AAPL data with Alpaca...")
        comparison, error = loader.compare_with_alpaca('AAPL', 30)
        if comparison:
            print(f"  Yahoo rows: {comparison['yahoo_rows']}")
            print(f"  Alpaca rows: {comparison['alpaca_rows']}")
            print(f"  Common dates: {comparison['common_dates']}")
            print(f"  Close correlation: {comparison['close_price_correlation']:.4f}")
            print(f"  Avg price diff: {comparison['avg_close_diff_pct']:.2f}%")
        else:
            print(f"  Comparison failed: {error}")
