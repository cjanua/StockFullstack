"""
Hybrid Data Manager
Combines Yahoo Finance (training) and Alpaca (testing/live) data sources
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from ai.data_sources.yahoo_loader import YahooFinanceLoader
from stock_fullstack.common.sdk.clients import AlpacaDataConnector

logger = logging.getLogger(__name__)

class HybridDataManager:
    """
    Manages data from multiple sources:
    - Yahoo Finance: High quality historical data for training (2+ years)  
    - Alpaca: Recent data for testing and live trading (last 3-6 months)
    """
    
    def __init__(self, config):
        self.config = config
        self.yahoo_loader = YahooFinanceLoader()
        self.training_lookback = config.LOOKBACK_DAYS  # Long term for training
        self.testing_lookback = min(180, config.LOOKBACK_DAYS // 4)  # 6 months for testing
        
        # Split date - use Alpaca for data after this date
        self.split_date = datetime.now() - timedelta(days=self.testing_lookback)
        
    async def get_training_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Get training data from Yahoo Finance
        Long-term historical data with high quality
        """
        logger.info(f"Fetching training data for {len(symbols)} symbols from Yahoo Finance...")
        logger.info(f"Training period: {self.training_lookback} days")
        
        training_data = self.yahoo_loader.get_historical_data(
            symbols, 
            lookback_days=self.training_lookback
        )
        
        # Filter to only include data before split_date for clean separation
        filtered_data = {}
        for symbol, df in training_data.items():
            # Keep only data before split date
            mask = df.index < self.split_date
            filtered_df = df[mask].copy()
            
            if len(filtered_df) > 50:  # Ensure minimum data points
                filtered_data[symbol] = filtered_df
                logger.info(f"Training data for {symbol}: {len(filtered_df)} rows ending {filtered_df.index.max()}")
            else:
                logger.warning(f"Insufficient training data for {symbol}: {len(filtered_df)} rows")
        
        return filtered_data
    
    async def get_testing_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Get testing data from Alpaca
        Recent data with real-time characteristics for testing
        """
        logger.info(f"Fetching testing data for {len(symbols)} symbols from Alpaca...")
        logger.info(f"Testing period: {self.testing_lookback} days from {self.split_date}")
        
        client = AlpacaDataConnector(self.config)
        testing_data = await client.get_historical_data(symbols, self.testing_lookback)
        
        # Filter to only include data after split_date
        filtered_data = {}
        for symbol, df in testing_data.items():
            # Keep only data after split date
            mask = df.index >= self.split_date
            filtered_df = df[mask].copy()
            
            if len(filtered_df) > 10:  # Minimum data points for testing
                filtered_data[symbol] = filtered_df
                logger.info(f"Testing data for {symbol}: {len(filtered_df)} rows starting {filtered_df.index.min()}")
            else:
                logger.warning(f"Insufficient testing data for {symbol}: {len(filtered_df)} rows")
        
        return filtered_data
    
    async def get_combined_data(self, symbols: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Get both training and testing data for comprehensive analysis
        
        Returns:
            Dict with structure: {symbol: {'training': df, 'testing': df}}
        """
        logger.info("Fetching hybrid dataset (Yahoo for training, Alpaca for testing)...")
        
        # Get both datasets
        training_data = await self.get_training_data(symbols)
        testing_data = await self.get_testing_data(symbols)
        
        # Combine into unified structure
        combined = {}
        all_symbols = set(training_data.keys()) | set(testing_data.keys())
        
        for symbol in all_symbols:
            combined[symbol] = {
                'training': training_data.get(symbol),
                'testing': testing_data.get(symbol),
                'has_training': symbol in training_data,
                'has_testing': symbol in testing_data
            }
            
            # Log data availability
            train_info = f"{len(training_data[symbol])} rows" if symbol in training_data else "No data"
            test_info = f"{len(testing_data[symbol])} rows" if symbol in testing_data else "No data"
            logger.info(f"{symbol}: Training({train_info}), Testing({test_info})")
        
        return combined
    
    def validate_data_quality(self, symbol_data: Dict[str, pd.DataFrame]) -> Dict[str, dict]:
        """
        Validate data quality across different sources
        Check for gaps, consistency, and potential issues
        """
        validation_results = {}
        
        for symbol, datasets in symbol_data.items():
            training_df = datasets.get('training')
            testing_df = datasets.get('testing')
            
            result = {
                'symbol': symbol,
                'training_quality': {},
                'testing_quality': {},
                'consistency': {}
            }
            
            # Validate training data
            if training_df is not None:
                result['training_quality'] = {
                    'rows': len(training_df),
                    'date_range': f"{training_df.index.min()} to {training_df.index.max()}",
                    'missing_values': training_df.isnull().sum().sum(),
                    'zero_volume_days': (training_df['volume'] == 0).sum(),
                    'price_gaps': self._detect_price_gaps(training_df),
                    'data_source': 'Yahoo Finance'
                }
            
            # Validate testing data
            if testing_df is not None:
                result['testing_quality'] = {
                    'rows': len(testing_df),
                    'date_range': f"{testing_df.index.min()} to {testing_df.index.max()}",
                    'missing_values': testing_df.isnull().sum().sum(),
                    'zero_volume_days': (testing_df['volume'] == 0).sum(),
                    'price_gaps': self._detect_price_gaps(testing_df),
                    'data_source': 'Alpaca'
                }
            
            # Check consistency between sources
            if training_df is not None and testing_df is not None:
                # Check for price continuity at the boundary
                train_last_price = training_df['close'].iloc[-1]
                test_first_price = testing_df['close'].iloc[0]
                price_gap_pct = abs(test_first_price - train_last_price) / train_last_price * 100
                
                result['consistency'] = {
                    'boundary_price_gap_pct': price_gap_pct,
                    'volume_scale_ratio': testing_df['volume'].median() / training_df['volume'].median(),
                    'data_gap_days': (testing_df.index.min() - training_df.index.max()).days
                }
            
            validation_results[symbol] = result
        
        return validation_results
    
    def _detect_price_gaps(self, df: pd.DataFrame, gap_threshold: float = 0.1) -> int:
        """Detect significant price gaps (>10% by default)"""
        if len(df) < 2:
            return 0
        
        price_changes = df['close'].pct_change().abs()
        gaps = (price_changes > gap_threshold).sum()
        return gaps
    
    async def get_market_context_data(self) -> pd.DataFrame:
        """
        Get market context data (SPY) using hybrid approach
        Combines Yahoo (historical) + Alpaca (recent)
        """
        logger.info("Fetching market context data (SPY)...")
        
        # Get training data from Yahoo
        yahoo_spy = self.yahoo_loader.get_historical_data(['SPY'], self.training_lookback)
        training_spy = yahoo_spy.get('SPY')
        
        # Get recent data from Alpaca  
        client = AlpacaDataConnector(self.config)
        alpaca_spy = await client.get_historical_data(['SPY'], self.testing_lookback)
        testing_spy = alpaca_spy.get('SPY')
        
        # Combine datasets
        if training_spy is not None and testing_spy is not None:
            # Filter to avoid overlap
            training_filtered = training_spy[training_spy.index < self.split_date]
            testing_filtered = testing_spy[testing_spy.index >= self.split_date]
            
            # Concatenate
            combined_spy = pd.concat([training_filtered, testing_filtered])
            combined_spy = combined_spy.sort_index()
            
            logger.info(f"Combined SPY data: {len(combined_spy)} rows, {combined_spy.index.min()} to {combined_spy.index.max()}")
            return combined_spy
            
        elif training_spy is not None:
            logger.info(f"Using Yahoo SPY data: {len(training_spy)} rows")
            return training_spy
            
        elif testing_spy is not None:
            logger.info(f"Using Alpaca SPY data: {len(testing_spy)} rows")
            return testing_spy
            
        else:
            raise ValueError("Could not fetch SPY market context data from any source")

# Example usage
async def main():
    """Example usage of HybridDataManager"""
    from ai.config.settings import TradingConfig
    
    config = TradingConfig()
    manager = HybridDataManager(config)
    
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Get combined dataset
    combined_data = await manager.get_combined_data(test_symbols)
    
    # Validate data quality
    validation = manager.validate_data_quality(combined_data)
    
    # Print results
    for symbol, data in combined_data.items():
        print(f"\n{symbol}:")
        if data['has_training']:
            train_df = data['training']
            print(f"  Training: {len(train_df)} rows ({train_df.index.min()} to {train_df.index.max()})")
        if data['has_testing']:  
            test_df = data['testing']
            print(f"  Testing: {len(test_df)} rows ({test_df.index.min()} to {test_df.index.max()})")
        
        # Print validation results
        val = validation[symbol]
        if 'boundary_price_gap_pct' in val['consistency']:
            print(f"  Boundary gap: {val['consistency']['boundary_price_gap_pct']:.2f}%")

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
