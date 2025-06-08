# test_pipeline.py
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import torch

from ai.agent.pytorch_system import train_lstm_model
from ai.config.settings import config
from backend.alpaca.sdk.clients import AlpacaDataConnector
from ai.features.feature_engine import AdvancedFeatureEngine
from ai.clean_data.pytorch_data import create_sequences

async def test_data_pipeline():
    """Test the data pipeline to identify where NaN values are introduced"""
    
    print("=== TESTING DATA PIPELINE ===\n")
    
    # 1. Test data acquisition
    print("1. Testing Alpaca data acquisition...")
    data_connector = AlpacaDataConnector(config)
    
    # Test with just one symbol
    test_symbol = 'AAPL'
    market_data = await data_connector.get_historical_data(
        symbols=[test_symbol],
        lookback_days=252  # Just 30 days for testing
    )
    
    if test_symbol not in market_data:
        print(f"ERROR: Could not fetch data for {test_symbol}")
        return
    
    raw_data = market_data[test_symbol]
    
    # 2. Test feature engineering
    print("2. Testing feature engineering...")
    feature_engine = AdvancedFeatureEngine()
    
    # Get SPY data for market context
    spy_data = feature_engine.get_market_context_data(raw_data.index)
    
    features = feature_engine.create_comprehensive_features(raw_data, spy_data)
    if features.empty:
        print("ERROR: Feature creation failed. DataFrame is empty.")
        return
    
    print(f"Features created with shape: {features.shape}")
    print(f"Features NaN count:\n{features.isnull().sum().sort_values(ascending=False).head()}")
    print(f"Features sample:\n{features.head()}\n")
    
    # 3. Test sequence creation
    print("3. Testing sequence creation...")
    if 'close' not in features.columns:
        print("ERROR: No 'close' column in features!")
        return
    
    trained_model = train_lstm_model(
        processed_data=features,
        symbol=test_symbol,
        config=config,
        num_epochs=5,  # Use fewer epochs for a quick test
    )

    if trained_model is None:
        print("ERROR: Model training failed in test pipeline.")
        return

    print("\n4. Testing sequence creation and forward pass...")
    
    targets = features['close'].copy()
    features = features.drop(columns=['close'], errors='ignore')
    
    X, y = create_sequences(features, targets, sequence_length=60)
    
    print(f"Sequences shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Any NaN in sequences: {np.isnan(X).any() if len(X) > 0 else 'No sequences'}")
    
    # 4. Test a simple forward pass
    if len(X) > 0:        
        test_batch = torch.FloatTensor(X[:1])  # Just one sample
        
        try:
            with torch.no_grad():
                output = trained_model(test_batch)
            print(f"Model output shape: {output.shape}")
            print(f"Model output: {output}")
            print(f"Model output has NaN: {torch.isnan(output).any()}")
        except Exception as e:
            print(f"ERROR in forward pass: {e}")
    
    print("\n=== PIPELINE TEST COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(test_data_pipeline())