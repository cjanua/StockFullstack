# debug_training.py
import asyncio

import torch

from agent.pytorch_system import train_lstm_model
from stock_fullstack.common.sdk.clients import AlpacaDataConnector
from features.feature_engine import AdvancedFeatureEngine
from config.settings import TradingConfig

config = TradingConfig()



async def debug_single_symbol():
    """Debug training for a single symbol."""
    print("🔍 DEBUGGING TRAINING PROCESS")

    # Get data for just one symbol
    data_connector = AlpacaDataConnector(config)
    market_data = await data_connector.get_historical_data(
        symbols=['AAPL'],
        lookback_days=252  # 1 year
    )

    # Feature engineering
    feature_engine = AdvancedFeatureEngine()
    spy_data = feature_engine.get_market_context_data(
        market_data['AAPL'].index)

    features = feature_engine.create_comprehensive_features(
        market_data['AAPL'], 'AAPL', spy_data)

    print(f"Original data shape: {market_data['AAPL'].shape}")
    print(f"Features shape: {features.shape}")
    print(f"Features NaN count: {features.isnull().sum().sum()}")

    # Test sequence creation
    targets = features['close'].copy()
    feature_only = features.drop(columns=['close'])

    from clean_data.pytorch_data import create_sequences
    X, y = create_sequences(feature_only, targets, 60)

    print(f"Sequences created: {len(X)}")
    print(f"Input shape: {X.shape if len(X) > 0 else 'None'}")
    print(f"Target shape: {y.shape if len(y) > 0 else 'None'}")

    if len(X) > 0:
        # Test training
        print("\n🤖 Testing model training...")
        model = train_lstm_model(features, 'AAPL', config, num_epochs=10)

        if model:
            print("✅ Training successful!")

            # Test prediction with proper device handling
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            test_input = torch.FloatTensor(X[:1]).to(device)
            with torch.no_grad():
                prediction = model(test_input)
                print(f"Test prediction: {prediction}")
        else:
            print("❌ Training failed!")
    else:
        print("❌ No sequences created!")

if __name__ == "__main__":
    asyncio.run(debug_single_symbol())
