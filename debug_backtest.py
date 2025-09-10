#!/usr/bin/env python3
"""
Debug backtest to see why no trades are generated
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ai.main import UltimateTradingSystem
import torch
import numpy as np

class DebugStrategy:
    """Debug strategy to check model predictions"""
    
    def __init__(self, model, feature_data):
        self.model = model
        self.feature_data = feature_data
        self.model.eval()
    
    def test_predictions(self, n_samples=10):
        """Test model predictions on sample data"""
        print(f"\n🔍 Testing model predictions on {n_samples} samples:")
        print(f"Model input size: {self.model.input_size}")
        print(f"Feature data shape: {self.feature_data.shape}")
        
        # Test different sequence lengths
        for seq_len in [60, 50, 30]:
            try:
                if len(self.feature_data) >= seq_len:
                    features = self.feature_data.iloc[-seq_len:].values
                    print(f"\n  Testing sequence length {seq_len}:")
                    print(f"    Features shape: {features.shape}")
                    
                    with torch.no_grad():
                        # Move model and tensor to same device
                        device = next(self.model.parameters()).device
                        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
                        print(f"    Tensor shape: {features_tensor.shape}")
                        
                        if features_tensor.shape[-1] == self.model.input_size:
                            prediction = self.model(features_tensor)
                            probabilities = prediction.numpy()[0]
                            
                            action = np.argmax(probabilities)
                            confidence = probabilities[action]
                            
                            print(f"    ✅ Prediction successful!")
                            print(f"    Action: {action} ({'Buy' if action == 2 else 'Sell' if action == 0 else 'Hold'})")
                            print(f"    Confidence: {confidence:.4f}")
                            print(f"    Probabilities: {probabilities}")
                            
                            # Check if confidence meets thresholds
                            if confidence > 0.65:
                                print(f"    📈 HIGH confidence - would trade!")
                            elif confidence > 0.55:
                                print(f"    📊 MEDIUM confidence - might trade")
                            elif confidence > 0.45:
                                print(f"    📉 LOW confidence - probably no trade")
                            else:
                                print(f"    ❌ Very low confidence - no trade")
                        else:
                            print(f"    ❌ Shape mismatch: expected {self.model.input_size}, got {features_tensor.shape[-1]}")
                    break
            except Exception as e:
                print(f"    ❌ Error with sequence length {seq_len}: {str(e)}")

async def debug_backtest():
    """Debug why backtest has 0 trades"""
    system = UltimateTradingSystem()
    
    print("🎯 Debugging backtest for SPY")
    
    # Get data and process features
    data = await system.fetch_data('SPY')
    full_features = system.process_features('SPY', data)
    
    print(f"📊 Data shape: {data.shape}")
    print(f"🔧 Features shape: {full_features.shape}")
    print(f"🎯 Feature columns: {list(full_features.columns)[:5]}...") # Show first 5
    
    # Split data
    train_idx = int(len(data) * 0.7)
    val_idx = int(len(data) * 0.85)
    train_data, val_data, test_data = data.iloc[:train_idx], data.iloc[train_idx:val_idx], data.iloc[val_idx:]
    
    print(f"📈 Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Train model (using updated method)
    train_idx = int(len(data) * 0.7)
    model, metrics = await system.train_model('SPY', full_features, train_idx)
    
    if model:
        print(f"✅ Model trained successfully")
        
        # Debug model predictions
        test_features = full_features.iloc[val_idx:].drop(columns=['close'], errors='ignore')
        debug_strategy = DebugStrategy(model, test_features)
        debug_strategy.test_predictions()
        
        # Test actual backtesting strategy
        print(f"\n🧪 Testing backtest strategy:")
        try:
            from ai.strategies.rnn_trading import create_rnn_strategy_class
            from backtesting import Backtest
            
            # Prepare backtest data
            ohlc_data = test_data[['open', 'high', 'low', 'close', 'volume']].copy()
            features_only = test_features
            combined_data = ohlc_data.join(features_only, how='inner')
            combined_data = combined_data.ffill().bfill()
            
            backtest_data = combined_data.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low', 
                'close': 'Close', 'volume': 'Volume'
            })
            
            print(f"📊 Backtest data shape: {backtest_data.shape}")
            print(f"📊 Backtest columns: {list(backtest_data.columns)[:10]}...")
            
            StrategyClass = create_rnn_strategy_class(model)
            bt = Backtest(backtest_data, StrategyClass, cash=100_000, commission=0.001)
            results = bt.run()
            
            print(f"\n📈 Backtest Results:")
            print(f"  Return: {results.get('Return [%]', 0):.2f}%")
            print(f"  Trades: {results.get('# Trades', 0)}")
            print(f"  Win Rate: {results.get('Win Rate [%]', 0):.2f}%")
            
            if results.get('# Trades', 0) == 0:
                print(f"\n❌ No trades generated - strategy might be too conservative or having prediction issues")
            
        except Exception as e:
            print(f"❌ Backtest error: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ Model training failed")

if __name__ == "__main__":
    asyncio.run(debug_backtest())
