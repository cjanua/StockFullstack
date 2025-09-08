#!/usr/bin/env python3
"""
Demo script for StockFullstack AI Pipeline
Quick demonstration with minimal configuration
"""
import asyncio
import torch
import time
from pathlib import Path

from stock_fullstack.common.sdk.clients import AlpacaDataConnector
from features.feature_engine import AdvancedFeatureEngine
from agent.pytorch_system import train_lstm_model
from config.demo_config import DemoConfig


async def run_demo():
    """Run a quick demo of the AI trading pipeline."""
    print("🎯 StockFullstack AI Pipeline Demo")
    print("=" * 50)
    
    start_time = time.time()
    config = DemoConfig()
    symbol = config.DEMO_SYMBOLS[0]
    
    try:
        # 1. Data Acquisition
        print(f"\n📊 Step 1: Fetching data for {symbol}")
        data_connector = AlpacaDataConnector()
        
        data_result = await data_connector.get_data(
            symbols=[symbol], 
            lookback_days=config.LOOKBACK_DAYS
        )
        
        if not data_result.success:
            print(f"❌ Failed to fetch data: {data_result.error}")
            return
            
        raw_data = data_result.data[symbol]
        print(f"✅ Fetched {len(raw_data)} data points for {symbol}")
        
        # 2. Feature Engineering
        print(f"\n🔧 Step 2: Creating features")
        feature_engine = AdvancedFeatureEngine(config)
        
        features = feature_engine.create_features(
            raw_data, 
            symbol=symbol,
            enable_market_context=config.ENABLE_MARKET_CONTEXT
        )
        
        print(f"✅ Created features: {features.shape}")
        
        # 3. Model Training
        print(f"\n🤖 Step 3: Training model")
        model = train_lstm_model(features, symbol, config, num_epochs=config.NUM_EPOCHS)
        
        if not model:
            print("❌ Model training failed")
            return
            
        print(f"✅ Model trained successfully")
        
        # 4. Quick Prediction Test
        print(f"\n🔮 Step 4: Testing prediction")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create test sequence from the last available data
        from clean_data.pytorch_data import create_lstm_sequences
        X, y = create_lstm_sequences(
            features, 
            sequence_length=config.SEQUENCE_LENGTH,
            prediction_horizon=1
        )
        
        if len(X) > 0:
            test_input = torch.FloatTensor(X[-1:]).to(device)  # Last sequence
            
            with torch.no_grad():
                prediction = model(test_input)
                probs = torch.softmax(prediction, dim=1)
                predicted_class = torch.argmax(probs, dim=1)
                confidence = torch.max(probs, dim=1)[0]
                
                class_names = ['Down', 'Hold', 'Up']
                pred_name = class_names[predicted_class.item()]
                
                print(f"✅ Prediction: {pred_name} (confidence: {confidence.item():.2%})")
        
        # 5. Performance Summary
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n📈 Demo completed successfully!")
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Data points: {len(raw_data)}")
        print(f"   Features: {features.shape[1]}")
        print(f"   Sequences: {len(X)}")
        print(f"   Device: {device}")
        
        return {
            'success': True,
            'symbol': symbol,
            'duration': duration,
            'data_points': len(raw_data),
            'features': features.shape[1],
            'sequences': len(X),
            'device': str(device)
        }
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


async def benchmark_demo():
    """Run multiple demos to benchmark performance."""
    print("🏃 Running Benchmark Suite")
    print("=" * 50)
    
    results = []
    
    # Run demo multiple times
    for i in range(3):
        print(f"\n🔄 Benchmark Run {i+1}/3")
        result = await run_demo()
        if result['success']:
            results.append(result['duration'])
            print(f"✅ Run {i+1} completed in {result['duration']:.1f}s")
        else:
            print(f"❌ Run {i+1} failed")
    
    if results:
        avg_time = sum(results) / len(results)
        min_time = min(results)
        max_time = max(results)
        
        print(f"\n📊 Benchmark Results:")
        print(f"   Average time: {avg_time:.1f}s")
        print(f"   Fastest time: {min_time:.1f}s")
        print(f"   Slowest time: {max_time:.1f}s")
        print(f"   Runs completed: {len(results)}/3")


if __name__ == "__main__":
    print("Select demo mode:")
    print("1. Quick Demo (single run)")
    print("2. Benchmark Suite (3 runs)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        asyncio.run(benchmark_demo())
    else:
        asyncio.run(run_demo())
