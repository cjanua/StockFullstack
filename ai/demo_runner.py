#!/usr/bin/env python3
"""
Quick StockFullstack AI Demo Runner
"""
import sys
import os

# Add paths
sys.path.append('/app')
sys.path.append('/workspace')
sys.path.append('/workspace/ai')

def quick_demo():
    print('🎯 Quick StockFullstack AI Demo')
    print('=' * 40)
    
    try:
        from ai.models.lstm import TradingLSTM
        from ai.config.settings import TradingConfig
        import torch
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        import time
        
        start_time = time.time()
        print('🔧 Initializing components...')
        
        config = TradingConfig()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'📱 Using device: {device}')
        
        # Generate mock data for demo (30 days of AAPL-like data)
        print('📊 Generating mock AAPL data (30 days)...')
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Realistic stock price simulation
        base_price = 150.0
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': [p + np.random.normal(0, 0.5) for p in prices],
            'high': [p + abs(np.random.normal(1, 0.8)) for p in prices],
            'low': [p - abs(np.random.normal(1, 0.8)) for p in prices],
            'close': prices,
            'volume': np.random.randint(50000000, 150000000, len(dates))
        })
        
        # Set timestamp as index for feature engineering
        data = data.set_index('timestamp')
        
        print(f'✅ Generated {len(data)} data points')
        
        # Simple feature engineering for demo
        print('🔧 Creating features...')
        
        # Basic price features
        data['returns'] = data['close'].pct_change()
        data['sma_5'] = data['close'].rolling(5).mean()
        data['sma_10'] = data['close'].rolling(10).mean()
        data['volatility'] = data['returns'].rolling(5).std()
        data['price_change'] = data['close'].diff()
        
        # Create feature matrix
        feature_columns = ['returns', 'sma_5', 'sma_10', 'volatility', 'price_change']
        features = data[feature_columns].fillna(0)
        
        print(f'✅ Created {len(feature_columns)} features from {len(features)} samples')
        
        # Quick LSTM training simulation (3 epochs)
        print('🤖 Quick LSTM training (3 epochs)...')
        
        # Prepare minimal training data
        feature_data = features.select_dtypes(include=[np.number]).fillna(0)
        if len(feature_data) > 10:
            X = feature_data.values[-15:-5]  # 10 samples for training
            y = data['close'].values[-15:-5]
            
            # Simple LSTM model
            input_size = X.shape[1]
            model = TradingLSTM(input_size=input_size, hidden_size=16, num_layers=1, output_size=1)
            model = model.to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.MSELoss()
            
            X_tensor = torch.FloatTensor(X).unsqueeze(0).to(device)  # Add batch dimension
            y_tensor = torch.FloatTensor(y).unsqueeze(0).to(device)
            
            # Quick training loop
            for epoch in range(3):
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs.squeeze(), y_tensor.squeeze())
                loss.backward()
                optimizer.step()
                
            print(f'✅ Training completed! Final loss: {loss.item():.4f}')
            
            # Make a prediction
            test_input = feature_data.values[-1:].reshape(1, 1, -1)
            test_tensor = torch.FloatTensor(test_input).to(device)
            
            with torch.no_grad():
                prediction = model(test_tensor)
                pred_price = prediction.item()
                
            current_price = data['close'].iloc[-1]
            print('🔮 Prediction Results:')
            print(f'   Current price: ${current_price:.2f}')
            print(f'   AI prediction: ${pred_price:.2f}')
            print(f'   Difference: ${abs(pred_price - current_price):.2f}')
        
        else:
            print('⚠️ Insufficient data for training, using mock results')
            print('🔮 Mock prediction: Next day price ~$151.50')
        
        duration = time.time() - start_time
        print(f'⚡ Demo completed in {duration:.1f} seconds')
        print('✅ Demo successful!')
        
    except Exception as e:
        print(f'❌ Demo failed: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    quick_demo()
