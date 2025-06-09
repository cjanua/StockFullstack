# ai/tests/test_config.py

import asyncio
from pathlib import Path
import pandas as pd
from ai.config.settings import TradingConfig
from backend.alpaca.sdk.clients import AlpacaDataConnector
from ai.features.feature_engine import AdvancedFeatureEngine
from ai.agent.pytorch_system import train_lstm_model
from ai.strategies.rnn_trading import RNNTradingStrategy
from backtesting import Backtest

async def test_fixes():
    """Test the drawdown fixes with minimal API calls"""
    
    class TestConfig:
        ALPACA_API_KEY = TradingConfig().ALPACA_API_KEY
        ALPACA_SECRET_KEY = TradingConfig().ALPACA_SECRET_KEY
        ALPACA_PAPER = True
        
        LSTM_INPUT_SIZE = 30
        LSTM_HIDDEN_SIZE = 128
        LSTM_NUM_LAYERS = 1
        SEQUENCE_LENGTH = 60
        LEARNING_RATE = 0.0005
        
        INITIAL_CAPITAL = 100000
        MAX_POSITION_SIZE = 0.25  # Much more conservative
        TRANSACTION_COST = 0.002
        RISK_PER_TRADE = 0.015
        
        CACHE_DIR = '.cache'
        CACHE_LIFESPAN_HOURS = 24
        
        # MINIMAL SYMBOL SET for testing
        LOOKBACK_DAYS = 252 * 1  # Reduced to 1 year to avoid Yahoo Finance issues
        SYMBOLS = ['SPY', 'QQQ', 'AAPL', 'MSFT']  # Just 4 symbols for testing
    
    config = TestConfig()

    print(f"🧪 TESTING FIXES with {len(config.SYMBOLS)} symbols: {config.SYMBOLS}")
    
    # 1. Data acquisition with rate limiting
    print("\n📊 Testing rate-limited data acquisition...")
    data_connector = AlpacaDataConnector(config)
    market_data = await data_connector.get_historical_data(
        symbols=config.SYMBOLS,
        lookback_days=config.LOOKBACK_DAYS
    )
    
    if not market_data:
        print("❌ No data fetched. Check your API credentials and rate limits.")
        return
    
    print(f"✅ Successfully fetched data for {len(market_data)} symbols")
    
    # 2. Clean data for duplicate timestamps
    print("\n🧹 Cleaning data for duplicate timestamps...")
    for symbol in market_data:
        df = market_data[symbol]
        
        print(f"Before cleaning {symbol}: {len(df)} rows")
        if df.index.has_duplicates:
            print(f"⚠️ Found duplicate timestamps in {symbol}")
            # Keep the last occurrence of each timestamp
            df = df.loc[~df.index.duplicated(keep='last')]
            print(f"After cleaning {symbol}: {len(df)} rows")
        
        # Ensure proper datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Remove any remaining NaN rows
        df = df.dropna()
        
        market_data[symbol] = df
        print(f"✅ {symbol}: {len(df)} clean rows")

    # 3. Feature engineering
    print("\n🔧 Testing feature engineering...")
    feature_engine = AdvancedFeatureEngine()
    
    # Get SPY data for market context (might be cached)
    try:
        # Use a shorter timeframe for SPY market context to avoid Yahoo Finance issues
        sample_dates = list(market_data.values())[0].index
        # Limit to last 365 days max to avoid Yahoo Finance errors
        recent_dates = sample_dates[-min(365, len(sample_dates)):]
        spy_data = feature_engine.get_market_context_data(recent_dates)
        print("✅ SPY market context data fetched successfully")
    except Exception as e:
        print(f"⚠️ SPY market context failed: {e}")
        print("Continuing without market context features...")
        spy_data = pd.DataFrame()  # Empty DataFrame as fallback
    
    processed_data = {}
    for symbol in config.SYMBOLS:
        if symbol not in market_data:
            continue
            
        try:
            features = feature_engine.create_comprehensive_features(
                market_data[symbol], symbol, spy_data
            )
            
            if not features.empty and len(features) > config.SEQUENCE_LENGTH:
                processed_data[symbol] = features
                print(f"✅ {symbol}: {len(features)} samples, {len(features.columns)} features")
            else:
                print(f"❌ {symbol}: Feature engineering failed - insufficient data")
        except Exception as e:
            print(f"❌ {symbol}: Feature engineering error - {e}")
            continue
    if not processed_data:
        print("❌ No symbols successfully processed. Exiting.")
        return
    
    # 4. Train one model to test
    print("\n🤖 Testing model training on one symbol...")
    test_symbol = config.SYMBOLS[0] if config.SYMBOLS[0] in processed_data else list(processed_data.keys())[0]
    
    try:
        model = train_lstm_model(
            processed_data[test_symbol],
            test_symbol,
            config,
            num_epochs=20  # Quick test
        )
        
        if model is None:
            print(f"❌ Model training failed for {test_symbol}")
            return
        
        print(f"✅ Model trained successfully for {test_symbol}")
    except Exception as e:
        print(f"❌ Model training error for {test_symbol}: {e}")
        return
    
    print(f"✅ Model trained successfully for {test_symbol}")
    
    # 4. Test improved strategy
    print(f"\n📈 Testing improved strategy on {test_symbol}...")
    
    def create_test_strategy(trained_model):
        class TestStrategy(RNNTradingStrategy):
            def init(self):
                self.rnn_model = trained_model
                self.rnn_model.eval()
                super().init()
        return TestStrategy
    
    # Prepare backtest data
    ohlc_data = market_data[test_symbol]
    feature_data = processed_data[test_symbol]
    
    ohlcv = ohlc_data[['open', 'high', 'low', 'close', 'volume']]
    features_only = feature_data.drop(columns=['close'], errors='ignore')
    combined_data = ohlcv.join(features_only, how='inner')
    
    combined_data.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume'
    }, inplace=True)
    
    combined_data.ffill(inplace=True)
    combined_data.bfill(inplace=True)
    
    if len(combined_data) < 100:
        print(f"❌ Insufficient combined data for {test_symbol}")
        return
    
    # Clean duplicates
    combined_data = combined_data.dropna()
    combined_data = combined_data.loc[~combined_data.index.duplicated(keep='last')]
    combined_data = combined_data.sort_index()

    # Run backtest
    strategy_class = create_test_strategy(model)
    bt = Backtest(combined_data, strategy_class, cash=100_000, commission=0.002)
    results = bt.run()
    
    # Analyze results
    print(f"\n🎯 RESULTS FOR {test_symbol.upper()}:")
    print(f"{'='*50}")
    print(f"Return:        {results['Return [%]']:>8.1f}%")
    print(f"Sharpe Ratio:  {results['Sharpe Ratio']:>8.2f}")
    print(f"Max Drawdown:  {results['Max. Drawdown [%]']:>8.1f}%")
    
    if hasattr(results, '_trades') and len(results._trades) > 0:
        trades = results._trades
        win_rate = len(trades[trades['ReturnPct'] > 0]) / len(trades) * 100
        print(f"Trades:        {len(trades):>8}")
        print(f"Win Rate:      {win_rate:>8.1f}%")
    
    max_dd = abs(results['Max. Drawdown [%]'])
    
    print(f"{'='*50}")
    
    if max_dd < 20:
        print("🎉 EXCELLENT: Max drawdown under 20%!")
    elif max_dd < 30:
        print("✅ GOOD: Max drawdown under 30%")
    elif max_dd < 50:
        print("⚠️  IMPROVED: Max drawdown under 50% (better than 68%)")
    else:
        print("❌ NEEDS MORE WORK: Still high drawdown")
    
    # Save results
    plot_path = Path("model_res/test_results") / f"test_{test_symbol}_improved.html"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    bt.plot(filename=str(plot_path), open_browser=False)
    print(f"\n📊 Results saved to: {plot_path}")

if __name__ == "__main__":
    asyncio.run(test_fixes())
