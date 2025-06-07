# ai/main.py
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd

from ai.agent.pytorch_system import train_lstm_model
from ai.features.feature_engine import AdvancedFeatureEngine
from ai.models.a3c import A3CTradingAgent
from ai.strategies.rnn_trading import run_comprehensive_backtest
from ai.config.settings import config
from ai.models.lstm import TradingLSTM
from ai.arenas.swing_trading import SwingTradingEnv, train_agent_parallel
from ai.arenas.swing_trading import train_agent_parallel as train_agent
from ai.agent.alpaca_system import AlpacaTradingSystem as LiveTradingSystem

from ai.arenas.swing_trading import SwingTradingEnv

from ai.strategies.rnn_trading import create_rnn_strategy_class
from ai.monitoring.performance_metrics import analyze_portfolio_performance
from backend.alpaca.sdk.clients import AlpacaDataConnector
from ai.clean_data.preprocessing import split_data_by_periods

from dotenv import load_dotenv
load_dotenv()


async def main():
    """Complete RNN trading system pipeline"""
    
    # 1. Data acquisition and preprocessing
    print("📊 Acquiring market data...")
    data_connector = AlpacaDataConnector(config)
    market_data = await data_connector.get_historical_data(
        symbols=config.SYMBOLS,
        lookback_days=config.LOOKBACK_DAYS
    )

    print("📊 Acquiring market context data (SPY)...")
    feature_engine = AdvancedFeatureEngine()
    # Get a union of all dates from the data we already downloaded
    all_dates = pd.DatetimeIndex([])
    for df in market_data.values():
        all_dates = all_dates.union(df.index)

    spy_data = feature_engine.get_market_context_data(all_dates)

    # 2. Feature engineering
    print("🔧 Engineering features...")
    feature_engine = AdvancedFeatureEngine()
    processed_data = {}
    
    for symbol in config.SYMBOLS:
        if symbol not in market_data: continue # Skip if data failed to download

        symbol_data = market_data[symbol]
        features = feature_engine.create_comprehensive_features(symbol_data, spy_data)
        if not features.empty:
            processed_data[symbol] = features
    
    # 3. Model training
    BENCHMARK_SYMBOL = 'SPY'
    trading_symbols = [s for s in config.SYMBOLS if s != BENCHMARK_SYMBOL]

    print("🤖 Training RNN models...")
    models = {}
    
    for symbol in trading_symbols:
        if symbol not in processed_data: continue
        print(f"Training model for {symbol}...")

        trained_model = train_lstm_model(
            processed_data[symbol], 
            config,
            num_epochs=5 # Let's use a smaller number for faster iteration
        )
        models[symbol] = trained_model
        
        print(f"✅ {symbol} model trained successfully")

    
    # 4. Backtesting validation
    print("📈 Running comprehensive backtests...")
    backtest_results = {}
    
    for symbol in trading_symbols:
        if symbol not in processed_data or symbol not in models: continue
        ohlc_data = market_data[symbol]
        feature_data = processed_data[symbol]

        print(ohlc_data.head())
        print(ohlc_data.columns)

        backtest_df = ohlc_data.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })

        combined_data = pd.concat([backtest_df, feature_data], axis=1)
        combined_data.dropna(inplace=True)  # Ensure no NaN values

        ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        feature_columns = [col for col in combined_data.columns if col not in ohlcv_columns]
        
        if len(feature_columns) >= config.LSTM_INPUT_SIZE:
            final_feature_cols = feature_columns[:config.LSTM_INPUT_SIZE]
            # Create the final dataframe with OHLCV and the fixed feature set
            final_backtest_data = combined_data[ohlcv_columns + final_feature_cols]
        else:
            print(f"Warning: Not enough features for backtesting {symbol}. Skipping.")
            continue

        print(f"\nDEBUG: Columns for backtesting {symbol}: {final_backtest_data.columns.to_list()}\n")

        strategy_class = create_rnn_strategy_class(models[symbol])
        results = run_comprehensive_backtest(
            final_backtest_data, 
            strategy_class
        )
        backtest_results[symbol] = results

        # --- ADD THIS BLOCK TO PRINT DETAILED RESULTS ---
        print(f"\n----------- Backtest Results for {symbol} -----------")
        # The results object from backtesting.py can be printed directly
        print(results['backtest_results']) 
        print(f"--------------------------------------------------\n")
        # --- END ADDITION ---

        # This part is still useful for a final summary
        sharpe = results.get('backtest_results', {}).get('Sharpe Ratio', 'N/A')
        if isinstance(sharpe, float):
            print(f"📊 {symbol} backtest complete - Sharpe: {sharpe:.2f}")
        else:
            print(f"📊 {symbol} backtest complete - Sharpe: {sharpe}")
    
    # 5. Performance validation
    performance_summary = analyze_portfolio_performance(backtest_results)
    
    if performance_summary['portfolio_sharpe'] > 1.0:
        print(f"🎯 Portfolio Sharpe ratio: {performance_summary['portfolio_sharpe']:.2f} - Ready for live trading!")
        
        # 6. Deploy to live trading (paper first)
        live_system = LiveTradingSystem(config, models)
        await live_system.start_trading()
        
    else:
        print(f"⚠️  Portfolio Sharpe ratio: {performance_summary['portfolio_sharpe']:.2f} - Requires optimization")

if __name__ == "__main__":
    asyncio.run(main())