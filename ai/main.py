# ai/main.py
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from backtesting import Backtest

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
from ai.utils import print_integrity_check
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

        features = feature_engine.create_comprehensive_features(market_data[symbol], spy_data)
        if not features.empty and len(features) > config.SEQUENCE_LENGTH:
            processed_data[symbol] = features
        else:
            print(f"Warning: Insufficient features for {symbol}, skipping...")
    
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
            symbol,
            config,
            num_epochs=100,
        )

        if trained_model:
            models[symbol] = trained_model
            print(f"  - ✅ {symbol} model trained successfully.")
        else:
            print(f"  - ❌ {symbol} model training failed.")
        

    
    # 4. Backtesting validation
    print("📈 Running comprehensive backtests...")
    backtest_results = {}
    
    for symbol in trading_symbols:
        if symbol not in processed_data or symbol not in models:
            continue
        ohlc_data_raw = market_data[symbol]
        feature_data = processed_data[symbol]

        if ohlc_data_raw.index.has_duplicates:
            ohlc_data_raw = ohlc_data_raw.loc[~ohlc_data_raw.index.duplicated(keep='first')]

        ohlcv_for_backtest = ohlc_data_raw[['open', 'high', 'low', 'close', 'volume']]

        features_only = feature_data.drop(columns=['close'], errors='ignore')

        combined_data = ohlcv_for_backtest.join(features_only, how='inner')

        combined_data.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)

        combined_data.ffill(inplace=True)
        combined_data.bfill(inplace=True)

        if combined_data.empty or len(combined_data) < 60:
            print(f"  - WARNING: Not enough combined data for {symbol}, skipping backtest.")
            continue


        StrategyClass = create_rnn_strategy_class(models[symbol])
    
        bt = Backtest(combined_data, StrategyClass, cash=100_000, commission=.002)
        results = bt.run()
        backtest_results[symbol] = {'backtest_results': results} # Store results

        print(f"\n----------- Backtest Results for {symbol} -----------")
        print(results)
        print("--------------------------------------------------\n")

        # Plotting
        plot_filename = Path("model_res/backtests") / f"backtest_{symbol}.html"
        plot_filename.parent.mkdir(parents=True, exist_ok=True)
        bt.plot(filename=str(plot_filename), open_browser=False)

    
    print("6. Analyzing portfolio performance...")
    # NOTE: The performance analysis part depends on the structure of backtest_results
    # which I have simplified above. You may need to adjust analyze_portfolio_performance
    # if you re-introduce walk-forward analysis etc.
    
    # This simplified check is based on individual backtests
    for symbol, result_dict in backtest_results.items():
        sharpe = result_dict.get('backtest_results', {}).get('Sharpe Ratio', 0)
        if sharpe > 1.0:
            print(f"✅ {symbol} passed performance check with Sharpe Ratio: {sharpe:.2f}")
        else:
            print(f"⚠️ {symbol} requires optimization. Sharpe Ratio: {sharpe:.2f}")

if __name__ == "__main__":
    asyncio.run(main())