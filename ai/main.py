# ai/main.py
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd

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

        trained_model = train_agent_parallel(
            processed_data[symbol], 
            total_timesteps=250000 # Let's use a smaller number for faster iteration
        )
        models[symbol] = trained_model
        
        print(f"✅ {symbol} model trained successfully")

    
    # 4. Backtesting validation
    print("📈 Running comprehensive backtests...")
    backtest_results = {}
    
    for symbol in trading_symbols:
        if symbol not in processed_data or symbol not in models: continue
        
        strategy_class = create_rnn_strategy_class(models[symbol])
        results = run_comprehensive_backtest(
            processed_data[symbol], 
            strategy_class
        )
        backtest_results[symbol] = results
        
        print(f"📊 {symbol} backtest complete - Sharpe: {results['backtest_results']['Sharpe Ratio']:.2f}")
    
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