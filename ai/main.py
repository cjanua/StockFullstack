import asyncio
import logging
from datetime import datetime, timedelta

from ai.features.feature_engine import AdvancedFeatureEngine
from ai.models.a3c import A3CTradingAgent
from ai.strategies.rnn_trading import run_comprehensive_backtest
from config.settings import config
from ai.models.lstm import TradingLSTM
from ai.arenas.swing_trading import SwingTradingEnv

async def main():
    """Complete RNN trading system pipeline"""
    
    # 1. Data acquisition and preprocessing
    print("📊 Acquiring market data...")
    data_connector = AlpacaDataConnector(config)
    market_data = await data_connector.get_historical_data(
        symbols=config.SYMBOLS,
        lookback_days=config.LOOKBACK_DAYS
    )
    
    # 2. Feature engineering
    print("🔧 Engineering features...")
    feature_engine = AdvancedFeatureEngine()
    processed_data = {}
    
    for symbol in config.SYMBOLS:
        symbol_data = market_data[symbol]
        features = feature_engine.create_comprehensive_features(symbol_data)
        processed_data[symbol] = features
    
    # 3. Model training
    print("🤖 Training RNN models...")
    models = {}
    
    for symbol in config.SYMBOLS:
        # Create training environment
        env = SwingTradingEnv(processed_data[symbol])
        
        # Train A3C agent
        agent = A3CTradingAgent(
            state_size=env.observation_space.shape[1],
            action_size=env.action_space.shape[0]
        )
        
        trained_agent = await train_agent(agent, env, episodes=5000)
        models[symbol] = trained_agent
        
        print(f"✅ {symbol} model trained successfully")
    
    # 4. Backtesting validation
    print("📈 Running comprehensive backtests...")
    backtest_results = {}
    
    for symbol in config.SYMBOLS:
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