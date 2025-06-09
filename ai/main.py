# ai/main.py
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from backtesting import Backtest

from ai.agent.pytorch_system import train_lstm_model
from ai.clean_data.asset_grouping import concatenate_asset_data, prepare_grouped_datasets
from ai.features.feature_engine import AdvancedFeatureEngine
from ai.models.a3c import A3CTradingAgent
from ai.strategies.rnn_trading import RNNTradingStrategy, run_comprehensive_backtest
from ai.config.settings import TradingConfig
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
    config = TradingConfig()

    # 1. Data acquisition and preprocessing
    print("📊 Acquiring market data...")
    data_connector = AlpacaDataConnector(config)
    market_data = await data_connector.get_historical_data(
        symbols=config.SYMBOLS,
        lookback_days=config.LOOKBACK_DAYS
    )

    print("📊 Acquiring market context data (SPY)...")
    feature_engine = AdvancedFeatureEngine()
    all_dates = pd.DatetimeIndex([])
    for df in market_data.values():
        all_dates = all_dates.union(df.index)
    spy_data = feature_engine.get_market_context_data(all_dates)

    # 2. Feature engineering
    print("🔧 Engineering features...")
    processed_data = {}
    
    for symbol in config.SYMBOLS:
        if symbol not in market_data:
            continue

        features = feature_engine.create_comprehensive_features(
            market_data[symbol], symbol, spy_data)

        if not features.empty and len(features) > config.SEQUENCE_LENGTH * 2:
            nan_threshold = 0.05
            good_features = features.loc[:, features.isnull().sum() / len(features) < nan_threshold]
            if len(good_features) < 10:
                print(f"Warning: No valid features for {symbol}, skipping...")
                continue
            processed_data[symbol] = good_features
            print(f"✅ {symbol}: {len(good_features)} samples, {len(good_features.columns)} features")
        else:
            print(f"Warning: Insufficient features for {symbol}, skipping...")
    
    # 3. Model training
    BENCHMARK_SYMBOL = 'SPY'
    trading_symbols = [s for s in config.SYMBOLS if s != BENCHMARK_SYMBOL]

    print("🤖 Training RNN models...")
    models = {}
    
    for symbol in config.SYMBOLS:
        if symbol not in processed_data:
            continue
            
        # Flexible parameters - you can adjust these!
        trained_model = train_lstm_model(
            processed_data[symbol],
            symbol,
            config,        # or "ensemble"
            num_epochs=150       # or any number you want
        )
        
        if trained_model is not None:
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

        def create_strategy_class(trained_model):
            class Strategy(RNNTradingStrategy):
                def init(self):
                    self.rnn_model = trained_model
                    self.rnn_model.eval()
                    super().init()
            return Strategy


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


        StrategyClass = create_strategy_class(models[symbol])
        bt = Backtest(combined_data, StrategyClass, cash=100_000, commission=0)
        results = bt.run()
        backtest_results[symbol] = {'backtest_results': results} # Store results

        print(f"\n--- Improved Results for {symbol} ---")
        print(f"Return: {results['Return [%]']:.1f}%")
        print(f"Sharpe: {results['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {results['Max. Drawdown [%]']:.1f}%")
        print(f"Trades: {len(results._trades) if hasattr(results, '_trades') else 0}")

        # Plotting
        plot_filename = Path("model_res/backtests") / f"backtest_{symbol}.html"
        plot_filename.parent.mkdir(parents=True, exist_ok=True)
        bt.plot(filename=str(plot_filename), open_browser=False)

    
    print("\n" + "="*100)
    print("📊 PERFORMANCE SUMMARY")
    print("="*100)
    
    total_positive = 0
    total_sharpe_above_05 = 0
    max_dd_list = []
    
    for symbol, result_dict in backtest_results.items():
        if 'backtest_results' not in result_dict:
            continue
            
        results = result_dict['backtest_results']
        ret = results.get('Return [%]', 0)
        sharpe = results.get('Sharpe Ratio', 0)
        max_dd = results.get('Max. Drawdown [%]', 0)
        
        if ret > 0:
            total_positive += 1
        if sharpe > 0.5:
            total_sharpe_above_05 += 1
            
        max_dd_list.append(abs(max_dd))
        
        print(f"{symbol:>6} | Return: {ret:>6.1f}% | Sharpe: {sharpe:>5.2f} | MaxDD: {max_dd:>6.1f}%")
    
    print(f"\nSUMMARY:")
    print(f"Positive Returns: {total_positive}/{len(backtest_results)} ({total_positive/len(backtest_results)*100:.1f}%)")
    print(f"Sharpe > 0.5: {total_sharpe_above_05}/{len(backtest_results)} ({total_sharpe_above_05/len(backtest_results)*100:.1f}%)")
    print(f"Average Max Drawdown: {np.mean(max_dd_list):.1f}%")
    print(f"Worst Max Drawdown: {max(max_dd_list):.1f}%")
    
    if max(max_dd_list) < 25:  # Much better than 68%!
        print("🎯 MAJOR IMPROVEMENT: Maximum drawdown under 25%!")
    
    print("="*100)


if __name__ == "__main__":
    asyncio.run(main())
