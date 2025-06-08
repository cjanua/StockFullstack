# ai/main.py
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from backtesting import Backtest

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

        features = feature_engine.create_comprehensive_features(market_data[symbol], symbol, spy_data)
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

        try:
            from ai.agent.pytorch_system import train_lstm_model
            trained_model = train_lstm_model(
                processed_data[symbol], 
                symbol,
                config,
                num_epochs=50,
            )
            
            # If it fails or returns None, use simple fallback
            if trained_model is None:
                print(f"  - Standard training failed for {symbol}, trying simple approach...")
                from ai.agent.simple_training import simple_train_lstm_model
                trained_model = simple_train_lstm_model(
                    processed_data[symbol],
                    symbol,
                    config,
                    num_epochs=30,
                )
        except Exception as e:
            print(f"  - Training error for {symbol}: {e}")
            print(f"  - Trying simple fallback approach...")
            try:
                from ai.agent.simple_training import simple_train_lstm_model
                trained_model = simple_train_lstm_model(
                    processed_data[symbol],
                    symbol,
                    config,
                    num_epochs=30,
                )
            except Exception as e2:
                print(f"  - Simple training also failed for {symbol}: {e2}")
                trained_model = None


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
    def display_performance_summary(backtest_results, models):
        """Display concise performance summary for statistical analysis"""
        print("\n" + "="*120)
        print("📊 PERFORMANCE SUMMARY & STATISTICAL ANALYSIS")
        print("="*120)
        
        asset_stats = []
        
        for symbol, result_dict in backtest_results.items():
            if 'backtest_results' not in result_dict:
                continue
                
            results = result_dict['backtest_results']
            trades = results._trades if hasattr(results, '_trades') else pd.DataFrame()
            
            if trades.empty:
                continue
                
            # Calculate trade statistics
            winning_trades = trades[trades['ReturnPct'] > 0]
            losing_trades = trades[trades['ReturnPct'] <= 0]
            
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            total_trades = len(trades)
            
            win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
            loss_rate = (loss_count / total_trades * 100) if total_trades > 0 else 0
            
            avg_win = winning_trades['ReturnPct'].mean() if win_count > 0 else 0
            avg_loss = losing_trades['ReturnPct'].mean() if loss_count > 0 else 0
            
            largest_win = winning_trades['ReturnPct'].max() if win_count > 0 else 0
            largest_loss = losing_trades['ReturnPct'].min() if loss_count > 0 else 0
            
            # Risk metrics
            profit_factor = results.get('Profit Factor', 0)
            sharpe_ratio = results.get('Sharpe Ratio', 0)
            max_drawdown = results.get('Max. Drawdown [%]', 0)
            total_return = results.get('Return [%]', 0)
            
            # Expectancy calculation
            expectancy = (win_rate/100 * avg_win) + (loss_rate/100 * avg_loss)
            
            asset_stats.append({
                'Symbol': symbol,
                'Return %': total_return,
                'Sharpe': sharpe_ratio,
                'Max DD %': max_drawdown,
                'Trades': total_trades,
                'Win %': win_rate,
                'Loss %': loss_rate,
                'Avg Win %': avg_win,
                'Avg Loss %': avg_loss,
                'Best %': largest_win,
                'Worst %': largest_loss,
                'PF': profit_factor,
                'Expect %': expectancy,
                'Trained': symbol in models
            })
        
        # Create DataFrame for easy display
        df = pd.DataFrame(asset_stats)
        
        if df.empty:
            print("No valid backtest results to display.")
            return df
        
        # Display individual asset performance
        print("\n📈 INDIVIDUAL ASSET PERFORMANCE")
        print("-" * 120)
        
        # Custom formatting for cleaner display
        for _, row in df.iterrows():
            trained_status = "✅" if row['Trained'] else "❌"
            print(f"{row['Symbol']:>6} | "
                  f"Return: {row['Return %']:>7.1f}% | "
                  f"Sharpe: {row['Sharpe']:>6.2f} | "
                  f"MaxDD: {row['Max DD %']:>6.1f}% | "
                  f"Trades: {row['Trades']:>3.0f} | "
                  f"Win: {row['Win %']:>5.1f}% | "
                  f"AvgW: {row['Avg Win %']:>5.2f}% | "
                  f"AvgL: {row['Avg Loss %']:>6.2f}% | "
                  f"PF: {row['PF']:>5.2f} | "
                  f"Expect: {row['Expect %']:>6.2f}% | "
                  f"Model: {trained_status}")
        
        # Portfolio-level statistics
        print(f"\n📊 PORTFOLIO STATISTICS")
        print("-" * 60)
        
        total_trades_all = df['Trades'].sum()
        avg_win_rate = df['Win %'].mean()
        avg_sharpe = df['Sharpe'].mean()
        avg_return = df['Return %'].mean()
        profitable_assets = len(df[df['Return %'] > 0])
        total_assets = len(df)
        
        print(f"Assets Traded:         {total_assets}")
        print(f"Profitable Assets:     {profitable_assets} ({profitable_assets/total_assets*100:.1f}%)")
        print(f"Total Trades:          {total_trades_all:.0f}")
        print(f"Average Win Rate:      {avg_win_rate:.1f}%")
        print(f"Average Sharpe:        {avg_sharpe:.2f}")
        print(f"Average Return:        {avg_return:.1f}%")
        
        # Performance tiers
        excellent = len(df[df['Sharpe'] > 1.0])
        good = len(df[(df['Sharpe'] > 0.5) & (df['Sharpe'] <= 1.0)])
        poor = len(df[df['Sharpe'] <= 0.5])
        
        print(f"\n🎯 PERFORMANCE TIERS")
        print(f"Excellent (Sharpe >1.0):   {excellent}")
        print(f"Good (Sharpe 0.5-1.0):     {good}")
        print(f"Poor (Sharpe ≤0.5):        {poor}")
        
        # Best and worst performers
        if len(df) > 0:
            best = df.loc[df['Sharpe'].idxmax(), 'Symbol']
            worst = df.loc[df['Sharpe'].idxmin(), 'Symbol']
            print(f"\n🏆 Best Performer:         {best} (Sharpe: {df.loc[df['Sharpe'].idxmax(), 'Sharpe']:.2f})")
            print(f"💥 Worst Performer:        {worst} (Sharpe: {df.loc[df['Sharpe'].idxmin(), 'Sharpe']:.2f})")
        
        print("\n" + "="*120)
        return df
    
    # Generate the summary
    performance_df = display_performance_summary(backtest_results, models)

if __name__ == "__main__":
    asyncio.run(main())
