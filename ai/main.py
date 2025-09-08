import asyncio
import os
from pathlib import Path
import hashlib
import torch
import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest
from dotenv import load_dotenv

from clean_data.utils import validate_data
from agent.pytorch_system import train_lstm_model
from config.settings import TradingConfig
from features.feature_engine import AdvancedFeatureEngine
from strategies.rnn_trading import RNNTradingStrategy, perform_walk_forward_analysis

load_dotenv()

# GPU optimization setup
if torch.cuda.is_available():
    # Show CUDA information
    print(f"CUDA is available with {torch.cuda.device_count()} device(s)")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    
    # Optimize memory usage
    torch.cuda.empty_cache()
    
    # Set memory allocation strategy
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = os.environ.get(
        'PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128'
    )
else:
    print("CUDA is not available, using CPU")
def create_strategy_class(trained_model):
    class Strategy(RNNTradingStrategy):
        def init(self):
            self.rnn_model = trained_model
            self.rnn_model.eval()
            super().init()
    return Strategy

async def main():
    config = TradingConfig()
    
    # Apply environment variable overrides for training parameters
    if 'BATCH_SIZE' in os.environ:
        batch_size = int(os.environ['BATCH_SIZE'])
        print(f"Using custom batch size from environment: {batch_size}")
        # Will be picked up by create_pytorch_dataloaders
    
    print(f"SYMBOLS from config: {config.SYMBOLS}")
    
    # Set the default tensor type to float32 for better efficiency
    torch.set_default_tensor_type(torch.FloatTensor)
    BENCHMARK_SYMBOL = 'SPY'
    trading_symbols = [s for s in config.SYMBOLS if s != BENCHMARK_SYMBOL]
    if not trading_symbols:
        print(
            "No symbols to trade and backtest. "
            f"The SYMBOLS list in your config must contain at least one symbol "
            f"other than the benchmark ('{BENCHMARK_SYMBOL}')."
        )
        print("Execution stopped.")
        return
    config.USE_UNCERTAINTY = True
    config.ENHANCED_FEATURES = False
    # 1. Data acquisition and preprocessing
    print("📊 Acquiring market data from Yahoo Finance for extended history...")
    market_data = {}
    start_date = '2010-01-01'
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    for symbol in config.SYMBOLS:
        yf_data = yf.download(symbol, start=start_date, end=end_date)
        if isinstance(yf_data.columns, pd.MultiIndex):
            yf_data.columns = yf_data.columns.get_level_values(0)
        yf_data = yf_data.drop(columns=['Adj Close'], errors='ignore')
        yf_data = yf_data.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low', 
            'Close': 'close', 'Volume': 'volume'
        })
        yf_data['trade_count'] = np.nan
        yf_data['vwap'] = (yf_data['open'] + yf_data['high'] + yf_data['low'] + yf_data['close']) / 4

        # Validate data
        if not validate_data(yf_data, symbol):
            continue

        extended_data = yf_data
        extended_data['trade_count'] = extended_data['trade_count'].fillna(extended_data['trade_count'].median())
        extended_data = extended_data.ffill().bfill()
        market_data[symbol] = extended_data
        print(f"✅ Fetched and extended {symbol}: {len(extended_data)} rows from {extended_data.index.min()} to {extended_data.index.max()}")

    # Standardize SPY data
    if 'SPY' in market_data:
        spy_data = market_data['SPY']
        print(f"✅ Standardized SPY data: {len(spy_data)} rows")
    else:
        print("⚠️ No SPY data; fallback to empty DataFrame")
        spy_data = pd.DataFrame()

    # 2. Feature engineering
    print("🔧 Engineering features...")
    feature_engine = AdvancedFeatureEngine()
    processed_data = {}
    for symbol in market_data:
        features = feature_engine.create_comprehensive_features(
            market_data[symbol], symbol, spy_data)
        if not features.empty and len(features) > config.SEQUENCE_LENGTH * 2:
            nan_threshold = 0.05
            good_features = features.loc[:, features.isnull().sum() / len(features) < nan_threshold]
            if len(good_features.columns) < 10:
                print(f"Warning: Insufficient valid features for {symbol}, skipping...")
                continue
            processed_data[symbol] = good_features
            print(f"✅ {symbol}: {len(good_features)} samples, {len(good_features.columns)} features")
        else:
            print(f"Warning: Insufficient features for {symbol}, skipping...")

    # 3. Model training
    project_root = Path(os.getenv("PROJECT_PATH", "."))
    cache_dir = Path(os.environ.get("MODEL_CACHE_DIR", project_root / "model_res" / "cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    print("🤖 Training RNN models...")
    models = {}
    for symbol in processed_data:
        features = processed_data[symbol]
        if len(features) < config.SEQUENCE_LENGTH * 3:
            print(f"Skipping {symbol}: too few samples after split consideration")
            continue
        split_idx = int(len(features) * 0.7)
        train_data = features.iloc[:split_idx]
        test_data = features.iloc[split_idx:]
        holdout_idx = int(len(test_data) * 0.8)
        val_data = test_data.iloc[:holdout_idx]
        holdout_data = test_data.iloc[holdout_idx:]
        print(f"Data splits for {symbol}: train={len(train_data)}, val={len(val_data)}, holdout={len(holdout_data)}")

        model_type = 'standard'
        config_hash = hashlib.sha256(str({
            'symbols': config.SYMBOLS,
            'lookback': config.LOOKBACK_DAYS,
            'epochs': config.NUM_EPOCHS,
            'seq_len': config.SEQUENCE_LENGTH,
            'type': model_type
        }).encode()).hexdigest()
        train_hash = hashlib.sha256(str({
            'rows': len(train_data),
            'first_date': str(train_data.index[0]),
            'last_date': str(train_data.index[-1])
        }).encode()).hexdigest()
        full_hash = config_hash + train_hash
        model_path = cache_dir / f"model_{symbol}_{full_hash}.pth"

        if model_path.exists():
            trained_model = train_lstm_model(
                train_data, symbol, config, num_epochs=0, model_type=model_type)
            trained_model.load_state_dict(torch.load(model_path))
            print(f" - ✅ Loaded cached model for {symbol} from {model_path}")
        else:
            trained_model = train_lstm_model(
                train_data, symbol, config, num_epochs=config.NUM_EPOCHS, model_type=model_type)
            if trained_model is not None:
                torch.save(trained_model.state_dict(), model_path)
                print(f" - ✅ Saved model cache at {model_path}")
        if trained_model is not None:
            models[symbol] = trained_model
            print(f" - ✅ {symbol} model trained successfully.")
        else:
            print(f" - ❌ {symbol} model training failed.")

    # 4. Backtesting validation with walk-forward
    print("📈 Running comprehensive backtests with walk-forward optimization...")
    backtest_results = {}
    for symbol in trading_symbols:
        if symbol not in processed_data or symbol not in models:
            print(f"Skipping backtest for {symbol}: no processed data or model available")
            continue
        print(f"Starting backtest iteration for {symbol}")

        # Extend data with yfinance if needed
        try:
            yf_data = yf.download(symbol, start=market_data[symbol].index.min() - pd.Timedelta(days=365), end=market_data[symbol].index.max())
            if not yf_data.empty:
                if isinstance(yf_data.columns, pd.MultiIndex):
                    yf_data.columns = yf_data.columns.get_level_values(0)
                yf_data = yf_data.drop(columns=['Adj Close'], errors='ignore')
                yf_data = yf_data.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low', 
                    'Close': 'close', 'Volume': 'volume'
                })
                if validate_data(yf_data, symbol):
                    extended_data = pd.concat([yf_data, market_data[symbol]]).sort_index().drop_duplicates(keep='last')
                    market_data[symbol] = extended_data
                    print(f"Extended {symbol} data with yfinance: {len(extended_data)} rows")
                else:
                    print(f"Skipping backtest for {symbol}: extended data invalid")
                    continue
        except Exception as e:
            print(f"yfinance extension failed for {symbol}: {e}")

        # Recompute features
        features = feature_engine.create_comprehensive_features(market_data[symbol], symbol, spy_data)
        good_features = features.loc[:, features.isnull().sum() / len(features) < 0.05]
        if len(good_features.columns) < 10:
            print(f"Skipping backtest for {symbol}: insufficient valid features")
            continue
        processed_data[symbol] = good_features
        print(f"Recomputed features for {symbol}: {len(good_features)} rows")

        # Split data
        split_idx = int(len(good_features) * 0.8)
        train_data = good_features.iloc[:split_idx]
        test_data = good_features.iloc[split_idx:]
        holdout_idx = int(len(test_data) * 0.8)
        val_data = test_data.iloc[:holdout_idx]
        holdout_data = test_data.iloc[holdout_idx:]
        print(f"Data splits for {symbol}: train={len(train_data)}, val={len(val_data)}, holdout={len(holdout_data)}")

        # Walk-forward optimization
        wf_results = perform_walk_forward_analysis(val_data, create_strategy_class(models[symbol]), train_window=252, test_window=63)
        if wf_results.empty:
            print(f"No walk-forward results for {symbol} due to insufficient data")
            continue

        print(f"Walk-forward results for {symbol}: Avg Return {wf_results['return'].mean():.1f}%")

        # Prepare holdout data for backtesting
        ohlc_data_raw = market_data[symbol].loc[holdout_data.index]
        ohlc_data_raw = ohlc_data_raw.loc[~ohlc_data_raw.index.duplicated(keep='last')]
        features_only = holdout_data.drop(columns=['close'], errors='ignore')
        features_only = features_only.loc[~features_only.index.duplicated(keep='last')]
        combined_data = ohlc_data_raw[['open', 'high', 'low', 'close', 'volume']].join(features_only, how='inner')
        combined_data = combined_data.loc[~combined_data.index.duplicated(keep='last')]
        combined_data.ffill(inplace=True)
        combined_data.bfill(inplace=True)
        print(f"Combined holdout data for {symbol}: {len(combined_data)} rows")

        if len(combined_data) < 50:
            print(f"Skipping backtest for {symbol}: holdout data too short (<50 rows)")
            continue

        if not validate_data(combined_data, symbol, min_rows=50):
            print(f"Skipping backtest for {symbol}: invalid holdout data")
            continue

        # Rename columns for backtesting library
        backtest_data = combined_data.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 
            'close': 'Close', 'volume': 'Volume'
        })

        StrategyClass = create_strategy_class(models[symbol])
        try:
            bt = Backtest(backtest_data, StrategyClass, cash=100_000, commission=0.001)
            results = bt.run()
            backtest_results[symbol] = {'backtest_results': results, 'wf_results': wf_results}
            print(f"\n--- Improved Results for {symbol} ---")
            print(f"Return: {results['Return [%]']:.1f}%")
            print(f"Sharpe: {results['Sharpe Ratio']:.2f}")
            print(f"Max Drawdown: {results['Max. Drawdown [%]']:.1f}%")
            print(f"Trades: {len(results._trades) if hasattr(results, '_trades') else 0}")
        except Exception as e:
            print(f"Backtest failed for {symbol}: {e}")
            continue

        # Plotting
        project_root = Path(os.getenv("PROJECT_PATH", "."))
        plot_filename = project_root / "model_res" / "backtests" / f"backtest_{symbol}.html"
        plot_filename.parent.mkdir(parents=True, exist_ok=True)
        bt.plot(filename=str(plot_filename), open_browser=False)

    # Performance summary
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

    print("\nSUMMARY:")
    num_results = len(backtest_results)
    if num_results > 0:
        positive_pct = (total_positive / num_results) * 100
        sharpe_pct = (total_sharpe_above_05 / num_results) * 100
        print(f"Positive Returns: {total_positive}/{num_results} ({positive_pct:.1f}%)")
        print(f"Sharpe > 0.5: {total_sharpe_above_05}/{num_results} ({sharpe_pct:.1f}%)")
    else:
        print("Positive Returns: 0/0")
        print("Sharpe > 0.5: 0/0")

    if max_dd_list:
        avg_max_dd = np.mean(max_dd_list)
        worst_max_dd = max(max_dd_list)
        print(f"Average Max Drawdown: {avg_max_dd:.1f}%")
        print(f"Worst Max Drawdown: {worst_max_dd:.1f}%")
        if worst_max_dd < 25:
            print("🎯 MAJOR IMPROVEMENT: Maximum drawdown under 25%!")
    else:
        print("Average Max Drawdown: N/A")
        print("Worst Max Drawdown: N/A")
    print("="*100)

def run_main():
    asyncio.run(main())

if __name__ == "__main__":
    asyncio.run(main())