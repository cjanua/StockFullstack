#!/usr/bin/env python3
"""
Ultimate Main.py: Comprehensive Model Training and Backtesting System
Enhanced with feature alignment, cache management, and robust error handling.
"""

import asyncio
import os
import sys
import json
import warnings
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from backtesting import Backtest
from sklearn.decomposition import PCA  # Import PCA for external handling if needed

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import custom modules
from ai.cache_utils import cache_on_disk
from ai.config.settings import TradingConfig
from ai.agent.pytorch_system import train_lstm_model
from ai.data_sources.hybrid_manager import HybridDataManager
from ai.features.feature_engine import AdvancedFeatureEngine
from ai.models.lstm import TradingLSTM, create_lstm, EnsembleLSTM, CustomTradingLoss
from ai.strategies.rnn_trading import RNNTradingStrategy, create_rnn_strategy_class, perform_walk_forward_analysis
from clean_data.utils import validate_data
from clean_data.pytorch_data import create_sequences, create_pytorch_dataloaders

load_dotenv()

# GPU optimization setup
if torch.cuda.is_available():
    print(f"CUDA is available with {torch.cuda.device_count()} device(s)")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')
else:
    print("CUDA is not available, using CPU")

@dataclass
class TrainingResult:
    symbol: str
    success: bool
    training_time: float
    final_accuracy: float
    final_loss: float
    epochs_completed: int
    best_accuracy: float
    model_path: str
    feature_columns: List[str]
    error_message: str = ""

@dataclass
class BacktestResult:
    symbol: str
    success: bool
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    avg_trade_return: float
    volatility: float
    calmar_ratio: float
    error_message: str = ""

@dataclass
class ComprehensiveReport:
    timestamp: str
    total_assets: int
    successful_trainings: int
    successful_backtests: int
    total_training_time: float
    system_info: Dict[str, Any]
    training_results: List[TrainingResult]
    backtest_results: List[BacktestResult]
    summary_statistics: Dict[str, float]

class UltimateTradingSystem:
    """Enhanced trading system with feature alignment and cache management."""

    def __init__(self):
        self.config = TradingConfig()
        self.data_manager = HybridDataManager(self.config)
        self.feature_engine = AdvancedFeatureEngine(use_pca=False, n_pca_components=self.config.PCA_COMPONENTS)  # Disable internal PCA for control
        self.results_dir = project_root / "model_res" / "ultimate"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.results_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Suppress warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

    async def fetch_data(self, symbol: str) -> pd.DataFrame:
        """Fetch and validate data asynchronously."""
        print(f"📊 Fetching data for {symbol}...")
        data = await asyncio.to_thread(
            self.data_manager.yahoo_loader.get_historical_data,
            [symbol],
            lookback_days=self.config.LOOKBACK_DAYS
        )
        data = data.get(symbol, pd.DataFrame())
        
        if data.empty or not validate_data(data, symbol):
            print(f"⚠️ Invalid or empty data for {symbol}")
            return pd.DataFrame()
        
        data = data.ffill().bfill()
        print(f"✅ Fetched {len(data)} rows for {symbol}")
        return data

    @cache_on_disk(dependencies=['ai/features/feature_engine.py'])
    def process_features(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Process features with caching and external PCA if enabled."""
        print(f"🔧 Engineering features for {symbol}...")
        features = self.feature_engine.create_comprehensive_features(data, symbol)
        
        nan_threshold = 0.05
        good_features = features.loc[:, features.isnull().sum() / len(features) < nan_threshold]
        if len(good_features.columns) < 10:
            print(f"⚠️ Insufficient features for {symbol}")
            return pd.DataFrame()
        
        # If USE_PCA, apply externally
        if self.config.USE_PCA:
            pca = PCA(n_components=self.config.PCA_COMPONENTS)
            pca_features = pca.fit_transform(good_features)
            good_features = pd.DataFrame(pca_features, index=good_features.index, columns=[f'pca_{i}' for i in range(self.config.PCA_COMPONENTS)])
            print(f"PCA explained variance ratio: {np.sum(pca.explained_variance_ratio_):.3f}")
        
        print(f"✅ Generated {len(good_features.columns)} features for {symbol}")
        return good_features

    async def train_model(self, symbol: str, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Tuple[torch.nn.Module, Dict[str, float]]:
        """Enhanced training with validation, scheduler, and feature consistency."""
        print(f"🚀 Training model for {symbol}...")
        start_time = datetime.now()
        
        try:
            train_features = await asyncio.to_thread(self.process_features, symbol, train_data)
            if train_features.empty:
                raise ValueError("No features generated for train data")
            
            val_features = await asyncio.to_thread(self.process_features, symbol, val_data)
            if val_features.empty:
                raise ValueError("No features generated for val data")
            
            # Align val_features to train_features columns
            expected_columns = train_features.columns
            val_features = val_features.reindex(columns=expected_columns, fill_value=0)
            
            # Use volatility-normalized thresholds for labeling
            train_atr = (train_data['high'] - train_data['low']).rolling(14).mean().shift(1)
            val_atr = (val_data['high'] - val_data['low']).rolling(14).mean().shift(1)
            train_threshold = train_atr * 0.5
            val_threshold = val_atr * 0.5
            
            # Create sequences with adaptive thresholds
            X_train, y_train = create_sequences(train_features, train_data['close'], self.config.SEQUENCE_LENGTH)
            X_val, y_val = create_sequences(val_features, val_data['close'], self.config.SEQUENCE_LENGTH)
            
            if len(X_train) == 0 or len(X_val) == 0:
                raise ValueError("Insufficient sequences for training")
            
            train_loader = create_pytorch_dataloaders(train_features, train_data['close'], self.config)
            val_loader = create_pytorch_dataloaders(val_features, val_data['close'], self.config)
            
            input_size = X_train.shape[-1] if len(X_train) > 0 else len(train_features.columns) - 1
            model = create_lstm(input_size, model_type='ensemble' if self.config.USE_ENSEMBLE else 'standard', hidden_size=self.config.LSTM_HIDDEN_SIZE)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
            criterion = CustomTradingLoss()
            
            # Train loop with validation
            best_val_acc = 0
            for epoch in range(self.config.NUM_EPOCHS):
                model.train()
                train_loss = 0
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0
                val_correct = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        output = model(batch_x)
                        val_loss += criterion(output, batch_y).item()
                        pred = output.argmax(dim=1)
                        val_correct += (pred == batch_y).sum().item()
                
                val_acc = val_correct / len(val_loader.dataset)
                scheduler.step(val_loss / len(val_loader))
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
            
            training_time = (datetime.now() - start_time).total_seconds()
            metrics = {
                'final_accuracy': val_acc,
                'final_loss': val_loss / len(val_loader),
                'epochs_completed': self.config.NUM_EPOCHS,
                'best_accuracy': best_val_acc,
                'training_time': training_time
            }
            
            model_path = str(self.results_dir / f"{symbol}_model.pth")
            torch.save(model.state_dict(), model_path)
            
            return model, metrics
        
        except Exception as e:
            print(f"❌ Training failed for {symbol}: {str(e)}")
            return None, {'final_accuracy': 0, 'final_loss': 0, 'epochs_completed': 0, 'best_accuracy': 0, 'training_time': 0}

    async def backtest_model(self, symbol: str, model: torch.nn.Module, test_data: pd.DataFrame) -> BacktestResult:
        """Advanced backtesting with RNNTradingStrategy, walk-forward, and benchmarks."""
        print(f"📊 Backtesting {symbol}...")
        
        try:
            features = await asyncio.to_thread(self.process_features, symbol, test_data)
            if features.empty:
                raise ValueError("No features for backtesting")
            
            # Prepare backtest data - ensure proper column naming
            # Check if columns are already properly named or need renaming
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            existing_cols = test_data.columns.tolist()
            
            # Debug: Print current columns
            print(f"📊 Current columns for {symbol}: {existing_cols}")
            
            # Create mapping for column renaming
            rename_map = {}
            for req_col in required_cols:
                # Check for lowercase version first
                if req_col.lower() in existing_cols:
                    rename_map[req_col.lower()] = req_col
                # Check if already properly named
                elif req_col not in existing_cols:
                    print(f"⚠️ Missing column {req_col} for {symbol}")
            
            backtest_data = test_data.rename(columns=rename_map)
            
            # Ensure we have the required columns
            missing_cols = [col for col in required_cols if col not in backtest_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Select only required columns plus features
            backtest_data = backtest_data[required_cols]
            backtest_data = backtest_data.join(features, how='inner').ffill().bfill()
            
            print(f"📊 Backtest data shape for {symbol}: {backtest_data.shape}")
            print(f"📊 Backtest columns: {backtest_data.columns.tolist()[:10]}...")
            
            StrategyClass = create_rnn_strategy_class(model)
            backtest_instance = Backtest(backtest_data, StrategyClass, cash=100_000, commission=0.001)
            results = backtest_instance.run()
            
            # Walk-forward analysis
            wf_results = perform_walk_forward_analysis(backtest_data, StrategyClass)
            
            # Benchmark comparison (e.g., buy-and-hold SPY) - skip for SPY itself
            spy_results = {'Return [%]': 0}  # Default fallback
            if symbol != 'SPY':
                try:
                    spy_data = await self.fetch_data('SPY')
                    if not spy_data.empty and len(spy_data) >= len(backtest_data):
                        # Prepare SPY data with same structure as backtest_data
                        spy_rename_map = {col.lower(): col for col in ['Open', 'High', 'Low', 'Close', 'Volume']}
                        spy_backtest_data = spy_data.rename(columns=spy_rename_map)
                        spy_backtest_data = spy_backtest_data[['Open', 'High', 'Low', 'Close', 'Volume']]
                        spy_backtest_data = spy_backtest_data.loc[backtest_data.index].ffill().bfill()
                        
                        if not spy_backtest_data.empty:
                            spy_backtest = Backtest(spy_backtest_data, StrategyClass, cash=100_000, commission=0.001)
                            spy_results = spy_backtest.run()
                except Exception as e:
                    print(f"⚠️ SPY benchmark failed: {str(e)}")
                    spy_results = {'Return [%]': 0}
            
            # Monte Carlo simulation (simple version: resample trades)
            if len(results._trades) > 0:
                trade_returns = results._trades['ReturnPct'].values
                mc_returns = [np.prod(1 + np.random.choice(trade_returns, len(trade_returns))) - 1 for _ in range(1000)]
                mc_avg = np.mean(mc_returns)
            else:
                mc_avg = 0
            
            # Compile results
            return BacktestResult(
                symbol=symbol,
                success=True,
                total_return=results['Return [%]'] / 100,
                annual_return=(results['Return [%]'] / 100) / (len(backtest_data) / 252),
                sharpe_ratio=results['Sharpe Ratio'],
                max_drawdown=results['Max. Drawdown [%]'] / 100,
                win_rate=results['Win Rate [%]'] / 100 if 'Win Rate [%]' in results else 0,
                total_trades=len(results._trades),
                profitable_trades=len(results._trades[results._trades['ReturnPct'] > 0]),
                avg_trade_return=results._trades['ReturnPct'].mean() if len(results._trades) > 0 else 0,
                volatility=results._trades['ReturnPct'].std() * np.sqrt(252) if len(results._trades) > 0 else 0,
                calmar_ratio=results['Calmar Ratio'] if 'Calmar Ratio' in results else 0
            )
        
        except Exception as e:
            print(f"❌ Backtest failed for {symbol}: {str(e)}")
            return BacktestResult(symbol=symbol, success=False, error_message=str(e), total_return=0, annual_return=0, sharpe_ratio=0, max_drawdown=0, win_rate=0, total_trades=0, profitable_trades=0, avg_trade_return=0, volatility=0, calmar_ratio=0)

    async def process_symbol(self, symbol: str, force_cache_refresh: bool = False) -> Tuple[TrainingResult, BacktestResult]:
        """Process single symbol with training and backtesting, with cache refresh option."""
        if force_cache_refresh:
            for cache_file in self.cache_dir.glob(f"*{symbol}*.joblib"):
                cache_file.unlink()
            print(f"🔄 Cache refreshed for {symbol}")
        
        data = await self.fetch_data(symbol)
        if data.empty:
            return (TrainingResult(symbol=symbol, success=False, error_message="No data", training_time=0, final_accuracy=0, final_loss=0, epochs_completed=0, best_accuracy=0, model_path="", feature_columns=[]),
                    BacktestResult(symbol=symbol, success=False, error_message="No data", total_return=0, annual_return=0, sharpe_ratio=0, max_drawdown=0, win_rate=0, total_trades=0, profitable_trades=0, avg_trade_return=0, volatility=0, calmar_ratio=0))
        
        # Enhanced splits: 70% train, 15% val, 15% test
        train_idx = int(len(data) * 0.7)
        val_idx = int(len(data) * 0.85)
        train_data, val_data, test_data = data.iloc[:train_idx], data.iloc[train_idx:val_idx], data.iloc[val_idx:]
        
        model, metrics = await self.train_model(symbol, train_data, val_data)
        training_result = TrainingResult(
            symbol=symbol,
            success=bool(model),
            training_time=metrics.get('training_time', 0),
            final_accuracy=metrics['final_accuracy'],
            final_loss=metrics['final_loss'],
            epochs_completed=metrics['epochs_completed'],
            best_accuracy=metrics['best_accuracy'],
            model_path=str(self.results_dir / f"{symbol}_model.pth") if model else "",
            feature_columns=list(self.process_features(symbol, train_data).columns),
            error_message="" if model else "Training failed"
        )
        
        backtest_result = await self.backtest_model(symbol, model, test_data) if model else BacktestResult(
            symbol=symbol, success=False, error_message="No model", total_return=0, annual_return=0, sharpe_ratio=0, max_drawdown=0, win_rate=0, total_trades=0, profitable_trades=0, avg_trade_return=0, volatility=0, calmar_ratio=0
        )
        
        return training_result, backtest_result

    async def run_comprehensive_analysis(self, symbols: List[str] = None, force_cache_refresh: bool = False) -> ComprehensiveReport:
        """Run full analysis with parallel symbol processing and cache management."""
        if symbols is None:
            symbols = self.config.SYMBOLS
        
        print(f"🎯 Analyzing {len(symbols)} assets: {', '.join(symbols)}")
        start_time = datetime.now()
        
        # Parallel processing with asyncio.gather
        tasks = [self.process_symbol(symbol, force_cache_refresh) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        training_results = []
        backtest_results = []
        for res in results:
            if isinstance(res, Exception):
                print(f"⚠️ Error in processing: {str(res)}")
                continue
            tr, br = res
            training_results.append(tr)
            backtest_results.append(br)
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        report = ComprehensiveReport(
            timestamp=datetime.now().isoformat(),
            total_assets=len(symbols),
            successful_trainings=sum(1 for r in training_results if r.success),
            successful_backtests=sum(1 for r in backtest_results if r.success),
            total_training_time=total_time,
            system_info=self._get_system_info(),
            training_results=training_results,
            backtest_results=backtest_results,
            summary_statistics=self._calculate_summary_statistics(training_results, backtest_results)
        )
        
        self._save_reports(report)
        return report

    def _get_system_info(self) -> Dict[str, Any]:
        return {
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count(),
            'pytorch_version': torch.__version__,
            'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'sequence_length': self.config.SEQUENCE_LENGTH,
            'lstm_hidden_size': self.config.LSTM_HIDDEN_SIZE,
            'learning_rate': self.config.LEARNING_RATE,
            'num_epochs': self.config.NUM_EPOCHS,
            'use_ensemble': self.config.USE_ENSEMBLE,
            'use_pca': self.config.USE_PCA
        }

    def _calculate_summary_statistics(self, training_results: List[TrainingResult], backtest_results: List[BacktestResult]) -> Dict[str, float]:
        succ_train = [r for r in training_results if r.success]
        succ_back = [r for r in backtest_results if r.success]
        return {
            'avg_training_time': np.mean([r.training_time for r in succ_train]) if succ_train else 0,
            'avg_final_accuracy': np.mean([r.final_accuracy for r in succ_train]) if succ_train else 0,
            'avg_total_return': np.mean([r.total_return for r in succ_back]) if succ_back else 0,
            'avg_annual_return': np.mean([r.annual_return for r in succ_back]) if succ_back else 0,
            'avg_sharpe_ratio': np.mean([r.sharpe_ratio for r in succ_back]) if succ_back else 0,
            'avg_max_drawdown': np.mean([r.max_drawdown for r in succ_back]) if succ_back else 0,
            'avg_win_rate': np.mean([r.win_rate for r in succ_back]) if succ_back else 0,
        }

    def _save_reports(self, report: ComprehensiveReport):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.results_dir / f"ultimate_report_{timestamp}.json"
        md_path = self.results_dir / f"ultimate_report_{timestamp}.md"
        
        with open(json_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        with open(md_path, 'w') as f:
            f.write("# Ultimate Trading System Report\n\n")
            f.write(f"**Generated:** {report.timestamp}\n")
            f.write(f"**Assets Processed:** {report.total_assets}\n")
            f.write(f"**Successful Trainings:** {report.successful_trainings}\n")
            f.write(f"**Successful Backtests:** {report.successful_backtests}\n")
            f.write(f"**Total Runtime:** {report.total_training_time:.2f}s\n\n")
            
            f.write("## System Information\n")
            for k, v in report.system_info.items():
                f.write(f"- {k.title()}: {v}\n")
            f.write("\n")
            
            f.write("## Summary Statistics\n")
            for k, v in report.summary_statistics.items():
                f.write(f"- {k.replace('_', ' ').title()}: {v:.4f}\n")
            f.write("\n")
            
            f.write("## Training Results\n")
            f.write("| Symbol | Success | Time (s) | Accuracy | Loss | Epochs | Best Acc |\n")
            f.write("|--------|---------|----------|----------|------|--------|----------|\n")
            for r in report.training_results:
                status = "✅" if r.success else "❌"
                f.write(f"| {r.symbol} | {status} | {r.training_time:.2f} | {r.final_accuracy:.4f} | {r.final_loss:.4f} | {r.epochs_completed} | {r.best_accuracy:.4f} |\n")
            f.write("\n")
            
            f.write("## Backtest Results\n")
            f.write("| Symbol | Success | Total Ret | Ann Ret | Sharpe | Max DD | Win Rate | Trades |\n")
            f.write("|--------|---------|-----------|---------|--------|--------|----------|--------|\n")
            for r in report.backtest_results:
                status = "✅" if r.success else "❌"
                f.write(f"| {r.symbol} | {status} | {r.total_return:.4f} | {r.annual_return:.4f} | {r.sharpe_ratio:.4f} | {r.max_drawdown:.4f} | {r.win_rate:.4f} | {r.total_trades} |\n")
        
        print(f"💾 Reports saved: {json_path}, {md_path}")

async def main():
    """Main entrypoint with cache refresh option."""
    system = UltimateTradingSystem()
    report = await system.run_comprehensive_analysis(force_cache_refresh=True)  # Force cache refresh for initial run
    
    print("\n🎉 Analysis Complete!")
    print(f"✅ Trainings: {report.successful_trainings}/{report.total_assets}")
    print(f"✅ Backtests: {report.successful_backtests}/{report.total_assets}")
    print(f"⏱️ Runtime: {report.total_training_time:.2f}s")
    print(f"📊 Avg Accuracy: {report.summary_statistics['avg_final_accuracy']:.4f}")
    print(f"📈 Avg Ann Return: {report.summary_statistics['avg_annual_return']:.4f}")
    print(f"📉 Avg Sharpe: {report.summary_statistics['avg_sharpe_ratio']:.4f}")

if __name__ == "__main__":
    asyncio.run(main())