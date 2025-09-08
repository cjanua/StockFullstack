#!/usr/bin/env python3
"""
Performance Benchmark Suite for StockFullstack AI Training
Measures training speed, memory usage, and model performance
"""

import time
import psutil
import os
import sys
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import torch
import numpy as np
import warnings

# Suppress common warnings
warnings.filterwarnings("ignore", message="redis-py works best with hiredis")
warnings.filterwarnings("ignore", message="Can't initialize amdsmi")

# Add paths for imports
sys.path.append('/workspace')
sys.path.append('/workspace/ai')

def get_commit_hash():
    """Get current git commit hash"""
    try:
        # Try multiple locations for git directory
        git_dirs = ['/workspace', '/app', '.']
        
        for git_dir in git_dirs:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, cwd=git_dir)
            if result.returncode == 0:
                return result.stdout.strip()[:8]  # Short hash
        
        # If git commands fail, try reading .git/HEAD directly
        for git_dir in git_dirs:
            git_head_path = Path(git_dir) / '.git' / 'HEAD'
            if git_head_path.exists():
                with open(git_head_path, 'r') as f:
                    head_content = f.read().strip()
                    if head_content.startswith('ref: '):
                        # It's a reference, read the actual commit
                        ref_path = Path(git_dir) / '.git' / head_content[5:]
                        if ref_path.exists():
                            with open(ref_path, 'r') as ref_file:
                                return ref_file.read().strip()[:8]
                    else:
                        # It's already a commit hash
                        return head_content[:8]
        
        return 'unknown'
    except:
        return 'unknown'

def get_system_info():
    """Get system information"""
    return {
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'pytorch_version': torch.__version__
    }

def monitor_resources():
    """Get current resource usage"""
    memory = psutil.virtual_memory()
    return {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': memory.percent,
        'memory_used_gb': round((memory.total - memory.available) / (1024**3), 2),
        'gpu_memory_allocated': torch.cuda.memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0,
        'gpu_memory_cached': torch.cuda.memory_reserved(0) / (1024**3) if torch.cuda.is_available() else 0
    }

class PerformanceBenchmark:
    def __init__(self, config):
        self.config = config
        self.commit_hash = get_commit_hash()
        self.system_info = get_system_info()
        self.results = {
            'benchmark_id': f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.commit_hash}",
            'timestamp': datetime.now().isoformat(),
            'commit_hash': self.commit_hash,
            'system_info': self.system_info,
            'tests': []
        }
        
        # Create results directory
        self.results_dir = Path('/workspace/model_res/benchmarks')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    async def benchmark_data_loading(self, symbols=['AAPL'], lookback_days=252):
        """Benchmark data loading performance - now includes hybrid approach"""
        print("🚀 Benchmarking hybrid data loading (Yahoo + Alpaca)...")
        
        from ai.data_sources.hybrid_manager import HybridDataManager
        
        start_time = time.time()
        start_resources = monitor_resources()
        
        try:
            # Initialize hybrid manager
            manager = HybridDataManager(self.config)
            manager_init_time = time.time() - start_time
            
            # Fetch combined data
            fetch_start = time.time()
            
            # Use await since we're in an async method now
            combined_data = await manager.get_combined_data(symbols)
            
            fetch_time = time.time() - fetch_start
            total_time = time.time() - start_time
            end_resources = monitor_resources()
            
            # Calculate statistics
            total_training_rows = sum(
                len(data['training']) for data in combined_data.values() 
                if data['training'] is not None
            )
            total_testing_rows = sum(
                len(data['testing']) for data in combined_data.values() 
                if data['testing'] is not None
            )
            
            test_result = {
                'test_name': 'hybrid_data_loading',
                'symbols': symbols,
                'lookback_days': lookback_days,
                'success': len(combined_data) > 0,
                'timing': {
                    'manager_init_seconds': round(manager_init_time, 3),
                    'data_fetch_seconds': round(fetch_time, 3),
                    'total_seconds': round(total_time, 3)
                },
                'data_stats': {
                    'symbols_processed': len(combined_data),
                    'training_rows': total_training_rows,
                    'testing_rows': total_testing_rows,
                    'total_rows': total_training_rows + total_testing_rows,
                    'symbols_with_training': sum(1 for d in combined_data.values() if d['has_training']),
                    'symbols_with_testing': sum(1 for d in combined_data.values() if d['has_testing'])
                },
                'resources': {
                    'start': start_resources,
                    'end': end_resources,
                    'memory_delta_mb': round((end_resources['memory_used_gb'] - start_resources['memory_used_gb']) * 1024, 1)
                }
            }
            
            print(f"✅ Hybrid data loading: {total_time:.2f}s")
            print(f"   Training rows: {total_training_rows}, Testing rows: {total_testing_rows}")
            return test_result
            
        except Exception as e:
            test_result = {
                'test_name': 'hybrid_data_loading',
                'symbols': symbols,
                'lookback_days': lookback_days,
                'success': False,
                'error': str(e),
                'timing': {'total_seconds': time.time() - start_time}
            }
            print(f"❌ Hybrid data loading failed: {e}")
            return test_result
    
    def benchmark_feature_engineering(self, data, symbol='AAPL'):
        """Benchmark feature engineering performance"""
        print("🔧 Benchmarking feature engineering...")
        
        from ai.features.feature_engine import AdvancedFeatureEngine
        
        start_time = time.time()
        start_resources = monitor_resources()
        
        try:
            engine = AdvancedFeatureEngine()
            features = engine.create_comprehensive_features(data, symbol)
            
            total_time = time.time() - start_time
            end_resources = monitor_resources()
            
            test_result = {
                'test_name': 'feature_engineering',
                'symbol': symbol,
                'success': features is not None and len(features) > 0,
                'timing': {
                    'total_seconds': round(total_time, 3),
                    'rows_per_second': round(len(data) / total_time, 1)
                },
                'data_stats': {
                    'input_rows': len(data),
                    'output_rows': len(features) if features is not None else 0,
                    'features_created': features.shape[1] if features is not None else 0,
                    'nan_count': features.isna().sum().sum() if features is not None else 0
                },
                'resources': {
                    'start': start_resources,
                    'end': end_resources,
                    'memory_delta_mb': round((end_resources['memory_used_gb'] - start_resources['memory_used_gb']) * 1024, 1)
                }
            }
            
            print(f"✅ Feature engineering: {total_time:.2f}s, {features.shape[1]} features")
            return test_result
            
        except Exception as e:
            test_result = {
                'test_name': 'feature_engineering',
                'symbol': symbol,
                'success': False,
                'error': str(e),
                'timing': {'total_seconds': time.time() - start_time}
            }
            print(f"❌ Feature engineering failed: {e}")
            return test_result
    
    def benchmark_training(self, features, symbol='AAPL', num_epochs=[5, 10, 20]):
        """Benchmark model training performance with different epoch counts"""
        print("🤖 Benchmarking model training...")
        
        from ai.agent.pytorch_system import train_lstm_model
        
        training_results = []
        
        for epochs in num_epochs:
            print(f"  Testing {epochs} epochs...")
            
            start_time = time.time()
            start_resources = monitor_resources()
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            try:
                model = train_lstm_model(features, symbol, self.config, num_epochs=epochs)
                
                total_time = time.time() - start_time
                end_resources = monitor_resources()
                
                # GPU memory stats
                gpu_stats = {}
                if torch.cuda.is_available():
                    gpu_stats = {
                        'peak_memory_gb': torch.cuda.max_memory_allocated(0) / (1024**3),
                        'final_memory_gb': torch.cuda.memory_allocated(0) / (1024**3)
                    }
                
                test_result = {
                    'test_name': f'training_{epochs}_epochs',
                    'symbol': symbol,
                    'epochs': epochs,
                    'success': model is not None,
                    'timing': {
                        'total_seconds': round(total_time, 3),
                        'seconds_per_epoch': round(total_time / epochs, 3),
                        'epochs_per_minute': round(epochs / (total_time / 60), 2)
                    },
                    'model_stats': {
                        'total_parameters': sum(p.numel() for p in model.parameters()) if model else 0,
                        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad) if model else 0
                    },
                    'resources': {
                        'start': start_resources,
                        'end': end_resources,
                        'memory_delta_mb': round((end_resources['memory_used_gb'] - start_resources['memory_used_gb']) * 1024, 1),
                        'gpu': gpu_stats
                    }
                }
                
                print(f"    ✅ {epochs} epochs: {total_time:.2f}s ({total_time/epochs:.2f}s/epoch)")
                training_results.append(test_result)
                
            except Exception as e:
                test_result = {
                    'test_name': f'training_{epochs}_epochs',
                    'symbol': symbol,
                    'epochs': epochs,
                    'success': False,
                    'error': str(e),
                    'timing': {'total_seconds': time.time() - start_time}
                }
                print(f"    ❌ {epochs} epochs failed: {e}")
                training_results.append(test_result)
        
        return training_results
    
    async def run_full_benchmark(self, symbols=['AAPL'], lookback_days=252, epoch_counts=[5, 10, 20]):
        """Run comprehensive benchmark suite"""
        print("🎯 Starting Full Performance Benchmark")
        print("=" * 50)
        print(f"Commit: {self.commit_hash}")
        print(f"System: {self.system_info['cpu_count']} CPU, {self.system_info['memory_gb']}GB RAM")
        if self.system_info['cuda_available']:
            print(f"GPU: {self.system_info['cuda_device_name']}")
        print("=" * 50)
        
        benchmark_start = time.time()
        
        # 1. Benchmark data loading
        data_test = await self.benchmark_data_loading(symbols, lookback_days)
        self.results['tests'].append(data_test)
        
        if not data_test['success']:
            print("❌ Data loading failed, stopping benchmark")
            return self.save_results()
        
        # Get first symbol's training data for remaining tests
        from ai.data_sources.hybrid_manager import HybridDataManager
        import asyncio
        
        async def get_training_data():
            manager = HybridDataManager(self.config)
            combined_data = await manager.get_combined_data([symbols[0]])
            return combined_data[symbols[0]]['training']
        
        try:
            training_data = await get_training_data()
            if training_data is None:
                print("❌ No training data available for benchmarking")
                return self.save_results()
        except Exception as e:
            print(f"❌ Could not get training data for benchmarking: {e}")
            return self.save_results()
        
        # 2. Benchmark feature engineering
        feature_test = self.benchmark_feature_engineering(training_data, symbols[0])
        self.results['tests'].append(feature_test)
        
        if not feature_test['success']:
            print("❌ Feature engineering failed, stopping benchmark")
            return self.save_results()
        
        # Get features for training benchmark
        from ai.features.feature_engine import AdvancedFeatureEngine
        try:
            engine = AdvancedFeatureEngine()
            features = engine.create_comprehensive_features(training_data, symbols[0])
        except:
            print("❌ Could not create features for training benchmark")
            return self.save_results()
        
        # 3. Benchmark training with different epoch counts
        training_tests = self.benchmark_training(features, symbols[0], epoch_counts)
        self.results['tests'].extend(training_tests)
        
        # Complete benchmark
        total_time = time.time() - benchmark_start
        self.results['total_benchmark_time'] = round(total_time, 3)
        
        print("=" * 50)
        print(f"🎉 Benchmark completed in {total_time:.2f} seconds")
        
        return self.save_results()
    
    def save_results(self):
        """Save benchmark results to file"""
        filename = f"benchmark_{self.results['benchmark_id']}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Also save a summary file
        summary = self.create_summary()
        summary_filename = f"summary_{self.results['benchmark_id']}.txt"
        summary_filepath = self.results_dir / summary_filename
        
        with open(summary_filepath, 'w') as f:
            f.write(summary)
        
        print(f"📊 Results saved to: {filepath}")
        print(f"📋 Summary saved to: {summary_filepath}")
        
        return filepath
    
    def create_summary(self):
        """Create a human-readable benchmark summary"""
        summary = []
        summary.append("StockFullstack AI Performance Benchmark")
        summary.append("=" * 50)
        summary.append(f"Timestamp: {self.results['timestamp']}")
        summary.append(f"Commit Hash: {self.results['commit_hash']}")
        summary.append(f"Benchmark ID: {self.results['benchmark_id']}")
        summary.append("")
        
        summary.append("System Information:")
        summary.append(f"  CPU Cores: {self.system_info['cpu_count']}")
        summary.append(f"  Memory: {self.system_info['memory_gb']} GB")
        summary.append(f"  CUDA Available: {self.system_info['cuda_available']}")
        if self.system_info['cuda_available']:
            summary.append(f"  GPU: {self.system_info['cuda_device_name']}")
        summary.append(f"  PyTorch Version: {self.system_info['pytorch_version']}")
        summary.append("")
        
        summary.append("Test Results:")
        summary.append("-" * 30)
        
        for test in self.results['tests']:
            summary.append(f"Test: {test['test_name']}")
            summary.append(f"  Success: {'✅' if test['success'] else '❌'}")
            
            if 'timing' in test and test['success']:
                if 'total_seconds' in test['timing']:
                    summary.append(f"  Time: {test['timing']['total_seconds']}s")
                if 'seconds_per_epoch' in test['timing']:
                    summary.append(f"  Time per epoch: {test['timing']['seconds_per_epoch']}s")
                if 'epochs_per_minute' in test['timing']:
                    summary.append(f"  Epochs per minute: {test['timing']['epochs_per_minute']}")
                    
            if not test['success'] and 'error' in test:
                summary.append(f"  Error: {test['error']}")
            
            summary.append("")
        
        if 'total_benchmark_time' in self.results:
            summary.append(f"Total Benchmark Time: {self.results['total_benchmark_time']}s")
        
        return "\n".join(summary)

async def main():
    """Main benchmark runner"""
    # Import config
    from ai.config.settings import TradingConfig
    
    config = TradingConfig()
    benchmark = PerformanceBenchmark(config)
    
    # Run benchmark with different configurations
    symbols = ['AAPL']  # Start with one symbol for speed
    lookback_days = 252  # 1 year
    epoch_counts = [5, 10, 20]  # Different training lengths
    
    results_path = await benchmark.run_full_benchmark(
        symbols=symbols,
        lookback_days=lookback_days,
        epoch_counts=epoch_counts
    )
    
    print(f"\n🎯 Benchmark complete! Results at: {results_path}")

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
