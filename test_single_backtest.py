#!/usr/bin/env python3
"""
Quick test to verify backtest is actually working
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ai.main import UltimateTradingSystem

async def test_single_backtest():
    """Test with just SPY to see if backtest completes"""
    system = UltimateTradingSystem()
    
    print("🎯 Testing single symbol: SPY")
    result = await system.process_symbol('SPY', force_cache_refresh=False)
    
    training_result, backtest_result = result
    
    print(f"\n📊 Training Result:")
    print(f"  Success: {training_result.success}")
    print(f"  Accuracy: {training_result.final_accuracy:.4f}")
    print(f"  Time: {training_result.training_time:.2f}s")
    
    print(f"\n📈 Backtest Result:")
    print(f"  Success: {backtest_result.success}")
    if backtest_result.success:
        print(f"  Total Return: {backtest_result.total_return:.4f}")
        print(f"  Annual Return: {backtest_result.annual_return:.4f}")
        print(f"  Sharpe Ratio: {backtest_result.sharpe_ratio:.4f}")
        print(f"  Max Drawdown: {backtest_result.max_drawdown:.4f}")
        print(f"  Win Rate: {backtest_result.win_rate:.4f}")
        print(f"  Total Trades: {backtest_result.total_trades}")
    else:
        print(f"  Error: {backtest_result.error_message}")

if __name__ == "__main__":
    asyncio.run(test_single_backtest())
