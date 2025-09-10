#!/usr/bin/env python3
"""
Test to identify exactly where the system hangs
"""
import asyncio
import sys
from pathlib import Path
import time

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ai.main import UltimateTradingSystem

async def test_with_timeouts():
    """Test with explicit timeouts to identify hanging point"""
    system = UltimateTradingSystem()
    
    # Test just 3 symbols to avoid long runs
    test_symbols = ['SPY', 'TQQQ', 'SQQQ']
    
    print(f"🎯 Testing {len(test_symbols)} symbols with timeouts")
    
    for symbol in test_symbols:
        print(f"\n🔄 Processing {symbol}...")
        start_time = time.time()
        
        try:
            # Add timeout to each symbol processing
            result = await asyncio.wait_for(
                system.process_symbol(symbol, force_cache_refresh=False),
                timeout=300  # 5 minute timeout per symbol
            )
            
            training_result, backtest_result = result
            elapsed = time.time() - start_time
            
            print(f"✅ {symbol} completed in {elapsed:.1f}s")
            print(f"   Training: {training_result.success} ({training_result.final_accuracy:.2%})")
            print(f"   Backtest: {backtest_result.success} ({backtest_result.total_trades} trades)")
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print(f"⏰ {symbol} TIMEOUT after {elapsed:.1f}s")
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ {symbol} ERROR after {elapsed:.1f}s: {str(e)}")
    
    print(f"\n🎉 Test completed!")

if __name__ == "__main__":
    asyncio.run(test_with_timeouts())
