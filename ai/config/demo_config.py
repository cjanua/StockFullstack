# Demo configuration for quick testing and benchmarks
from config.settings import TradingConfig


class DemoConfig(TradingConfig):
    """Minimized configuration for demos and benchmarks."""
    
    # Reduced dataset for fast testing
    LOOKBACK_DAYS = 30  # Instead of 252
    SEQUENCE_LENGTH = 20  # Instead of 60
    
    # Minimal model for quick training
    LSTM_HIDDEN_SIZE = 64  # Instead of 128
    NUM_EPOCHS = 5  # Instead of 20
    BATCH_SIZE = 16  # Instead of 32
    
    # Single symbol for demo
    DEMO_SYMBOLS = ['AAPL']
    
    # Reduced features for faster processing
    ENABLE_TECHNICAL_INDICATORS = True
    ENABLE_MARKET_CONTEXT = False  # Disable heavy market context for demo
    ENABLE_CROSS_ASSET = False  # Disable cross-asset features
    
    # Fast validation
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 3
    
    def __init__(self):
        super().__init__()
        print("🚀 Demo Configuration Loaded:")
        print(f"  - Lookback days: {self.LOOKBACK_DAYS}")
        print(f"  - Sequence length: {self.SEQUENCE_LENGTH}")
        print(f"  - LSTM hidden size: {self.LSTM_HIDDEN_SIZE}")
        print(f"  - Training epochs: {self.NUM_EPOCHS}")
        print(f"  - Demo symbols: {self.DEMO_SYMBOLS}")
        print("  - Optimized for speed and demos")
