# ai/strategies/rnn_trading.py
from backtesting import Backtest, Strategy
import numpy as np
import pandas as pd
import torch

from ai.monitoring.performance_metrics import get_benchmark_returns, test_statistical_significance, calculate_comprehensive_risk_metrics
from ai.features.feature_engine import AdvancedFeatureEngine

class RNNTradingStrategy(Strategy):
    def init(self):
        # Load pre-trained RNN model
        # self.rnn_model = torch.load('trained_rnn_model.pth')
        # self.rnn_model.eval()
        
        # # Feature engineering pipeline
        # self.feature_engine = AdvancedFeatureEngine()
        
        # Portfolio state tracking
        self.portfolio_value_history = []
        self.signals_history = []

        self.signal_strength = self.I(lambda: pd.Series(self.signals_history), name='Signal')

        
    def next(self):
        # Minimum data requirement
        if len(self.data) < 60:
            return
        
        ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        feature_cols = [col for col in self.data.df.columns if col not in ohlcv_columns]
        
        features_df = self.data.df[feature_cols].iloc[-60:][feature_cols]
        features = features_df.values
        # Get RNN prediction
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            prediction = self.rnn_model(features_tensor)
            buy_prob = torch.softmax(prediction, dim=1).numpy()[0][1]
        
        # Trading logic with risk management
        signal_strength = buy_prob
        
        self.signals_history.append(signal_strength)

        # Position management
        if signal_strength > 0.5 and not self.position:
            # Calculate position size
            # size = self.calculate_position_size(buy_prob)
            self.buy(size=0.1)
            
        elif signal_strength < 0.48 and self.position:
            self.position.close()
        
        # Track performance
        self.portfolio_value_history.append(self.equity)
        self.signals_history.append(signal_strength)
    
    def calculate_position_size(self, signal_strength):
        """Dynamic position sizing based on signal confidence"""
        max_position = 0.95  # 95% maximum allocation
        base_size = signal_strength * max_position
        
        # Volatility adjustment
        recent_volatility = self.data.Close[-20:].pct_change().std()
        vol_adjustment = max(0.5, 1.0 - recent_volatility * 10)
        
        return base_size * vol_adjustment

# Comprehensive backtesting workflow
def run_comprehensive_backtest(data, strategy_class, plt_file):
    """Execute full backtesting pipeline with performance analysis"""
    
    # Primary backtest
    bt = Backtest(data, strategy_class, cash=100000, commission=0.002)
    results = bt.run()
    
    if plt_file:
        bt.plot(filename=plt_file, open_browser=True)

    # Walk-forward analysis
    wf_results = perform_walk_forward_analysis(data, strategy_class)
    
    # Statistical significance testing
    benchmark_returns = get_benchmark_returns(data.index[0], data.index[-1])
    significance_tests = test_statistical_significance(
        results._trades['ReturnPct'], 
        benchmark_returns
    )
    
    # Risk analysis
    risk_metrics = calculate_comprehensive_risk_metrics(results)
    
    return {
        'backtest_results': results,
        'walk_forward': wf_results,
        'significance_tests': significance_tests,
        'risk_metrics': risk_metrics
    }

def perform_walk_forward_analysis(data, strategy_class, 
                                 train_window=504, test_window=63):
    """Walk-forward analysis with parameter optimization"""
    results = []
    
    for i in range(train_window, len(data) - test_window, test_window):
        # Training period
        train_data = data.iloc[i-train_window:i]
        
        # Parameter optimization on training data
        bt_train = Backtest(train_data, strategy_class)
        optimized_params = bt_train.optimize(
            signal_threshold=range(60, 81, 5),
            position_size=np.arange(0.7, 1.0, 0.1),
            constraint=lambda p: p.signal_threshold < 80
        )
        
        # Out-of-sample testing
        test_data = data.iloc[i:i+test_window]
        bt_test = Backtest(test_data, strategy_class)
        test_results = bt_test.run(**optimized_params)
        
        results.append({
            'period_start': test_data.index[0],
            'period_end': test_data.index[-1],
            'return': test_results['Return [%]'],
            'sharpe': test_results['Sharpe Ratio'],
            'max_drawdown': test_results['Max. Drawdown [%]'],
            'win_rate': len(test_results._trades[test_results._trades['ReturnPct'] > 0]) / len(test_results._trades)
        })
    
    return pd.DataFrame(results)


def create_rnn_strategy_class(trained_model):
    """
    Factory function to create a new Strategy class with a specific pre-trained model.
    
    The backtesting library requires a class, so we can't just pass the model to an
    instance. Instead, we dynamically create a new class that has the model
    "baked in".
    """
    
    class DynamicRNNStrategy(RNNTradingStrategy):
        """
        A dynamically created strategy class that uses a pre-loaded model.
        """
        def init(self):
            # --- Override the parent's init method ---
            
            # 1. Use the trained model passed in from the factory
            self.rnn_model = trained_model
            self.rnn_model.eval()
            
            # 2. Copy the rest of the setup from the parent class
            self.feature_engine = AdvancedFeatureEngine()
            self.portfolio_value_history = []
            self.signals_history = []

            # Note: We do NOT call super().init() because we are
            # intentionally overriding the model loading behavior.

    return DynamicRNNStrategy