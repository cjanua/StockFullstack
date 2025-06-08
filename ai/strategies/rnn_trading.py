# ai/strategies/rnn_trading.py
from backtesting import Backtest, Strategy
import numpy as np
import pandas as pd
import torch

from ai.monitoring.performance_metrics import get_benchmark_returns, test_statistical_significance, calculate_comprehensive_risk_metrics
from ai.features.feature_engine import AdvancedFeatureEngine

class RNNTradingStrategy(Strategy):
    stop_loss_pct = 0.025
    take_profit_pct = 0.05
    max_position_size = 0.70
    increment_position_size = 0.33
    confidence_threshold = 0.35


    def init(self):
        self.portfolio_value_history = []
        self.signals_history = []
        self.last_prediction_time = None
        self.prediction_cache = None
        self.consecutive_losses = 0
        self.recent_trades = []

        self.current_long_position = 0.0   # Track cumulative long position
        self.current_short_position = 0.0  # Track cumulative short position
        self.position_levels = [] 
        
    def next(self):
        # Warm-up
        if len(self.data) < 60:
            return
        
        current_time = self.data.index[-1]

        # Cache predictions to avoid recalculation
        if self.last_prediction_time != current_time:
            self.prediction_cache = self._get_prediction()
            self.last_prediction_time = current_time
        
        if self.prediction_cache is None:
            return

        action, confidence = self.prediction_cache
        current_price = self.data.Close[-1]

        self._update_performance_tracking()

        self._manage_cascading_positions(action, confidence, current_price)

        # if self.position:
        #     self._manage_existing_position(action, confidence, current_price)
        # else:
        #     self._enter_new_position(action, confidence, current_price)



    def _manage_cascading_positions(self, action, confidence, current_price):
        """Manage positions with cascading thirds"""
        
        if confidence < self.confidence_threshold:
            return
            
        increment = self.max_position_size * self.increment_position_size  # FIX: Use max_position_size
        
        if action == 2:  # UP signal
            self._handle_up_signal(increment, current_price, confidence)
        elif action == 0:  # DOWN signal  
            self._handle_down_signal(increment, current_price, confidence)
        # action == 1 (HOLD) - do nothing

    def _handle_up_signal(self, increment, current_price, confidence):
        """Handle UP signal with cascading logic"""
        
        # If we have short positions, close them first (pyramid out of shorts)
        if self.current_short_position > 0:
            close_size = min(increment, self.current_short_position)
            self._close_short_position(close_size, "Signal reversal")
            return
        
        # If we're neutral or long, add to long position (pyramid into longs)
        max_long_position = self.max_position_size  # FIX: Use max_position_size
        if self.current_long_position < max_long_position:
            # Calculate how much we can still add
            remaining_capacity = max_long_position - self.current_long_position
            actual_increment = min(increment, remaining_capacity)
            
            if actual_increment > 0.05:  # Only trade if meaningful size (>5%)
                self._add_long_position(actual_increment, current_price, confidence)

    def _handle_down_signal(self, increment, current_price, confidence):
        """Handle DOWN signal with cascading logic"""
        
        # If we have long positions, close them first (pyramid out of longs)
        if self.current_long_position > 0:
            close_size = min(increment, self.current_long_position)
            self._close_long_position(close_size, "Signal reversal")
            return
            
        # If we're neutral or short, add to short position (pyramid into shorts)
        max_short_position = self.max_position_size  # FIX: Use max_position_size
        if self.current_short_position < max_short_position:
            # Calculate how much we can still add
            remaining_capacity = max_short_position - self.current_short_position
            actual_increment = min(increment, remaining_capacity)
            
            if actual_increment > 0.05:  # Only trade if meaningful size (>5%)
                self._add_short_position(actual_increment, current_price, confidence)

    def _add_long_position(self, size, entry_price, confidence):
        """Add to long position"""
        sl = entry_price * (1 - self.stop_loss_pct)
        tp = entry_price * (1 + self.take_profit_pct)
        
        # Execute the trade
        self.buy(size=size, sl=sl, tp=tp)
        
        # Track the position
        self.current_long_position += size
        self.position_levels.append({
            'type': 'long',
            'size': size,
            'entry_price': entry_price,
            'confidence': confidence,
            'timestamp': self.data.index[-1]
        })
        
        print(f"Added LONG {size:.2f} at {entry_price:.2f} (Total long: {self.current_long_position:.2f})")

    def _add_short_position(self, size, entry_price, confidence):
        """Add to short position"""
        sl = entry_price * (1 + self.stop_loss_pct)
        tp = entry_price * (1 - self.take_profit_pct)
        
        # Execute the trade
        self.sell(size=size, sl=sl, tp=tp)
        
        # Track the position
        self.current_short_position += size
        self.position_levels.append({
            'type': 'short',
            'size': size,
            'entry_price': entry_price,
            'confidence': confidence,
            'timestamp': self.data.index[-1]
        })
        
        print(f"Added SHORT {size:.2f} at {entry_price:.2f} (Total short: {self.current_short_position:.2f})")

    def _close_long_position(self, size, reason):
        """Close portion of long position"""
        if self.current_long_position <= 0:
            return
            
        # This is conceptual - backtesting.py handles actual position closing
        # In live trading, you'd implement partial position closing here
        
        self.current_long_position = max(0, self.current_long_position - size)
        
        # Remove closed positions from tracking
        self._update_position_levels('long', size)
        
        print(f"Closed LONG {size:.2f} ({reason}) - Remaining long: {self.current_long_position:.2f}")

    def _close_short_position(self, size, reason):
        """Close portion of short position"""
        if self.current_short_position <= 0:
            return
            
        self.current_short_position = max(0, self.current_short_position - size)
        
        # Remove closed positions from tracking
        self._update_position_levels('short', size)
        
        print(f"Closed SHORT {size:.2f} ({reason}) - Remaining short: {self.current_short_position:.2f}")

    def _update_position_levels(self, position_type, closed_size):
        """Update position tracking when positions are closed"""
        remaining_to_close = closed_size
        
        # Remove positions FIFO (first in, first out)
        for i in range(len(self.position_levels) - 1, -1, -1):
            if self.position_levels[i]['type'] == position_type and remaining_to_close > 0:
                position_size = self.position_levels[i]['size']
                
                if position_size <= remaining_to_close:
                    # Close entire position level
                    remaining_to_close -= position_size
                    self.position_levels.pop(i)
                else:
                    # Partially close position level
                    self.position_levels[i]['size'] -= remaining_to_close
                    remaining_to_close = 0
                    break

    def _update_performance_tracking(self):
        """Track recent performance for adaptive behavior (same as before)"""
        if hasattr(self, 'closed_trades') and len(self.closed_trades) > len(self.recent_trades):
            new_trades = self.closed_trades[len(self.recent_trades):]
            for trade in new_trades:
                self.recent_trades.append(trade.pl)
                if len(self.recent_trades) > 10:
                    self.recent_trades.pop(0)
            
            if new_trades and new_trades[-1].pl <= 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

    def _get_prediction(self):
        """Get model prediction with error handling (same as before)"""
        try:
            ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            feature_cols = [col for col in self.data.df.columns if col not in ohlcv_columns]
            
            if len(feature_cols) == 0:
                return None
            
            features_df = self.data.df[feature_cols].iloc[-60:]
            
            if features_df.isnull().sum().sum() > len(features_df) * 0.1:
                return None
                
            features_df = features_df.ffill().fillna(0)
            features = features_df.values

            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0)

                model_input_size = getattr(self.rnn_model, 'input_size', None)
                if model_input_size is None:
                    if hasattr(self.rnn_model, 'lstm'):
                        model_input_size = self.rnn_model.lstm.input_size
                    else:
                        model_input_size = features_tensor.shape[-1]
                
                if features_tensor.shape[-1] != model_input_size:
                    print(f"Feature size mismatch: expected {model_input_size}, got {features_tensor.shape[-1]}")
                    return None
                    
                prediction = self.rnn_model(features_tensor)
                probabilities = prediction.numpy()[0]
                
                action = np.argmax(probabilities)
                confidence = probabilities[action]
                
                return action, confidence
                
        except Exception as e:
            return None

#  Comprehensive backtesting workflow
def run_comprehensive_backtest(data, strategy_class, plt_file):
    """Execute full backtesting pipeline with performance analysis"""
    
    # Primary backtest
    bt = Backtest(data, strategy_class, cash=100000, commission=0.002)
    results = bt.run()
    
    if plt_file:
        bt.plot(filename=plt_file, open_browser=True)

    # Walk-forward analysis
    # wf_results = perform_walk_forward_analysis(data, strategy_class)
    
    # # Statistical significance testing
    # benchmark_returns = get_benchmark_returns(data.index[0], data.index[-1])
    # significance_tests = test_statistical_significance(
    #     results._trades['ReturnPct'], 
    #     benchmark_returns
    # )
    
    # # Risk analysis
    # risk_metrics = calculate_comprehensive_risk_metrics(results)
    
    return {
        'backtest_results': results,
        'summary_stats': {
            'total_return': results['Return [%]'],
            'sharpe_ratio': results['Sharpe Ratio'],
            'max_drawdown': results['Max. Drawdown [%]'],
            'win_rate': results['Win Rate [%]'] if 'Win Rate [%]' in results else 0,
            'profit_factor': results['Profit Factor'] if 'Profit Factor' in results else 1,
        }
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
            # self.feature_engine = AdvancedFeatureEngine()
            # self.portfolio_value_history = []
            # self.signals_history = []
            super().init()

            # Note: We do NOT call super().init() because we are
            # intentionally overriding the model loading behavior.

    return DynamicRNNStrategy