# ai/strategies/rnn_trading.py
from backtesting import Backtest, Strategy
import numpy as np
import pandas as pd
import torch

from ai.monitoring.performance_metrics import get_benchmark_returns, test_statistical_significance, calculate_comprehensive_risk_metrics
from ai.features.feature_engine import AdvancedFeatureEngine

class RNNTradingStrategy(Strategy):
    stop_loss_pct = 0.03  # 5% stop loss
    take_profit_pct = 0.06 # 10% take profit 2:1
    position_size = 0.70
    confidence_threshold = 0.45


    def init(self):
        self.portfolio_value_history = []
        self.signals_history = []
        self.last_prediction_time = None
        self.prediction_cache = None

        
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

        if self.position:
            # Exit logic for existing positions
            if self.position.is_long and action == 0 and confidence > 0.4:
                self.position.close()
            elif self.position.is_short and action == 2 and confidence > 0.4:
                self.position.close()
        else:
            # Entry logic when no position
            if action == 2 and confidence > self.confidence_threshold:  # UP signal
                sl = current_price * (1 - self.stop_loss_pct)
                tp = current_price * (1 + self.take_profit_pct)
                size = self._calculate_position_size(confidence)
                self.buy(size=size, sl=sl, tp=tp)
                
            elif action == 0 and confidence > self.confidence_threshold:  # DOWN signal
                sl = current_price * (1 + self.stop_loss_pct)
                tp = current_price * (1 - self.take_profit_pct)
                size = self._calculate_position_size(confidence)
                self.sell(size=size, sl=sl, tp=tp)


    def _get_prediction(self):
        """Get model prediction with error handling"""
        try:
            ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            feature_cols = [col for col in self.data.df.columns if col not in ohlcv_columns]
            
            if len(feature_cols) == 0:
                return None
            
            features_df = self.data.df[feature_cols].iloc[-60:]
            
            # Handle NaN values more gracefully
            if features_df.isnull().sum().sum() > len(features_df) * 0.1:  # >10% NaN
                return None
                
            # Forward fill NaN values
            features_df = features_df.ffill().fillna(0)
            features = features_df.values

            # Get RNN prediction
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                
                # Handle tensor shape issues
                if features_tensor.shape[-1] != self.rnn_model.input_size:
                    return None
                    
                prediction = self.rnn_model(features_tensor)
                probabilities = prediction.numpy()[0]  # [down, hold, up]
                
                # Get the class with highest probability
                action = np.argmax(probabilities)
                confidence = probabilities[action]
                
                return action, confidence
                
        except Exception as e:
            print(f"Error in model prediction: {e}")
            return None
    
    def _calculate_position_size(self, confidence):
        """Dynamic position sizing based on signal confidence"""
        base_size = self.position_size * min(confidence, 1.0)
        
        try:
            recent_returns = self.data.Close[-20:].pct_change().dropna()
            if len(recent_returns) > 5:
                volatility = recent_returns.std()
                vol_adjustment = max(0.3, 1.0 - volatility * 20)  # Reduce size in high vol
                base_size *= vol_adjustment
        except:
            pass
            
        return max(0.1, min(base_size, 0.95))

# Comprehensive backtesting workflow
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