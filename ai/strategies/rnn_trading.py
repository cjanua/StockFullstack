# ai/strategies/rnn_trading.py
from collections import deque
from backtesting import Backtest, Strategy
import numpy as np
import pandas as pd
import torch

import talipp.indicators as ta
from talipp.ohlcv import OHLCV


from ai.monitoring.performance_metrics import get_benchmark_returns, test_statistical_significance, calculate_comprehensive_risk_metrics
from ai.features.feature_engine import AdvancedFeatureEngine

def atr_indicator(open, high, low, close, volume, period=14):
    """
    Calculates ATR using the talipp library and formats it for backtesting.py.
    The self.I() method will pass the data columns as numpy arrays.
    """
    ohlcv_list = [OHLCV(o, h, l, c, v) for o, h, l, c, v in zip(open, high, low, close, volume)]
    atr_output = ta.ATR(period, input_values=ohlcv_list)
    
    # Pad the start with NaNs so the output array has the same length as the input
    padding_size = len(close) - len(atr_output)
    atr_padded = np.r_[[np.nan] * padding_size, [val for val in atr_output]]
    return atr_padded

class RNNTradingStrategy(Strategy):
    # Stop Loss
    stop_loss_pct = 0.02
    take_profit_pct = 0.06

    # Dynamic Position Sizing
    base_position_size = 0.5
    max_position_size = 0.95
    min_position_size = 0.1

    high_confidence_threshold = 0.65
    medium_confidence_threshold = 0.45
    low_confidence_threshold = 0.35

    max_consecutive_losses = 3
    volatility_lookback = 20

    regime_lookback = 60


    def init(self):
        super().init()
        self.signal_buffer = deque(maxlen=3)  # Signal smoothing
        self.regime_indicator = self._calculate_market_regime()
        self.volatility_indicator = self._calculate_volatility_regime()

        # INIT ATR indicator
        self.data.ATR = self.I(
            atr_indicator,
            self.data.Open,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.data.Volume,
            period=14
        )
        
        # Performance tracking
        self.consecutive_losses = 0
        self.trade_count = 0
        self.win_count = 0
        
        # Position tracking for pyramiding
        self.position_levels = []
        self.average_entry_price = 0
        
    def next(self):
        # Warm-up
        if len(self.data) < max(60, self.volatility_lookback, self.regime_lookback):
            return
        
        current_time = self.data.index[-1]

        prediction_data = self._get_prediction()
        if prediction_data is None:
            return


        action, confidence, attention_weights = prediction_data

        self.signal_buffer.append((action, confidence))
        
        # Get smoothed signal
        smoothed_signal = self._get_smoothed_signal()
        if smoothed_signal is None:
            return
            
        final_action, final_confidence = smoothed_signal

        current_regime = self.regime_indicator[-1] if hasattr(self, 'regime_indicator') else 'normal'
        volatility_regime = self.volatility_indicator[-1] if hasattr(self, 'volatility_indicator') else 'normal'
        
        # Adjust confidence based on regime
        adjusted_confidence = self._adjust_confidence_by_regime(
            final_confidence, current_regime, volatility_regime
        )
        
        # Position management
        if self.position:
            self._manage_position_enhanced(final_action, adjusted_confidence, current_regime)
        else:
            self._enter_position_enhanced(final_action, adjusted_confidence, current_regime)


    def _get_prediction(self):
        """Get model prediction with error handling (same as before)"""
        try:
            # Extract features (excluding OHLCV)
            ohlcv_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            feature_cols = [col for col in self.data.df.columns if col not in ohlcv_columns]
            
            if len(feature_cols) == 0:
                return None
            
            # Get sequence of features
            features_df = self.data.df[feature_cols].iloc[-60:]
            features = features_df.values
            
            # Prepare input
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                
                # For attention models, we might get attention weights too
                if hasattr(self.rnn_model, 'attention'):
                    prediction = self.rnn_model(features_tensor)
                    attention_weights = None  # Model would need to return these
                else:
                    prediction = self.rnn_model(features_tensor)
                    attention_weights = None
                
                probabilities = prediction.numpy()[0]
                action = np.argmax(probabilities)
                confidence = probabilities[action]
                
                return action, confidence, attention_weights
                
        except Exception as e:
            return None

    def _get_smoothed_signal(self):
        """Enhanced signal smoothing with weighted averaging"""
        if len(self.signal_buffer) < 2:
            return None
        
        # Weight recent signals more heavily
        weights = np.array([0.2, 0.3, 0.5])[-len(self.signal_buffer):]
        weights = weights / weights.sum()
        
        # Calculate weighted action and confidence
        actions = [s[0] for s in self.signal_buffer]
        confidences = [s[1] for s in self.signal_buffer]
        
        # Majority vote for action
        action_counts = {0: 0, 1: 0, 2: 0}
        for i, action in enumerate(actions):
            action_counts[action] += weights[i]
        
        smoothed_action = max(action_counts, key=action_counts.get)
        
        # Weighted average confidence
        smoothed_confidence = np.sum(weights * confidences)
        
        # Boost confidence if all signals agree
        if len(set(actions)) == 1:
            smoothed_confidence = min(smoothed_confidence * 1.2, 1.0)
        
        return smoothed_action, smoothed_confidence
    
    def _calculate_market_regime(self):
        """Detect market regime using multiple indicators"""
        closes = pd.Series(self.data.Close)
        
        # Trend regime
        sma_50 = closes.rolling(50).mean()
        sma_200 = closes.rolling(200).mean()
        
        # Define regimes
        regimes = []
        for i in range(len(closes)):
            if i < 200:
                regimes.append('unknown')
            elif sma_50.iloc[i] > sma_200.iloc[i] and closes.iloc[i] > sma_50.iloc[i]:
                regimes.append('bull')
            elif sma_50.iloc[i] < sma_200.iloc[i] and closes.iloc[i] < sma_50.iloc[i]:
                regimes.append('bear')
            else:
                regimes.append('neutral')
        
        return regimes
    
    def _calculate_volatility_regime(self):
        """Detect volatility regime"""
        returns = pd.Series(self.data.Close).pct_change()
        volatility = returns.rolling(self.volatility_lookback).std()
        
        # Calculate volatility percentiles
        vol_percentile = volatility.rolling(252).rank(pct=True)
        
        regimes = []
        for i in range(len(vol_percentile)):
            if pd.isna(vol_percentile.iloc[i]):
                regimes.append('normal')
            elif vol_percentile.iloc[i] < 0.33:
                regimes.append('low')
            elif vol_percentile.iloc[i] > 0.67:
                regimes.append('high')
            else:
                regimes.append('normal')
        
        return regimes
    
    def _adjust_confidence_by_regime(self, confidence, market_regime, volatility_regime):
        """Adjust confidence based on market conditions"""
        adjusted = confidence
        
        # Market regime adjustments
        if market_regime == 'bull':
            # More confident in upward predictions during bull market
            adjusted *= 1.1
        elif market_regime == 'bear':
            # Less confident overall during bear market
            adjusted *= 0.9
        
        # Volatility adjustments
        if volatility_regime == 'high':
            # Require higher confidence in high volatility
            adjusted *= 0.85
        elif volatility_regime == 'low':
            # Can be more confident in low volatility
            adjusted *= 1.05
        
        # Cap at reasonable bounds
        return max(0.1, min(adjusted, 0.95))
    
    def _calculate_dynamic_position_size(self, confidence, regime):
        """Kelly Criterion-inspired position sizing"""
        # Base position size on confidence
        kelly_fraction = (confidence - 0.5) * 2  # Scale to 0-1
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Adjust for regime
        if regime == 'high' or self.consecutive_losses >= self.max_consecutive_losses:
            # Reduce position size in high volatility or after losses
            kelly_fraction *= 0.5
        
        # Calculate final position size
        position_size = self.base_position_size + (self.max_position_size - self.base_position_size) * kelly_fraction
        
        # Ensure within bounds
        return max(self.min_position_size, min(position_size, self.max_position_size))
    
    def _enter_position_enhanced(self, action, confidence, regime):
        """Enhanced entry logic with regime filtering"""
        current_price = self.data.Close[-1]
        
        # Skip trades in unfavorable regimes with low confidence
        if regime == 'high' and confidence < self.high_confidence_threshold:
            return
        
        # Dynamic confidence threshold
        if regime == 'bear':
            threshold = self.high_confidence_threshold
        elif regime == 'bull':
            threshold = self.low_confidence_threshold
        else:
            threshold = self.medium_confidence_threshold
        
        if confidence < threshold:
            return
        
        # Calculate position size
        position_size = self._calculate_dynamic_position_size(confidence, regime)
        
        # Volatility-adjusted stops
        # volatility = pd.Series(self.data.Close[-self.volatility_lookback:]).pct_change().std()
        # vol_multiplier = max(1.0, min(2.0, volatility / 0.01))  # Scale stops with volatility
        
        atr = self.data.ATR[-1]
        
        # ATR multipliers for stop-loss and take-profit
        stop_loss_atr_multiplier = 2.0  # Example: 2 * ATR for stop-loss
        take_profit_atr_multiplier = 3.0  # Example: 3 * ATR for take-prof

        # Place order based on signal
        if action == 2:  # UP signal
            stop_loss = current_price - (atr * stop_loss_atr_multiplier)
            take_profit = current_price + (atr * take_profit_atr_multiplier)
            
            self.buy(size=position_size, sl=stop_loss, tp=take_profit)
            self.position_levels.append({
                'price': current_price,
                'size': position_size,
                'confidence': confidence
            })

        elif action == 0:  # DOWN signal
            stop_loss = current_price + (atr * stop_loss_atr_multiplier)
            take_profit = current_price - (atr * take_profit_atr_multiplier)
            
            self.sell(size=position_size, sl=stop_loss, tp=take_profit)
            self.position_levels.append({
                'price': current_price,
                'size': position_size,
                'confidence': action
            })

    
    def _manage_position_enhanced(self, action, confidence, regime):
        """Enhanced position management with trailing stops and pyramiding"""
        current_price = self.data.Close[-1]
        
        # Check for exit signals
        if self.position.is_long and action == 0 and confidence > 0.5:
            self.position.close()
            self._reset_position_tracking()
        elif self.position.is_short and action == 2 and confidence > 0.5:
            self.position.close()
            self._reset_position_tracking()
        
        # Pyramiding logic (add to winners)
        elif confidence > self.high_confidence_threshold and len(self.position_levels) < 3:
            # Only pyramid if position is profitable
            entry_price = self.position_levels[0]['price'] if self.position_levels else self.position.avg_fill_price
            
            if self.position.is_long and current_price > entry_price * 1.02:
                # Add to long position
                additional_size = self._calculate_dynamic_position_size(confidence, regime) * 0.5
                self.buy(size=additional_size)
                self.position_levels.append({
                    'price': current_price,
                    'size': additional_size,
                    'confidence': confidence
                })
                
            elif self.position.is_short and current_price < entry_price * 0.98:
                # Add to short position
                additional_size = self._calculate_dynamic_position_size(confidence, regime) * 0.5
                self.sell(size=additional_size)
                self.position_levels.append({
                    'price': current_price,
                    'size': additional_size,
                    'confidence': confidence
                })
    
    def _reset_position_tracking(self):
        """Reset position tracking variables"""
        self.position_levels = []
        self.average_entry_price = 0
        
        # Update performance tracking
        if hasattr(self, 'trades') and len(self.trades) > 0:
            last_trade = self.trades[-1]
            if last_trade.pl > 0:
                self.win_count += 1
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1

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