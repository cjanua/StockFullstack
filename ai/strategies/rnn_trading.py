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
    stop_loss_pct = 0.015
    take_profit_pct = 0.045

    # Dynamic Position Sizing
    base_position_size = 0.15
    max_position_size = 0.25
    min_position_size = 0.05
    max_portfolio_heat = 0.06

    high_confidence_threshold = 0.75
    medium_confidence_threshold = 0.65
    low_confidence_threshold = 0.55

    max_consecutive_losses = 2
    max_daily_trades = 1
    daily_loss_limit = 0.03

    volatility_lookback = 20
    regime_lookback = 60


    def init(self):
        super().init()
        self.signal_buffer = deque(maxlen=5)  # Signal smoothing
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
        self.daily_trades = 0
        self.last_trade_date = None
        self.daily_pnl = 0.0
        self.starting_equity = None
        
        # Position tracking for pyramiding
        self.position_levels = []
        self.risk_per_trade = {}
        
    def next(self):
        current_date = self.data.index[-1].date() if hasattr(self.data.index[-1], 'date') else self.data.index[-1]

        if self.last_trade_date != current_date:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_trade_date = current_date

        if self.starting_equity is None:
            self.starting_equity = self.equity

        if self._should_halt_trading():
            return

        # Warm-up period
        if len(self.data) < max(60, self.volatility_lookback, self.regime_lookback):
            return
    
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
            self._manage_position(final_action, adjusted_confidence)
        else:
            self._enter_position(final_action, adjusted_confidence, current_regime)


    def _should_halt_trading(self):
        """Circuit breakers to prevent catastrophic losses"""
        
        # Daily loss limit
        if self.daily_pnl < -self.daily_loss_limit:
            return True
            
        # Daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return True
            
        # Consecutive losses limit
        if self.consecutive_losses >= self.max_consecutive_losses:
            return True
            
        # Portfolio heat check
        current_risk = self._calculate_portfolio_risk()
        if current_risk > self.max_portfolio_heat:
            return True
            
        return False

    def _calculate_portfolio_risk(self):
        """Calculate current portfolio risk exposure"""
        if not self.position:
            return 0.0
            
        # Risk is the potential loss from stop-loss
        position_value = abs(self.position.size * self.data.Close[-1])
        stop_distance = self.stop_loss_pct
        risk_amount = position_value * stop_distance
        
        return risk_amount / self.equity


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
            if features_df.isnull().sum().sum() > len(features_df) * 0.05:  # Max 5% NaN
                return None
            features_df = features_df.ffill().fillna(0)
            features = features_df.values
            
            # Prepare input
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                if features_tensor.shape[-1] != self.rnn_model.input_size:
                    return None
                prediction = self.rnn_model(features_tensor)
                probabilities = prediction.numpy()[0]

                action = np.argmax(probabilities)
                confidence = probabilities[action]
                
                # Penalize low-conviction predictions
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
                if entropy > 0.8:  # High uncertainty
                    confidence *= 0.7

                return action, confidence, None
                
        except Exception as e:
            return None

    def _get_smoothed_signal(self):
        """Enhanced signal smoothing with weighted averaging"""
        if len(self.signal_buffer) < 2:
            return None
        buffer_size = len(self.signal_buffer)
        if buffer_size == 2:
            weights = np.array([0.4, 0.6])
        elif buffer_size == 3:
            weights = np.array([0.2, 0.3, 0.5])
        elif buffer_size == 4:
            weights = np.array([0.1, 0.2, 0.3, 0.4])
        else:  # buffer_size == 5 (maxlen)
            weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])

        
        # Weight recent signals more heavily
        weights = weights / weights.sum()
        
        # Calculate weighted action and confidence
        actions = [s[0] for s in self.signal_buffer]
        confidences = [s[1] for s in self.signal_buffer]
        
        # Majority vote for action
        action_counts = {0: 0, 1: 0, 2: 0}
        for i, action in enumerate(actions):
            action_counts[action] += weights[i]
        
        dominant_action = max(action_counts, key=action_counts.get)
        consensus_strength = action_counts[dominant_action]
        if consensus_strength < 0.6:  # No strong consensus
            return 1, 0.3  # Default to hold

        # Weighted average confidence
        weighted_confidence = np.sum(weights * confidences)
        
        # Boost confidence if all signals agree
        if consensus_strength > 0.8:
            weighted_confidence = min(weighted_confidence * 1.2, 1.0)
        
        return dominant_action, weighted_confidence
    
    def _calculate_market_regime(self):
        """Detect market regime using multiple indicators"""
        closes = pd.Series(self.data.Close)
        sma_20 = closes.rolling(20).mean()
        sma_50 = closes.rolling(50).mean()
        
        # Define regimes
        regimes = []
        for i in range(len(closes)):
            if i < 50:
                regimes.append('neutral')
            elif sma_20.iloc[i] > sma_50.iloc[i] and closes.iloc[i] > sma_20.iloc[i]:
                regimes.append('bull')
            elif sma_20.iloc[i] < sma_50.iloc[i] and closes.iloc[i] < sma_20.iloc[i]:
                regimes.append('bear')
            else:
                regimes.append('neutral')
        
        return regimes
    
    def _calculate_volatility_regime(self):
        """Detect volatility regime"""
        returns = pd.Series(self.data.Close).pct_change()
        volatility = returns.rolling(self.volatility_lookback).std()
        vol_ma = volatility.rolling(60).mean()

        # Calculate volatility percentiles
        # vol_percentile = volatility.rolling(252).rank(pct=True)
        
        regimes = []
        for i in range(len(volatility)):
            if pd.isna(volatility.iloc[i]) or pd.isna(vol_ma.iloc[i]):
                regimes.append('normal')
            elif volatility.iloc[i] > vol_ma.iloc[i] * 1.5:
                regimes.append('high')
            elif volatility.iloc[i] < vol_ma.iloc[i] * 0.7:
                regimes.append('low')
            else:
                regimes.append('normal')
        
        return regimes
    
    def _adjust_confidence_by_regime(self, confidence, market_regime, volatility_regime):
        """Adjust confidence based on market conditions"""
        adjusted = confidence

        # Volatility adjustments
        if volatility_regime == 'high':
            # Require higher confidence in high volatility
            adjusted *= 0.7
        elif market_regime == 'bear':
            # Less confident overall during bear market
            adjusted *= 0.8
        elif market_regime == 'neutral':
            # More confident in upward predictions during bull market
            adjusted *= 0.9
        
        
        if market_regime == 'bull' and volatility_regime == 'low':
            adjusted *= 1.05

        # Cap at reasonable bounds
        return max(0.1, min(adjusted, 0.95))
    
    def _calculate_dynamic_position_size(self, confidence, regime):
        """Kelly Criterion-inspired position sizing"""
        # Base position size on confidence
        kelly_fraction = max(0, (confidence - 0.6) * 2)  # Only size up above 60% confidence
        kelly_fraction = min(kelly_fraction, 0.15)  # Cap at a percent of portfolio
        
        # Heavy penalty for recent losses
        if self.consecutive_losses > 0:
            kelly_fraction *= (0.5 ** self.consecutive_losses)

        # Volatility adjustment
        try:
            recent_returns = pd.Series(self.data.Close[-20:]).pct_change().dropna()
            if len(recent_returns) > 5:
                volatility = recent_returns.std()
                vol_penalty = max(0.3, 1.0 - volatility * 20)  # Harsher volatility penalty
                kelly_fraction *= vol_penalty
        except:
            kelly_fraction *= 0.5  # Conservative fallback

        
        # Calculate final position size
        position_size = self.base_position_size * (1 + kelly_fraction)
        return max(self.min_position_size, min(position_size, self.max_position_size))
    
    def _enter_position(self, action, confidence, regime):
        """Enhanced entry logic with regime filtering"""
        current_price = self.data.Close[-1]
        if action == 1:
            return
        
        if regime == 'bear' or self.consecutive_losses > 0:
            threshold = self.high_confidence_threshold
        elif regime == 'bull':
            threshold = self.medium_confidence_threshold
        else:
            threshold = self.high_confidence_threshold
        
        if confidence < threshold:
            return

        
        # Calculate position size
        position_size = self._calculate_dynamic_position_size(confidence, regime)
        
        # Volatility-adjusted stops
        # volatility = pd.Series(self.data.Close[-self.volatility_lookback:]).pct_change().std()
        # vol_multiplier = max(1.0, min(2.0, volatility / 0.01))  # Scale stops with volatility
        
        estimated_risk = position_size * self.stop_loss_pct
        if estimated_risk > self.max_portfolio_heat:
            position_size = self.max_portfolio_heat / self.stop_loss_pct

        atr = self.data.ATR[-1]
        if pd.isna(atr) or atr <= 0:
            atr = current_price * 0.02  # Fallback to 2% of price
        
        # ATR multipliers for stop-loss and take-profit
        stop_loss_atr_multiplier = 1.5  # Example: 2 * ATR for stop-loss
        take_profit_atr_multiplier = 4.5  # Example: 3 * ATR for take-prof

        # Place order based on signal
        if action == 2:  # UP signal
            stop_loss = current_price - (atr * stop_loss_atr_multiplier)
            take_profit = current_price + (atr * take_profit_atr_multiplier)
            
            # Ensure minimum risk/reward ratio
            risk = current_price - stop_loss
            reward = take_profit - current_price
            if reward / risk < 2.5:  # Minimum 2.5:1 ratio
                return

            self.buy(size=position_size, sl=stop_loss, tp=take_profit)
            # self.position_levels.append({
            #     'price': current_price,
            #     'size': position_size,
            #     'confidence': confidence
            # })
            self.daily_trades += 1


        elif action == 0:  # DOWN signal
            stop_loss = current_price + (atr * stop_loss_atr_multiplier)
            take_profit = current_price - (atr * take_profit_atr_multiplier)
            
            risk = stop_loss - current_price
            reward = current_price - take_profit
            if reward / risk < 2.5:
                return


            self.sell(size=position_size, sl=stop_loss, tp=take_profit)
            # self.position_levels.append({
            #     'price': current_price,
            #     'size': position_size,
            #     'confidence': action
            # })
            self.daily_trades += 1

    
    def _manage_position(self, action, confidence):
        """Enhanced position management with trailing stops and pyramiding"""
        # current_price = self.data.Close[-1]
        
        # Check for exit signals
        if self.position.is_long and action == 0 and confidence > 0.5:
            self.position.close()
            self._reset_position_tracking()
        elif self.position.is_short and action == 2 and confidence > 0.5:
            self.position.close()
            self._reset_position_tracking()
        
        # # Pyramiding logic (add to winners)
        # elif confidence > self.high_confidence_threshold and len(self.position_levels) < 3:
        #     # Only pyramid if position is profitable
        #     entry_price = self.position_levels[0]['price'] if self.position_levels else self.position.avg_fill_price
            
        #     if self.position.is_long and current_price > entry_price * 1.02:
        #         # Add to long position
        #         additional_size = self._calculate_dynamic_position_size(confidence, regime) * 0.5
        #         self.buy(size=additional_size)
        #         self.position_levels.append({
        #             'price': current_price,
        #             'size': additional_size,
        #             'confidence': confidence
        #         })
                
        #     elif self.position.is_short and current_price < entry_price * 0.98:
        #         # Add to short position
        #         additional_size = self._calculate_dynamic_position_size(confidence, regime) * 0.5
        #         self.sell(size=additional_size)
        #         self.position_levels.append({
        #             'price': current_price,
        #             'size': additional_size,
        #             'confidence': confidence
        #         })

    def _update_performance_tracking(self):
        """Update performance metrics after trade close"""
        if hasattr(self, 'trades') and len(self.trades) > 0:
            last_trade = self.trades[-1]
            if last_trade.pl > 0:
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
    
    def _reset_position_tracking(self):
        """Reset position tracking variables"""
        self.position_levels = []
        self.average_entry_price = 0
        
        # Update performance tracking
        if hasattr(self, 'trades') and len(self.trades) > 0:
            last_trade = self.trades[-1]
            if last_trade.pl > 0:
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