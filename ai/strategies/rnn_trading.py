# ai/strategies/rnn_trading.py

from collections import deque

import numpy as np
import pandas as pd
import talipp.indicators as ta
import torch
from backtesting import Backtest, Strategy
from talipp.ohlcv import OHLCV


def atr_indicator(open, high, low, close, volume, period=14):
    """Calculates ATR using the talipp library and formats it for backtesting.py.
    The self.I() method will pass the data columns as numpy arrays.
    """
    ohlcv_list = [OHLCV(o, h, l, c, v) for o, h, l, c, v in zip(open, high, low, close, volume, strict=False)]  # noqa: E741
    atr_output = ta.ATR(period, input_values=ohlcv_list)
    padding_size = len(close) - len(atr_output)
    atr_padded = np.r_[[np.nan] * padding_size, list(atr_output)]
    return atr_padded

class RNNTradingStrategy(Strategy):
    # Stop Loss
    stop_loss_pct = 0.015
    take_profit_pct = 0.045

    # Dynamic Position Sizing
    base_position_size = 0.20
    max_position_size = 0.35
    min_position_size = 0.08
    max_portfolio_heat = 0.12

    high_confidence_threshold = 0.65
    medium_confidence_threshold = 0.55
    low_confidence_threshold = 0.45

    max_consecutive_losses = 4
    max_daily_trades = 2
    daily_loss_limit = 0.08

    stop_loss_pct = 0.018  # Slightly tighter than 0.015
    take_profit_pct = 0.05

    volatility_lookback = 20
    regime_lookback = 60
    trend_confirmation_period = 5

    def init(self):
        super().init()

        self.signal_buffer = deque(maxlen=5)  # Signal smoothing

        self.regime_indicator = self._calculate_market_regime()
        self.volatility_indicator = self._calculate_volatility_regime()
        self.trend_indicator = self._calculate_trend_strength()
        # INIT ATR indicator
        self.data.ATR = self.I(
            atr_indicator,
            self.data.Open, self.data.High, self.data.Low, self.data.Close,
            self.data.Volume, period=14
        )

        # Performance tracking
        self.consecutive_losses = 0
        self.daily_trades = 0
        self.last_trade_date = None
        self.daily_pnl = 0.0
        self.starting_equity = None
        # self.win_count = 0
        # self.trade_count = 0
        # self.position_levels = []
        # self.risk_per_trade = {}

        self.recent_trades = deque(maxlen=10)
        self.recent_win_rate = 0.5

    def next(self):
        current_date = self.data.index[-1].date() if hasattr(self.data.index[-1], 'date') else self.data.index[-1]

        if self.last_trade_date != current_date:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_trade_date = current_date

        if self.starting_equity is None:
            self.starting_equity = self.equity

        self.daily_pnl = (self.equity - self.starting_equity) / self.starting_equity

        if self._should_halt_trading():
            return

        # Warm-up period
        if len(self.data) < max(60, self.volatility_lookback, self.regime_lookback):
            return

        prediction_data = self._get_prediction()
        if prediction_data is None:
            return

        action, confidence, _attention_weights = prediction_data
        self.signal_buffer.append((action, confidence))

        # Get smoothed signal
        smoothed_signal = self._get_smoothed_signal()
        if smoothed_signal is None:
            return

        final_action, final_confidence = smoothed_signal

        current_regime = self.regime_indicator[-1] if hasattr(self, 'regime_indicator') else 'normal'
        volatility_regime = self.volatility_indicator[-1] if hasattr(self, 'volatility_indicator') else 'normal'
        trend_strength = self.trend_indicator[-1] if len(self.trend_indicator) > 0 else 0

        # Adjust confidence based on regime
        adjusted_confidence = self._adjust_confidence(
            final_confidence, current_regime, volatility_regime, trend_strength
        )

        # Position management
        if self.position:
            self._manage_position(final_action, adjusted_confidence, trend_strength)
        else:
            self._enter_position(final_action, adjusted_confidence, current_regime, trend_strength)


    def _should_halt_trading(self):
        """Circuit breakers to prevent catastrophic losses."""
        if len(self.recent_trades) >= 5:
            recent_wins = sum(1 for trade in self.recent_trades if trade > 0)
            self.recent_win_rate = recent_wins / len(self.recent_trades)

        if self.recent_win_rate > 0.6:
            # Relax limits when doing well
            daily_limit = self.daily_loss_limit * 1.5
            consecutive_limit = self.max_consecutive_losses + 2
        elif self.recent_win_rate < 0.35:
            daily_limit = self.daily_loss_limit * 0.8      # 4.8% instead of 6%
            consecutive_limit = max(2, self.max_consecutive_losses - 1)  # 3 instead of 4
            # daily_trade_limit = self.max_daily_trades       # 3 (standard)
        else:
            daily_limit = self.daily_loss_limit             # Standard 6%
            consecutive_limit = self.max_consecutive_losses  # Standard 4
            # daily_trade_limit = self.max_daily_trades       # Standard 3


        # Daily loss limit
        if self.daily_pnl < -daily_limit:
            return True

        # Daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return True

        # Consecutive losses limit
        if self.consecutive_losses >= consecutive_limit:
            return True

        # Portfolio heat check
        current_risk = self._calculate_portfolio_risk()
        if current_risk > self.max_portfolio_heat:
            return True

        return False

    def _calculate_portfolio_risk(self):
        """Calculate current portfolio risk exposure."""
        if not self.position:
            return 0.0

        # Risk is the potential loss from stop-loss
        position_value = abs(self.position.size * self.data.Close[-1])
        stop_distance = self.stop_loss_pct
        risk_amount = position_value * stop_distance

        return risk_amount / self.equity


    def _get_prediction(self):
        """Get model prediction with error handling (same as before)."""
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
                if entropy > 0.85:  # High uncertainty
                    confidence *= 0.8

                return action, confidence, None

        except Exception:
            return None

    def _get_smoothed_signal(self):
        """Enhanced signal smoothing with weighted averaging."""
        if len(self.signal_buffer) < 3:
            return None

        buffer_size = len(self.signal_buffer)

        trend_strength = abs(self.trend_indicator[-1]) if len(self.trend_indicator) > 0 else 0

        if trend_strength > 0.5:  # Strong trend - favor recent signals
            if buffer_size == 3:
                weights = np.array([0.1, 0.3, 0.6])
            elif buffer_size == 4:
                weights = np.array([0.1, 0.2, 0.3, 0.4])
            elif buffer_size == 5:
                weights = np.array([0.05, 0.1, 0.15, 0.3, 0.4])
            elif buffer_size == 6:
                weights = np.array([0.05, 0.08, 0.12, 0.2, 0.25, 0.3])
            else:  # buffer_size == 7
                weights = np.array([0.03, 0.05, 0.08, 0.14, 0.2, 0.25, 0.25])
        else:  # Weak trend - balanced weighting
            if buffer_size == 3:
                weights = np.array([0.2, 0.3, 0.5])
            elif buffer_size == 4:
                weights = np.array([0.15, 0.2, 0.3, 0.35])
            elif buffer_size == 5:
                weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
            elif buffer_size == 6:
                weights = np.array([0.08, 0.12, 0.15, 0.2, 0.22, 0.23])
            else:  # buffer_size == 7
                weights = np.array([0.06, 0.08, 0.12, 0.15, 0.18, 0.2, 0.21])


        # Weight recent signals more heavily
        weights = weights / weights.sum()

        # SAFETY CHECK: Ensure weights array matches buffer size
        assert len(weights) == buffer_size, f"Weights size {len(weights)} != buffer size {buffer_size}"

        # Calculate weighted action and confidence
        actions = [s[0] for s in self.signal_buffer]
        confidences = [s[1] for s in self.signal_buffer]

        # Majority vote for action
        action_counts = {0: 0, 1: 0, 2: 0}
        for i, action in enumerate(actions):
            action_counts[action] += weights[i]

        dominant_action = max(action_counts, key=action_counts.get)
        consensus_strength = action_counts[dominant_action]

        if consensus_strength < 0.5:  # No strong consensus
            return 1, 0.3  # Default to hold

        # Weighted average confidence
        weighted_confidence = np.sum([w * c for w, c in zip(weights, confidences, strict=False)])

        if trend_strength > 0.3:
            if (dominant_action == 2 and trend_strength > 0) or (dominant_action == 0 and trend_strength < 0):
                weighted_confidence = min(weighted_confidence * 1.15, 1.0)

        # Consensus bonus (reduced threshold)
        if consensus_strength > 0.7:  # Reduced from 0.8
            weighted_confidence = min(weighted_confidence * 1.1, 1.0)

        return dominant_action, weighted_confidence


    def _calculate_market_regime(self):
        """Detect market regime using multiple indicators."""
        closes = pd.Series(self.data.Close)
        sma_10 = closes.rolling(10).mean()
        sma_30 = closes.rolling(30).mean()

        # Define regimes
        regimes = []
        for i in range(len(closes)):
            if i < 30:
                regimes.append('neutral')
            elif sma_10.iloc[i] > sma_30.iloc[i] * 1.008:
                regimes.append('bull')
            elif sma_10.iloc[i] < sma_30.iloc[i] * 0.992:
                regimes.append('bear')
            else:
                regimes.append('neutral')

        return regimes

    def _calculate_volatility_regime(self):
        """Detect volatility regime."""
        returns = pd.Series(self.data.Close).pct_change()
        volatility = returns.rolling(self.volatility_lookback).std()
        vol_ma = volatility.rolling(40).mean()

        # Calculate volatility percentiles
        # vol_percentile = volatility.rolling(252).rank(pct=True)

        regimes = []
        for i in range(len(volatility)):
            if pd.isna(volatility.iloc[i]) or pd.isna(vol_ma.iloc[i]):
                regimes.append('normal')
            elif volatility.iloc[i] > vol_ma.iloc[i] * 1.25:
                regimes.append('high')
            elif volatility.iloc[i] < vol_ma.iloc[i] * 0.85:
                regimes.append('low')
            else:
                regimes.append('normal')

        return regimes

    def _adjust_confidence(self, confidence, market_regime, volatility_regime, trend_strength):
        """Adjust confidence based on market conditions."""
        adjusted = confidence

        if market_regime == 'bull' and trend_strength > 0.15:
            adjusted *= 1.20
        elif market_regime == 'bear' and trend_strength < -0.15:
            adjusted *= 1.15

        if volatility_regime == 'high':
            adjusted *= 0.95
        elif volatility_regime == 'low':
            adjusted *= 1.15

        if self.recent_win_rate > 0.6:
            adjusted *= 1.15
        elif self.recent_win_rate < 0.35:
            adjusted *= 0.85

        if abs(trend_strength) > 0.3:
            adjusted *= 1.10

        return max(0.15, min(adjusted, 0.95))

    def _calculate_position_size(self, confidence, regime, trend_strength):
        """Kelly Criterion-inspired position sizing."""
        # Base position size on confidence
        kelly_fraction = max(0, (confidence - 0.45) * 3)  # Only size up above 60% confidence
        kelly_fraction = min(kelly_fraction, 0.3)  # Cap at a percent of portfolio

        # Weigh based on recent performance
        if self.recent_win_rate > 0.6:
            kelly_fraction *= 1.3  # Size up when winning
        elif self.recent_win_rate < 0.4:
            kelly_fraction *= 0.8  # Size down when losing

        # Trend strength adjustment
        if abs(trend_strength) > 0.25:
            kelly_fraction *= 1.15  # Size up in strong trends

        # Volatility adjustment
        try:
            recent_returns = pd.Series(self.data.Close[-20:]).pct_change().dropna()
            if len(recent_returns) > 5:
                volatility = recent_returns.std()
                vol_penalty = max(0.6, 1.0 - volatility * 12)  # Harsher volatility penalty
                kelly_fraction *= vol_penalty
        except Exception as _:
            kelly_fraction *= 0.9  # Conservative fallback

        # Consecutive losses penalty (gentler)
        if self.consecutive_losses > 0:
            kelly_fraction *= (0.8 ** min(self.consecutive_losses, 3))  # Gentler than 0.5

        # Calculate final position size
        position_size = self.base_position_size * (1 + kelly_fraction)
        return max(self.min_position_size, min(position_size, self.max_position_size))

    def _enter_position(self, action, confidence, regime, trend_strength):
        """Enhanced entry logic with regime filtering."""
        current_price = self.data.Close[-1]

        if action == 1:
            return

        if regime == 'bull' and trend_strength > 0.15:
            threshold = self.low_confidence_threshold * 0.9  # More aggressive in bull trends
        elif regime == 'bear' and trend_strength < -0.15:
            threshold = self.medium_confidence_threshold * 0.95  # Bear market short opportunities
        elif abs(trend_strength) > 0.3:  # Strong trend either direction
            threshold = self.medium_confidence_threshold
        else:
            threshold = self.high_confidence_threshold * 0.95  # Ranging market

        if confidence < threshold:
            return

        # Calculate position size
        position_size = self._calculate_position_size(confidence, regime, trend_strength)

        # Volatility-adjusted stops
        # volatility = pd.Series(self.data.Close[-self.volatility_lookback:]).pct_change().std()
        # vol_multiplier = max(1.0, min(2.0, volatility / 0.01))  # Scale stops with volatility

        estimated_risk = position_size * self.stop_loss_pct
        if estimated_risk > self.max_portfolio_heat * 1.1:
            position_size = self.max_portfolio_heat * 1.1 / self.stop_loss_pct

        atr = self.data.ATR[-1]
        if pd.isna(atr) or atr <= 0:
            atr = current_price * 0.02  # Fallback to 2% of price

        # IMPROVEMENT 12: Trend-adaptive stop/target ratios
        if abs(trend_strength) > 0.35:
            # Wider targets in strong trends
            stop_multiplier = 1.2  # Tighter stops
            target_multiplier = 4.5  # Wider targets (better than 3:1)
        else:
            # Standard ratios in ranging markets
            stop_multiplier = 1.4
            target_multiplier = 4.0

        min_risk_reward = 2.2

        # Place order based on signal
        if action == 2:  # UP signal
            stop_loss = current_price - (atr * stop_multiplier)
            take_profit = current_price + (atr * target_multiplier)

            # Ensure minimum risk/reward ratio
            risk = current_price - stop_loss
            reward = take_profit - current_price
            if reward / risk < min_risk_reward:  # Minimum 2.5:1 ratio
                return

            self.buy(size=position_size, sl=stop_loss, tp=take_profit)
            # self.position_levels.append({
            #     'price': current_price,
            #     'size': position_size,
            #     'confidence': confidence
            # })
            self.daily_trades += 1


        elif action == 0:  # DOWN signal
            stop_loss = current_price + (atr * stop_multiplier)
            take_profit = current_price - (atr * target_multiplier)

            risk = stop_loss - current_price
            reward = current_price - take_profit
            if reward / risk < min_risk_reward:
                return


            self.sell(size=position_size, sl=stop_loss, tp=take_profit)
            # self.position_levels.append({
            #     'price': current_price,
            #     'size': position_size,
            #     'confidence': action
            # })
            self.daily_trades += 1


    def _manage_position(self, action, confidence, trend_strength):
        """Enhanced position management with trailing stops and pyramiding."""
        # current_price = self.data.Close[-1]

        exit_threshold = 0.55 if abs(trend_strength) > 0.25 else 0.45

        # Check for exit signals
        if self.position.is_long and action == 0 and confidence > exit_threshold:
            self.position.close()
            self._reset_position_tracking()
        elif self.position.is_short and action == 2 and confidence > exit_threshold:
            self.position.close()
            self._reset_position_tracking()

    def _update_performance_tracking(self):
        """Update performance metrics after trade close."""
        if hasattr(self, 'trades') and len(self.trades) > 0:
            last_trade = self.trades[-1]

            self.recent_trades.append(last_trade.pl)

            if last_trade.pl > 0:
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1

    def _reset_position_tracking(self):
        """Reset position tracking variables."""
        self.position_levels = []
        self.average_entry_price = 0

        # Update performance tracking
        if hasattr(self, 'trades') and len(self.trades) > 0:
            last_trade = self.trades[-1]
            if last_trade.pl > 0:
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1

    def _calculate_trend_strength(self):
        """Calculate trend strength indicator."""
        closes = pd.Series(self.data.Close)

        # Multiple timeframe trend analysis
        ema_12 = closes.ewm(span=12).mean()
        ema_26 = closes.ewm(span=26).mean()
        ema_50 = closes.ewm(span=50).mean()

        # Trend strength based on EMA alignment and slope
        trend_values = []
        for i in range(len(closes)):
            if i < 50:
                trend_values.append(0)
            else:
                # EMA alignment score
                alignment_score = 0
                if ema_12.iloc[i] > ema_26.iloc[i]:
                    alignment_score += 1
                if ema_26.iloc[i] > ema_50.iloc[i]:
                    alignment_score += 1
                if ema_12.iloc[i] > ema_50.iloc[i]:
                    alignment_score += 1

                # Slope strength (normalized)
                slope_12 = (ema_12.iloc[i] - ema_12.iloc[i-5]) / ema_12.iloc[i-5]
                slope_26 = (ema_26.iloc[i] - ema_26.iloc[i-5]) / ema_26.iloc[i-5]

                # Combine alignment and slope
                if alignment_score == 3:  # All EMAs aligned bullish
                    trend_strength = min(1.0, max(0.2, (slope_12 + slope_26) * 50))
                elif alignment_score == 0:  # All EMAs aligned bearish
                    trend_strength = max(-1.0, min(-0.2, (slope_12 + slope_26) * 50))
                else:  # Mixed signals
                    trend_strength = (slope_12 + slope_26) * 25  # Weaker signal

                trend_values.append(trend_strength)

        return trend_values


#  Comprehensive backtesting workflow
def run_comprehensive_backtest(data, strategy_class, plt_file):
    """Execute full backtesting pipeline with performance analysis."""
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
                                 train_window=180, test_window=45):
    """Walk-forward analysis with parameter optimization."""
    results = []

    for i in range(train_window, len(data) - test_window, test_window):
        # Training period
        train_data = data.iloc[i-train_window:i].copy()
        train_data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)

        # Parameter optimization on training data
        bt_train = Backtest(train_data, strategy_class)
        optimized_params = bt_train.optimize(
            signal_threshold=range(60, 81, 5),
            position_size=np.arange(0.7, 1.0, 0.1),
            constraint=lambda p: p.signal_threshold < 80
        )

        # Out-of-sample testing
        test_data = data.iloc[i:i+test_window].copy()
        test_data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        bt_test = Backtest(test_data, strategy_class)
        test_results = bt_test.run(**optimized_params)

        results.append({
            'period_start': test_data.index[0],
            'period_end': test_data.index[-1],
            'return': test_results['Return [%]'],
            'sharpe': test_results['Sharpe Ratio'],
            'max_drawdown': test_results['Max. Drawdown [%]'],
            'win_rate': len(test_results._trades[test_results._trades['ReturnPct'] > 0]) / len(test_results._trades) if len(test_results._trades) > 0 else 0
        })

    return pd.DataFrame(results)


def create_rnn_strategy_class(trained_model):
    """Factory function to create a new Strategy class with a specific pre-trained model.

    The backtesting library requires a class, so we can't just pass the model to an
    instance. Instead, we dynamically create a new class that has the model
    "baked in".
    """

    class DynamicRNNStrategy(RNNTradingStrategy):
        """A dynamically created strategy class that uses a pre-loaded model."""

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