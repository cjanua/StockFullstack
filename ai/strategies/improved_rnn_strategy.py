# ai/strategies/improved_rnn_strategy.py

import numpy as np
import torch
from backtesting import Strategy


class ImprovedRNNTradingStrategy(Strategy):
    # More conservative and adaptive parameters
    stop_loss_pct = 0.02  # Tighter stops
    take_profit_pct = 0.04 # 2:1 ratio
    position_size = 0.80
    confidence_threshold = 0.40  # Lower for more trades

    # New: Signal smoothing and confirmation
    signal_lookback = 3  # Require consistent signals

    def init(self):
        self.portfolio_value_history = []
        self.signals_history = []
        self.last_prediction_time = None
        self.prediction_cache = None
        self.signal_buffer = []  # Store recent signals for smoothing

    def next(self):
        if len(self.data) < 60:
            return

        current_time = self.data.index[-1]

        # Get prediction
        if self.last_prediction_time != current_time:
            self.prediction_cache = self._get_prediction()
            self.last_prediction_time = current_time

        if self.prediction_cache is None:
            return

        action, confidence = self.prediction_cache
        current_price = self.data.Close[-1]

        # Add to signal buffer for smoothing
        self.signal_buffer.append((action, confidence))
        if len(self.signal_buffer) > self.signal_lookback:
            self.signal_buffer.pop(0)

        # Enhanced signal processing
        smoothed_signal = self._process_signals()
        if smoothed_signal is None:
            return

        final_action, final_confidence = smoothed_signal

        # Position management with improved logic
        if self.position:
            self._manage_existing_position(final_action, final_confidence)
        else:
            self._enter_new_position(final_action, final_confidence, current_price)

    def _process_signals(self):
        """Enhanced signal processing with smoothing and confirmation."""
        if len(self.signal_buffer) < self.signal_lookback:
            return None

        # Get recent signals
        recent_actions = [s[0] for s in self.signal_buffer]
        [s[1] for s in self.signal_buffer]

        # Signal confirmation: require majority agreement
        action_counts = {0: 0, 1: 0, 2: 0}
        for action in recent_actions:
            action_counts[action] += 1

        # Find dominant signal
        dominant_action = max(action_counts, key=action_counts.get)
        agreement_ratio = action_counts[dominant_action] / len(recent_actions)

        # Require at least 60% agreement for strong signals
        if agreement_ratio < 0.6:
            return (1, 0.3)  # Default to hold with low confidence

        # Average confidence for the dominant signal
        dominant_confidences = [conf for act, conf in self.signal_buffer if act == dominant_action]
        avg_confidence = np.mean(dominant_confidences)

        # Boost confidence if there's strong agreement
        boosted_confidence = avg_confidence * (1 + agreement_ratio * 0.2)

        return (dominant_action, min(boosted_confidence, 1.0))

    def _manage_existing_position(self, action, confidence):
        """Improved position management with trend following."""
        # Exit on strong opposing signals
        if self.position.is_long and action == 0 and confidence > 0.5:
            self.position.close()
        elif self.position.is_short and action == 2 and confidence > 0.5:
            self.position.close()

        # Pyramid on strong confirming signals (optional)
        # if confidence > 0.7 and not hasattr(self, '_pyramided'):
        #     if self.position.is_long and action == 2:
        #         self.buy(size=0.1)  # Small additional position
        #         self._pyramided = True

    def _enter_new_position(self, action, confidence, current_price):
        """Enhanced entry logic with multiple confirmation filters."""
        # Market condition filter
        if not self._check_market_conditions():
            return

        # Volatility filter
        if not self._check_volatility():
            return

        # Time filter (avoid end of day trades)
        if not self._check_timing():
            return

        # Dynamic confidence threshold based on recent performance
        dynamic_threshold = self._get_dynamic_threshold()

        if confidence > dynamic_threshold:
            size = self._calculate_position_size(confidence)

            if action == 2:  # UP signal
                sl = current_price * (1 - self.stop_loss_pct)
                tp = current_price * (1 + self.take_profit_pct)
                self.buy(size=size, sl=sl, tp=tp)

            elif action == 0:  # DOWN signal
                sl = current_price * (1 + self.stop_loss_pct)
                tp = current_price * (1 - self.take_profit_pct)
                self.sell(size=size, sl=sl, tp=tp)

    def _check_market_conditions(self):
        """Check if market conditions are favorable for trading."""
        try:
            # Avoid trading in extreme volatility
            recent_returns = self.data.Close[-10:].pct_change().dropna()
            if len(recent_returns) > 0:
                volatility = recent_returns.std()
                if volatility > 0.05:  # >5% daily volatility
                    return False

            # Avoid trading after large moves
            last_move = abs(self.data.Close[-1] / self.data.Close[-2] - 1)
            if last_move > 0.03:  # >3% single day move
                return False

            return True
        except Exception as _:
            return True  # Default to allow trading

    def _check_volatility(self):
        """Volatility-based trading filter."""
        try:
            # Use ATR-like measure
            high_low = (self.data.High[-10:] - self.data.Low[-10:]) / self.data.Close[-10:]
            avg_range = high_low.mean()

            # Avoid trading in very low or very high volatility
            if avg_range < 0.005 or avg_range > 0.08:
                return False
            return True
        except Exception as _:
            return True

    def _check_timing(self):
        """Time-based trading filters."""
        # This is a placeholder - in live trading, you'd check actual time
        # For backtesting, we can use data frequency patterns
        try:
            # Avoid trading on volume spikes (might indicate news)
            recent_volume = self.data.Volume[-5:].mean()
            current_volume = self.data.Volume[-1]
            if current_volume > recent_volume * 3:
                return False
            return True
        except Exception as _:
            return True

    def _get_dynamic_threshold(self):
        """Adjust confidence threshold based on recent performance."""
        if len(self.closed_trades) < 5:
            return self.confidence_threshold

        # Get recent win rate
        recent_trades = self.closed_trades[-10:]
        recent_wins = sum(1 for trade in recent_trades if trade.pl > 0)
        recent_win_rate = recent_wins / len(recent_trades)

        # Adjust threshold: lower if doing well, higher if doing poorly
        if recent_win_rate > 0.6:
            return max(0.3, self.confidence_threshold - 0.1)
        elif recent_win_rate < 0.3:
            return min(0.7, self.confidence_threshold + 0.1)
        else:
            return self.confidence_threshold

    def _get_prediction(self):
        """Get model prediction with error handling (same as before)."""
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

                if features_tensor.shape[-1] != self.rnn_model.input_size:
                    return None

                prediction = self.rnn_model(features_tensor)
                probabilities = prediction.numpy()[0]

                action = np.argmax(probabilities)
                confidence = probabilities[action]

                return action, confidence

        except Exception:
            return None

    def _calculate_position_size(self, confidence):
        """Enhanced position sizing."""
        base_size = self.position_size * confidence

        # Volatility adjustment
        try:
            recent_returns = self.data.Close[-20:].pct_change().dropna()
            if len(recent_returns) > 5:
                volatility = recent_returns.std()
                vol_adjustment = max(0.3, 1.0 - volatility * 15)
                base_size *= vol_adjustment
        except Exception as e:
            print(e)
            pass

        # Performance adjustment
        if len(self.closed_trades) >= 5:
            recent_pnl = sum(trade.pl for trade in self.closed_trades[-5:])
            if recent_pnl > 0:
                base_size *= 1.1  # Increase size when doing well
            else:
                base_size *= 0.8  # Decrease size when struggling

        return max(0.1, min(base_size, 0.95))
