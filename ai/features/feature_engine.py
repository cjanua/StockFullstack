# ai/features/feature_engine.py

from datetime import datetime
from pathlib import Path
import time
from talipp.indicators import SMA, EMA, RSI, MACD, BB, ATR, OBV, ADX, Stoch, CCI
from talipp.ohlcv import OHLCV
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import RobustScaler

from ai.config.settings import config

class AdvancedFeatureEngine:
    def __init__(self):
        self.scalers = {}
        self.lookback_window = 60
        self.cache_dir = Path(config.CACHE_DIR) / 'yahoo'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _chaikin_money_flow(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 20
    ) -> pd.Series:
        mfv = ((close - low) - (high - close)) / (high - low) * volume
        mfv = mfv.fillna(0)
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        return cmf
    
    def _accumulation_distribution_line(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)
        mfv = mfm * volume
        return mfv.cumsum()
    
    def _create_ohlcv_list(self, data):
        """Helper to create OHLCV list for talipp indicators"""
        from talipp.ohlcv import OHLCV
        return [OHLCV(row.open, row.high, row.low, row.close, row.volume) for row in data.itertuples()]


    def create_comprehensive_features(
        self,
        data: pd.DataFrame,
        symbol: str,
        market_context_data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Generate multi-timeframe feature set for RNN"""
        if data.index.has_duplicates:
            print("Warning: Data contains duplicate timestamps, dropping duplicates.")
            data = data.loc[~data.index.duplicated(keep='first')]
        
        agg_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        data = data.resample('D').agg(agg_rules).dropna()
        if data.empty:
            print("Warning: Not enough data to create hourly features.")
            return pd.DataFrame()

        
        ohlcv_list = self._create_ohlcv_list(data)
        features = {}
        
        #  1. Technical Indicators
        features.update(self.calculate_technical_indicators(data, ohlcv_list))

        #  2, Multi-timeframe features
        features.update(self.create_multitimeframe_features(data))

        # 3. Advanced Volatility Features (GARCH-style)
        features.update(self.calculate_advanced_volatility_features(data))

        # 4. Cross-Asset Correlations (31% annualized excess returns)
        if market_context_data is not None:
            features.update(self.calculate_cross_asset_features(data, market_context_data))

        # 5. Cyclical Temporal Encoding (10-15% error reduction)
        features.update(self.calculate_cyclical_temporal_features(data))

        # 6. Market Microstructure Features
        features.update(self.calculate_microstructure_features(data))
        
        # 7. Price Action Patterns
        features.update(self.calculate_price_action_features(data))

        # 8. Volume Profile
        features.update(self.calculate_volume_profile_features(data))

        # 9. Regime Detection Features
        features.update(self.detect_market_regime(data, ohlcv_list))

        # 10. 
        # features.update(self.calculate_asset_specific_features(data, symbol))
        
        
        if market_context_data is not None:
            features.update(self.calculate_market_context_features(data, market_context_data))
        
        features.update(self.calculate_risk_features(data))
        features['close'] = data['close']
        
        return self.normalize_and_structure_features(features)
    
    def calculate_technical_indicators(self, data, ohlcv_list: list[OHLCV]):
        """Comprehensive technical indicator calculation"""
        indicators = {}
        close, high, low, volume = data['close'], data['high'], data['low'], data['volume']
        close_list = close.tolist()


        # EMA Ratios for Golden/Deatch Cross Detection
        ema_50 = pd.Series(EMA(50, close_list), index=data.index)
        ema_200 = pd.Series(EMA(200, close_list), index=data.index)
        indicators['ema_ratio_50_200'] = ema_50 / (ema_200 + 1e-8)
        # Multiple EMAs
        for period in [9, 12, 26, 50]:
            indicators[f'ema_{period}'] = pd.Series(EMA(period, close_list), index=data.index)

        # Price position relative to key MAs
        sma_20 = pd.Series(SMA(20, close_list), index=data.index)
        indicators['price_to_sma20'] = close / (sma_20 + 1e-8)


        indicators['rsi_9'] = pd.Series(RSI(9, close_list), index=data.index)
        indicators['rsi_14'] = pd.Series(RSI(14, close_list), index=data.index)
        indicators['rsi_21'] = pd.Series(RSI(21, close_list), index=data.index)
        # RSI divergence
        indicators['rsi_divergence'] = indicators['rsi_14'].diff() * close.pct_change()
        
        
        macd_output = MACD(fast_period=12, slow_period=26, signal_period=9, input_values=close_list)
        indicators['macd_line'] = pd.Series([val.macd if val else None for val in macd_output], index=data.index)
        indicators['macd_signal'] = pd.Series([val.signal if val else None for val in macd_output], index=data.index)
        indicators['macd_histogram'] = pd.Series([val.histogram if val else None for val in macd_output], index=data.index)
        # MACD zero-cross indicator
        indicators['macd_above_zero'] = (indicators['macd_line'] > 0).astype(int)
        indicators['macd_signal_cross'] = (indicators['macd_line'] > indicators['macd_signal']).astype(int)
        

        bb_output = BB(period=20, std_dev_mult=2.0, input_values=close_list)
        indicators['bb_upper'] = pd.Series([val.ub if val else None for val in bb_output], index=data.index)
        indicators['bb_middle'] = pd.Series([val.cb if val else None for val in bb_output], index=data.index)
        indicators['bb_lower'] = pd.Series([val.lb if val else None for val in bb_output], index=data.index)
        indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']
        indicators['bb_percent'] = (close - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        # Bollinger Band Squeeze (low volatility precedes big moves)
        indicators['bb_squeeze'] = (indicators['bb_width'] < indicators['bb_width'].rolling(20).quantile(0.2)).astype(int)
        
        # ATR for volatility-adjusted position sizing
        indicators['atr'] = pd.Series(ATR(14, input_values=ohlcv_list), index=data.index)
        indicators['atr_percent'] = indicators['atr'] / close

        # Stochastic Oscillator
        stoch_output = Stoch(period=14, smoothing_period=3, input_values=ohlcv_list)
        indicators['stoch_k'] = pd.Series([val.k if val else None for val in stoch_output], index=data.index)
        indicators['stoch_d'] = pd.Series([val.d if val else None for val in stoch_output], index=data.index)

        # Moved to Volume I think
        # indicators['obv'] = pd.Series(OBV(input_values=ohlcv_list), index=data.index)
        # indicators['ad'] = self._accumulation_distribution_line(high, low, close, volume)

        indicators['cci'] = pd.Series(CCI(20, input_values=ohlcv_list), index=data.index)

        return indicators
    
    def create_multitimeframe_features(self, data):
        """Enhanced multi-timeframe analysis optimized for different holding periods"""
        mtf_features = {}
        
        # Define timeframes relevant for swing trading
        timeframes = {
            '3D': '3D',   # Short-term momentum
            '1W': 'W',   # Weekly trends
            '1M': 'ME',   # Monthly trends
        }
        
        for tf_name, tf_resample in timeframes.items():
            # Resample data
            tf_data = data.resample(tf_resample).agg({
                'open': 'first', 'high': 'max', 'low': 'min', 
                'close': 'last', 'volume': 'sum'
            }).dropna()
            
            if len(tf_data) < 50:
                continue
                
            tf_close_list = tf_data['close'].tolist()
            
            # Trend strength indicator
            ema_short = pd.Series(EMA(10, tf_close_list), index=tf_data.index)
            ema_long = pd.Series(EMA(30, tf_close_list), index=tf_data.index)
            trend_strength = ((ema_short - ema_long) / ema_long * 100).reindex(data.index, method='ffill')
            mtf_features[f'{tf_name}_trend_strength'] = trend_strength
            
            # Momentum
            momentum = tf_data['close'].pct_change(5).reindex(data.index, method='ffill')
            mtf_features[f'{tf_name}_momentum'] = momentum
            
            # RSI for overbought/oversold across timeframes
            rsi = pd.Series(RSI(14, tf_close_list), index=tf_data.index).reindex(data.index, method='ffill')
            mtf_features[f'{tf_name}_rsi'] = rsi
            
            # Volatility
            volatility = tf_data['close'].pct_change().rolling(20).std().reindex(data.index, method='ffill')
            mtf_features[f'{tf_name}_volatility'] = volatility
        
        return mtf_features
    
    
    def calculate_advanced_volatility_features(self, data):
        """GARCH-style volatility decomposition"""
        features = {}
        returns = data['close'].pct_change()
        
        # Realized volatility at multiple horizons
        for period in [5, 10, 20, 60]:
            features[f'realized_vol_{period}'] = returns.rolling(period).std() * np.sqrt(252)
        
        # Parkinson volatility (using high-low range)
        parkinson_vol = np.sqrt(252 / (4 * np.log(2))) * np.log(data['high'] / data['low']).rolling(20).mean()
        features['parkinson_volatility'] = parkinson_vol
        
        # Garman-Klass volatility
        log_hl = np.log(data['high'] / data['low']) ** 2
        log_co = np.log(data['close'] / data['open']) ** 2
        gk_vol = np.sqrt(252 * (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(20).mean())
        features['garman_klass_vol'] = gk_vol
        
        # Volatility of volatility
        vol_20 = returns.rolling(20).std()
        features['vol_of_vol'] = vol_20.rolling(20).std()
        
        # Skewness and kurtosis
        features['returns_skew'] = returns.rolling(60).skew()
        features['returns_kurt'] = returns.rolling(60).kurt()
        
        # Up/down volatility separation
        up_returns = returns[returns > 0]
        down_returns = returns[returns < 0]
        features['upside_vol'] = up_returns.rolling(20, min_periods=1).std() * np.sqrt(252)
        features['downside_vol'] = down_returns.rolling(20, min_periods=1).std() * np.sqrt(252)
        features['vol_asymmetry'] = features['upside_vol'] / (features['downside_vol'] + 1e-8)
        
        return features
    

    def calculate_cross_asset_features(self, data, market_data):
        """Cross-asset correlations and beta calculations"""
        features = {}
        
        # Ensure alignment
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        if market_data.index.tz is not None:
            market_data.index = market_data.index.tz_localize(None)
        
        asset_returns = data['close'].pct_change()
        market_returns = market_data['Close'].pct_change()
        
        # Rolling correlations at multiple windows
        for window in [20, 60, 252]:
            correlation = asset_returns.rolling(window).corr(market_returns)
            features[f'market_corr_{window}d'] = correlation
        
        # Rolling beta
        for window in [20, 60]:
            covariance = asset_returns.rolling(window).cov(market_returns)
            market_variance = market_returns.rolling(window).var()
            beta = covariance / (market_variance + 1e-8)
            features[f'market_beta_{window}d'] = beta
        
        # Relative strength
        features['relative_strength'] = (data['close'] / data['close'].shift(20)) / \
                                       (market_data['Close'] / market_data['Close'].shift(20))
        
        # Correlation stability (how stable is the correlation)
        corr_20 = asset_returns.rolling(20).corr(market_returns)
        features['correlation_stability'] = corr_20.rolling(60).std()
        
        return features
    
    def calculate_cyclical_temporal_features(self, data):
        """Cyclical encoding for temporal patterns"""
        features = {}
        
        # Day of week
        day_of_week = data.index.dayofweek
        features['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        features['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        
        # Day of month
        day_of_month = data.index.day
        features['dom_sin'] = np.sin(2 * np.pi * day_of_month / 31)
        features['dom_cos'] = np.cos(2 * np.pi * day_of_month / 31)
        
        # Month of year
        month = data.index.month
        features['month_sin'] = np.sin(2 * np.pi * month / 12)
        features['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        # Week of year
        week = data.index.isocalendar().week
        features['week_sin'] = np.sin(2 * np.pi * week / 52)
        features['week_cos'] = np.cos(2 * np.pi * week / 52)
        
        # Options expiry week (3rd Friday)
        features['options_week'] = ((data.index.day >= 15) & (data.index.day <= 21) & 
                                   (data.index.dayofweek == 4)).astype(int)
        
        # Quarter end effects
        features['quarter_end'] = data.index.is_quarter_end.astype(int)
        features['month_end'] = data.index.is_month_end.astype(int)
        
        return features

    def calculate_directional_features(self, data):
        """Features optimized for directional accuracy rather than price prediction"""
        features = {}
        close = data['close']
        
        # Directional momentum
        for period in [3, 5, 10, 20]:
            returns = close.pct_change(period)
            features[f'direction_momentum_{period}'] = np.sign(returns)
            features[f'direction_strength_{period}'] = np.abs(returns)
        
        # Consecutive up/down days
        daily_direction = np.sign(close.pct_change())
        features['consecutive_direction'] = daily_direction.groupby((daily_direction != daily_direction.shift()).cumsum()).cumsum()
        
        # Time since last reversal
        reversal_points = (daily_direction != daily_direction.shift()).astype(int)
        features['bars_since_reversal'] = reversal_points.groupby(reversal_points.cumsum()).cumcount()
        
        # Directional volatility
        up_moves = close.pct_change()[close.pct_change() > 0]
        down_moves = close.pct_change()[close.pct_change() < 0]
        features['up_move_avg'] = up_moves.rolling(20, min_periods=1).mean()
        features['down_move_avg'] = down_moves.rolling(20, min_periods=1).mean()
        
        # Directional volume
        volume = data['volume']
        features['up_volume_ratio'] = (volume * (close.pct_change() > 0)).rolling(20).sum() / \
                                     volume.rolling(20).sum()
        
        return features



    def calculate_microstructure_features(self, data):
        """Market microstructure features"""
        features = {}
        high, low, close, volume = data['high'], data['low'], data['close'], data['volume']
        
        # Bid-ask spread proxy
        features['spread_proxy'] = (high - low) / close
        features['spread_proxy_pct_rank'] = features['spread_proxy'].rolling(252).rank(pct=True)
        
        # Amihud illiquidity measure
        features['illiquidity'] = np.abs(close.pct_change()) / (volume + 1)
        features['illiquidity_ma'] = features['illiquidity'].rolling(20).mean()

        # Price efficiency measures
        returns = close.pct_change()
        features['autocorr_1'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=1))
        features['autocorr_2'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=2))
        
        # Kyle's lambda (price impact)
        features['kyle_lambda'] = np.abs(returns) / np.log(volume + 1)
        
        return features
    

    def calculate_price_action_features(self, data):
        """Price action and pattern recognition features"""
        features = {}
        close, high, low, open_price = data['close'], data['high'], data['low'], data['open']
        
        # Candlestick patterns
        body = abs(close - open_price)
        full_range = high - low + 1e-8
        upper_shadow = high - np.maximum(close, open_price)
        lower_shadow = np.minimum(close, open_price) - low
        
        features['body_ratio'] = body / full_range
        features['upper_shadow_ratio'] = upper_shadow / full_range
        features['lower_shadow_ratio'] = lower_shadow / full_range
        
        # Doji detection
        features['is_doji'] = (features['body_ratio'] < 0.1).astype(int)
        features['is_hammer'] = ((lower_shadow > 2 * body) &
                                (upper_shadow < body)).astype(int)
        features['is_shooting_star'] = ((upper_shadow > 2 * body) &
                                        (lower_shadow < body)).astype(int)
        
        # Pin bars
        features['bullish_pin_bar'] = ((lower_shadow > 2 * body) & 
                                      (close > open_price) & 
                                      (close > close.shift(1))).astype(int)
        features['bearish_pin_bar'] = ((upper_shadow > 2 * body) & 
                                      (close < open_price) & 
                                      (close < close.shift(1))).astype(int)
        
        # Inside bars and outside bars
        features['inside_bar'] = ((high < high.shift(1)) & 
                                 (low > low.shift(1))).astype(int)
        features['outside_bar'] = ((high > high.shift(1)) & 
                                  (low < low.shift(1))).astype(int)
        
        # Gap detection
        prev_close = close.shift(1)
        features['gap_up'] = ((open_price > prev_close * 1.002)).astype(int)
        features['gap_down'] = ((open_price < prev_close * 0.998)).astype(int)
        features['gap_filled'] = (((open_price > close.shift(1)) & (low <= close.shift(1))) |
                                 ((open_price < close.shift(1)) & (high >= close.shift(1)))).astype(int)
        
        # Support/Resistance levels
        for period in [20, 50]:
            features[f'distance_from_high_{period}'] = (close - high.rolling(period).max()) / close
            features[f'distance_from_low_{period}'] = (close - low.rolling(period).min()) / close
            features[f'near_resistance_{period}'] = (close / high.rolling(period).max() > 0.98).astype(int)
            features[f'near_support_{period}'] = (close / low.rolling(period).min() < 1.02).astype(int)
        
        return features

    def calculate_volume_profile_features(self, data):
        """Volume-based features for institutional activity detection"""
        features = {}
        volume = data['volume']
        close = data['close']
        
        # Volume moving averages and ratios
        vol_sma_20 = volume.rolling(20).mean()
        vol_sma_50 = volume.rolling(50).mean()
        features['volume_ratio_20_50'] = vol_sma_20 / (vol_sma_50 + 1e-8)
        features['volume_ratio_to_20ma'] = volume / (vol_sma_20 + 1e-8)

        # Volume rate of change
        features['volume_roc_5'] = volume.pct_change(5)
        features['volume_roc_10'] = volume.pct_change(10)

        # On-Balance Volume variations
        obv_sign = np.sign(close.pct_change())
        features['obv'] = (volume * obv_sign).cumsum()
        features['obv_ema_ratio'] = features['obv'] / features['obv'].ewm(span=20).mean()
        
        # Volume-weighted price
        features['vwap'] = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
        features['price_to_vwap'] = close / features['vwap']

        # Money Flow
        features['cmf'] = self._chaikin_money_flow(data['high'], data['low'], close, volume)
        features['ad_line'] = self._accumulation_distribution_line(data['high'], data['low'], close, volume)
        
        # Volume concentration
        features['volume_concentration'] = volume.rolling(5).sum() / volume.rolling(20).sum()
        
        # Large volume detection
        volume_zscore = (volume - vol_sma_20) / volume.rolling(20).std()
        features['high_volume_spike'] = (volume_zscore > 2).astype(int)
        features['low_volume_day'] = (volume_zscore < -1).astype(int)

        # Volume moving averages
        vol_sma_20 = volume.rolling(20).mean()
        features['volume_ratio'] = volume / vol_sma_20
        
        # Volume-price trend
        features['vpt'] = ((close.pct_change() * volume).cumsum())
        
        # Price-volume correlation
        features['pv_corr_10'] = close.rolling(10).corr(volume)
        
        # Volume clustering (institutions tend to trade in similar sizes)
        volume_std = volume.rolling(20).std()
        features['volume_zscore'] = (volume - vol_sma_20) / (volume_std + 1e-8)
                
        return features
    
    # def calculate_asset_specific_features(self, data, symbol):
    #     """Asset-specific features based on known characteristics"""
    #     features = {}
    #     close = data['close']
        
    #     # Different assets have different volatility patterns
    #     if symbol in ['TSLA']:
    #         # TSLA is news-sensitive, add momentum features
    #         features['momentum_strength'] = abs(close.pct_change(3))
    #         features['breakout_potential'] = (close / close.rolling(5).max()).rolling(3).mean()
        
    #     elif symbol in ['AAPL', 'MSFT', 'GOOGL']:
    #         # Tech stocks - earnings and product cycle sensitive
    #         features['earnings_cycle'] = np.sin(2 * np.pi * np.arange(len(close)) / 63)  # ~quarterly
    #         features['product_cycle'] = np.sin(2 * np.pi * np.arange(len(close)) / 252)  # ~yearly
        
    #     elif symbol in ['QQQ', 'SPY', 'IWM']:
    #         # ETFs - macro sensitive
    #         features['macro_momentum'] = close.pct_change(20)  # Monthly momentum
    #         features['sector_rotation'] = close.rolling(10).std() / close.rolling(50).std()
        
    #     # Universal features with asset-specific calibration
    #     volatility_window = 20 if symbol in ['TSLA'] else 30  # TSLA needs shorter window
    #     features['adaptive_volatility'] = close.pct_change().rolling(volatility_window).std()
        
    #     return features
    
   

    # def create_multitimeframe_features(self, data):
    #     """Multi-timeframe feature engineering"""
    #     mtf_features = {}
    #     feature_index = data.index
        
    #     agg_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    #     hourly_data = data.resample('h').agg(agg_rules).dropna()
    #     daily_data = data.resample('D').agg(agg_rules).dropna()

    #     for tf_name, tf_data in [('hourly', hourly_data), ('daily', daily_data)]:
    #         if len(tf_data) < 50:
    #             mtf_features[f'{tf_name}_trend_alignment'] = pd.Series(0, index=feature_index)
    #             mtf_features[f'{tf_name}_rsi'] = pd.Series(50, index=feature_index)
    #             mtf_features[f'{tf_name}_momentum'] = pd.Series(0, index=feature_index)
    #         else:
    #             tf_close_list = tf_data['close'].tolist()
    #             sma_20 = pd.Series(SMA(20, tf_close_list), index=tf_data.index)
    #             sma_50 = pd.Series(SMA(50, tf_close_list), index=tf_data.index)
    #             trend_alignment = (sma_20 > sma_50).astype(int).reindex().ffill().fillna(0)
    #             mtf_features[f'{tf_name}_trend_alignment'] = trend_alignment
                
    #             rsi = pd.Series(RSI(14, tf_close_list), index=tf_data.index).reindex().ffill().fillna(50)
    #             mtf_features[f'{tf_name}_rsi'] = rsi
                
    #             momentum = tf_data['close'].diff(periods=10).reindex().ffill().fillna(0)
    #             mtf_features[f'{tf_name}_momentum'] = momentum
        
    #     return mtf_features
    
    def detect_market_regime(self, data: pd.DataFrame, ohlcv_list: list[OHLCV]):
        """Market regime detection features"""
        regime_features = {}
        close = data['close']
        
        volatility = close.pct_change().rolling(20).std()
        vol_q33 = volatility.rolling(252).quantile(0.33)
        vol_q67 = volatility.rolling(252).quantile(0.67)
        
        regime_features['low_vol_regime'] = (volatility < vol_q33).astype(int)
        regime_features['high_vol_regime'] = (volatility > vol_q67).astype(int)
        
        adx_output = ADX(di_period=14, adx_period=14, input_values=ohlcv_list)
        adx_series = pd.Series([val.adx if val else None for val in adx_output], index=data.index)
        
        regime_features['trending_market'] = (adx_series > 25).astype(int)
        regime_features['consolidating_market'] = (adx_series < 20).astype(int)
        
        return regime_features
    
    def get_market_context_data(self, index, benchmark='SPY'):
        """Fetches benchmark data (e.g., SPY) for the given time index."""
        if index.empty: return pd.DataFrame()

        start_str = index.min().strftime('%Y-%m-%d')
        end_str = (index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        interval = "1h"
        cache_path = self.cache_dir / f"{benchmark}_{start_str}_to_{end_str}_{interval}.csv"

        market_data = None
        if cache_path.exists():
            file_mod_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
            if (datetime.now() - file_mod_time).total_seconds() < config.CACHE_LIFESPAN_HOURS * 3600:
                market_data = pd.read_csv(cache_path, index_col=0)
        
        if market_data is None:
            time.sleep(1)
            market_data = yf.download(benchmark, start=start_str, end=end_str, interval=interval)
            if not market_data.empty: market_data.to_csv(cache_path)

        if market_data is None or market_data.empty:
            return pd.DataFrame()

        if isinstance(market_data.columns, pd.MultiIndex):
            market_data.columns = market_data.columns.get_level_values(0)

        # Use ISO8601 format which is more robust for datetime with timezones
        try:
            # The %z directive handles the UTC offset (e.g., -0400).
            market_data.index = pd.to_datetime(market_data.index, format='%Y-%m-%d %H:%M:%S%z', errors='raise')
        except (ValueError, TypeError):
            market_data.index = pd.to_datetime(market_data.index, errors='coerce', utc=True)
        
        market_data = market_data[market_data.index.notna()]
        market_data.index.name = 'timestamp'

        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in numeric_cols:
            if col in market_data.columns:
                market_data[col] = pd.to_numeric(market_data[col], errors='coerce')

        market_data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
            
        return market_data

    def calculate_market_context_features(self, data, market_data):
        """Calculates features based on the asset's relation to the broader market."""
        if market_data.empty: return {}

        context_features = {}
        # Ensure both dataframes are timezone-naive for comparison
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        if market_data.index.tz is not None:
             market_data.index = market_data.index.tz_localize(None)

        asset_returns = data['close'].pct_change()
        market_returns = market_data['Close'].pct_change()

        returns_df = pd.DataFrame({'asset': asset_returns, 'market': market_returns}).ffill()
        rolling_corr = returns_df['asset'].rolling(window=50).corr(returns_df['market'])

        context_features['market_corr'] = rolling_corr.reindex(data.index, method='ffill').bfill()
        return context_features

    def calculate_risk_features(self, data):
        """Calculates features related to risk and volatility."""
        risk_features = {}
        returns = data['close'].pct_change()
        risk_features['volatility_20d'] = returns.rolling(window=20).std() * np.sqrt(252) # Note: This is still an annualized daily volatility
        return risk_features

    def normalize_and_structure_features(self, features: dict) -> pd.DataFrame:
        """Combines all feature series into a single, cleaned, and normalized DataFrame."""
        valid_features = {k: v for k, v in features.items() if isinstance(v, pd.Series) and not v.isnull().all()}

        if not valid_features: return pd.DataFrame()
        
        df = pd.concat(valid_features.values(), axis=1)
        df.columns = valid_features.keys()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        df.dropna(thresh=int(df.shape[1] * 0.8), axis=0, inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        if df.empty:
            print("ERROR: DataFrame is empty after handling NaN values.")
            return df

        scaler = RobustScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
        
        return df_scaled