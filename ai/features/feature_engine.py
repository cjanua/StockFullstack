# ai/features/feature_engine.py

from datetime import datetime
from pathlib import Path
import time
from talipp.indicators import SMA, EMA, RSI, MACD, BB, ATR, OBV, ADX
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

    def create_comprehensive_features(
        self,
        data: pd.DataFrame,
        market_context_data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Generate multi-timeframe feature set for RNN"""
        if data.index.has_duplicates:
            print("Warning: Data contains duplicate timestamps, dropping duplicates.")
            data = data.loc[~data.index.duplicated(keep='first')]
        
        agg_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        data_hourly = data.resample('D').agg(agg_rules).dropna()
        if data_hourly.empty:
            print("Warning: Not enough data to create hourly features.")
            return pd.DataFrame()

        
        ohlcv_list = [
            OHLCV(row.open, row.high, row.low, row.close, row.volume)
            for row in data.itertuples()
        ]

        features = {}
        
        features.update(self.calculate_technical_indicators(data, ohlcv_list))
        features.update(self.calculate_volume_features(data))
        features.update(self.detect_market_regime(data, ohlcv_list))

        # features.update(self.create_multitimeframe_features(data))
        
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
        
        for period in [10, 20, 50]:
            indicators[f'sma_{period}'] = pd.Series(SMA(period, close_list), index=data.index)
            indicators[f'ema_{period}'] = pd.Series(EMA(period, close_list), index=data.index)
        
        indicators['rsi_14'] = pd.Series(RSI(14, close_list), index=data.index)
        indicators['rsi_21'] = pd.Series(RSI(21, close_list), index=data.index)
        
        macd_output = MACD(fast_period=12, slow_period=26, signal_period=9, input_values=close_list)
        indicators['macd_line'] = pd.Series([val.macd if val else None for val in macd_output], index=data.index)
        indicators['macd_signal'] = pd.Series([val.signal if val else None for val in macd_output], index=data.index)
        indicators['macd_histogram'] = pd.Series([val.histogram if val else None for val in macd_output], index=data.index)
        
        bb_output = BB(period=20, std_dev_mult=2.0, input_values=close_list)
        indicators['bb_upper'] = pd.Series([val.ub if val else None for val in bb_output], index=data.index)
        indicators['bb_middle'] = pd.Series([val.cb if val else None for val in bb_output], index=data.index)
        indicators['bb_lower'] = pd.Series([val.lb if val else None for val in bb_output], index=data.index)
        indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']
        indicators['bb_percent'] = (close - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        indicators['atr'] = pd.Series(ATR(14, input_values=ohlcv_list), index=data.index)
        indicators['obv'] = pd.Series(OBV(input_values=ohlcv_list), index=data.index)
        indicators['ad'] = self._accumulation_distribution_line(high, low, close, volume)
        
        return indicators

    def calculate_volume_features(self, data):
        """Calculates volume-based indicators."""
        features = {}
        features['cmf'] = self._chaikin_money_flow(data['high'], data['low'], data['close'], data['volume'])
        return features

    def create_multitimeframe_features(self, data):
        """Multi-timeframe feature engineering"""
        mtf_features = {}
        feature_index = data.index
        
        agg_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        hourly_data = data.resample('h').agg(agg_rules).dropna()
        daily_data = data.resample('D').agg(agg_rules).dropna()

        for tf_name, tf_data in [('hourly', hourly_data), ('daily', daily_data)]:
            if len(tf_data) < 50:
                mtf_features[f'{tf_name}_trend_alignment'] = pd.Series(0, index=feature_index)
                mtf_features[f'{tf_name}_rsi'] = pd.Series(50, index=feature_index)
                mtf_features[f'{tf_name}_momentum'] = pd.Series(0, index=feature_index)
            else:
                tf_close_list = tf_data['close'].tolist()
                sma_20 = pd.Series(SMA(20, tf_close_list), index=tf_data.index)
                sma_50 = pd.Series(SMA(50, tf_close_list), index=tf_data.index)
                trend_alignment = (sma_20 > sma_50).astype(int).reindex(feature_index, method='ffill').fillna(0)
                mtf_features[f'{tf_name}_trend_alignment'] = trend_alignment
                
                rsi = pd.Series(RSI(14, tf_close_list), index=tf_data.index).reindex(feature_index, method='ffill').fillna(50)
                mtf_features[f'{tf_name}_rsi'] = rsi
                
                momentum = tf_data['close'].diff(periods=10).reindex(feature_index, method='ffill').fillna(0)
                mtf_features[f'{tf_name}_momentum'] = momentum
        
        return mtf_features
    
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