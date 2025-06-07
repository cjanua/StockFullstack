# ai/features/feature_engine.py
from datetime import datetime
from pathlib import Path
import time
import pandas_ta as ta
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

    def create_comprehensive_features(
        self,
        data: pd.DataFrame,
        market_context_data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Generate multi-timeframe feature set for RNN"""
        features = {}
        
        # 1. Core technical indicators
        features.update(self.calculate_technical_indicators(data))

        # 2. Market microstructure / Volume features
        if 'volume' in data.columns and not data['volume'].isnull().all():
            features.update(self.calculate_volume_features(data))
        
        # 3. Multi-timeframe analysis
        features.update(self.create_multitimeframe_features(data))
        
        # 4. Regime detection features
        features.update(self.detect_market_regime(data))
        
        # 5. Cross-asset features
        if market_context_data is not None:
            features.update(self.calculate_market_context_features(data, market_context_data))
        
        # 6. Risk features
        features.update(self.calculate_risk_features(data))
        features['close'] = data['close']  # Always include close price
        
        # 7. Combine and normalize
        return self.normalize_and_structure_features(features)
    
    def calculate_technical_indicators(self, data):
        """Comprehensive technical indicator calculation"""
        indicators = {}
        close = data['close']
        high = data['high']
        low = data['low']
        
        # Trend indicators
        for period in [10, 20, 50]:
            indicators[f'sma_{period}'] = ta.sma(close, length=period)
            indicators[f'ema_{period}'] = ta.ema(close, length=period)
        
        # Momentum indicators
        indicators['rsi_14'] = ta.rsi(close, length=14)
        indicators['rsi_21'] = ta.rsi(close, length=21)
        
        # MACD family
        macd = ta.macd(close)
        indicators['macd_line'] = macd['MACD_12_26_9']
        indicators['macd_signal'] = macd['MACDs_12_26_9']
        indicators['macd_histogram'] = macd['MACDh_12_26_9']
        
        BB_LEN = 20
        BB_STD = 2.0
        # Volatility indicators
        bb = ta.bbands(close, length=BB_LEN, std=BB_STD)
        def wrap_bb(key):
            return f'{key}_{BB_LEN}_{BB_STD}'
        indicators['bb_lower'] = bb.get(wrap_bb('BBL'))
        indicators['bb_middle'] = bb.get(wrap_bb('BBM'))
        indicators['bb_upper'] = bb.get(wrap_bb('BBU'))
        indicators['bb_width'] = bb.get(wrap_bb('BBB'))
        indicators['bb_percent'] = bb.get(wrap_bb('BBP'))
        
        indicators['atr'] = ta.atr(high, low, close)
        
        # Volume indicators
        indicators['obv'] = ta.obv(close, data['volume'])
        indicators['ad'] = ta.ad(high, low, close, data['volume'])
        
        return indicators

    def calculate_volume_features(self, data):
        """Calculates volume-based indicators."""
        features = {}
        volume = data['volume']
        close = data['close']
        
        features['obv'] = ta.obv(close, volume)
        # Chaikin Money Flow
        cmf = ta.cmf(data['high'], data['low'], close, volume)
        if cmf is not None:
            features['cmf'] = cmf
        return features

    def create_multitimeframe_features(self, data):
        """Multi-timeframe feature engineering"""
        mtf_features = {}
        
        # Resample to different timeframes
        hourly_data = data.resample('1h').agg({
            'open': 'first', 'high': 'max', 
            'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        
        daily_data = data.resample('1D').agg({
            'open': 'first', 'high': 'max',
            'low': 'min', 'close': 'last', 'volume': 'sum'  
        }).dropna()
        
        # Calculate indicators for each timeframe
        for tf_name, tf_data in [('hourly', hourly_data), ('daily', daily_data)]:
            # Trend alignment
            sma_20 = ta.sma(tf_data['close'], length=20)
            sma_50 = ta.sma(tf_data['close'], length=50)
            mtf_features[f'{tf_name}_trend_alignment'] = (sma_20 > sma_50).astype(int)
            
            # Momentum
            mtf_features[f'{tf_name}_rsi'] = ta.rsi(tf_data['close'])
            mtf_features[f'{tf_name}_momentum'] = ta.mom(tf_data['close'])
        
        return mtf_features
    
    def detect_market_regime(self, data):
        """Market regime detection features"""
        regime_features = {}
        close, high, low = data['close'], data['high'], data['low']
        
        # Volatility regime
        volatility = close.pct_change().rolling(20).std()
        # Calculate each quantile in a separate call
        vol_q33 = volatility.rolling(252).quantile(0.33)
        vol_q67 = volatility.rolling(252).quantile(0.67)
        
        # Use the new variables to create the regime features
        regime_features['low_vol_regime'] = (volatility < vol_q33).astype(int)
        regime_features['high_vol_regime'] = (volatility > vol_q67).astype(int)
        
        # Trend regime using ADX
        adx = ta.adx(data['high'], data['low'], data['close'])['ADX_14']
        regime_features['trending_market'] = (adx > 25).astype(int)
        regime_features['consolidating_market'] = (adx < 20).astype(int)
        
        return regime_features
    
    def get_market_context_data(self, index, benchmark='SPY'):
        """Fetches benchmark data (e.g., SPY) for the given time index."""
        if index.empty:
            return pd.DataFrame()

        start_str = index.min().strftime('%Y-%m-%d')
        end_str = index.max().strftime('%Y-%m-%d')

        cache_path = self.cache_dir / f"{benchmark}_{start_str}_to_{end_str}.csv"

        if cache_path.exists():
            file_mod_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
            if (datetime.now() - file_mod_time).total_seconds() < config.CACHE_LIFESPAN_HOURS * 3600:
                print(f"CACHE HIT: Loading benchmark '{benchmark}' from {cache_path}")
                # When reading from CSV, 'Date' becomes a regular column, so we set it as the index
                return pd.read_csv(cache_path, index_col='Date', parse_dates=True)

        benchmark_sym = yf.Ticker(benchmark)
        attempts = 0
        max_attempts = 5
        while attempts < max_attempts:
            try:
                market_data = benchmark_sym.history(
                    start=index.min(),
                    end=index.max(),
                    interval="4h",
                )
                if not market_data.empty:
                    market_data.index = market_data.index.tz_localize(None)
                    market_data.to_csv(cache_path)
                    print(f"CACHE SAVE: Saved '{benchmark}' data to {cache_path}")
                return market_data
            except Exception as e:
                print(f"ERROR downloading benchmark data: {repr(e)}")
                attempts += 1
                time.sleep(5)
        print(f"Failed to download benchmark data for {benchmark} after {max_attempts} attempts.")
        return pd.DataFrame()  # Return empty DataFrame if download fails


    def calculate_market_context_features(self, data, market_data):
        """Calculates features based on the asset's relation to the broader market."""
        context_features = {}
        asset_returns = data['close'].pct_change()
        market_returns = market_data['Close'].pct_change()
        
        # Rolling correlation with the market
        context_features['market_corr'] = asset_returns.rolling(window=20).corr(market_returns)
        return context_features

    def calculate_risk_features(self, data):
        """Calculates features related to risk and volatility."""
        risk_features = {}
        returns = data['close'].pct_change()
        # Historical volatility (rolling standard deviation of returns)
        risk_features['volatility_20d'] = returns.rolling(window=20).std() * np.sqrt(252)
        return risk_features

    def normalize_and_structure_features(self, features: dict) -> pd.DataFrame:
        """Combines all feature series into a single, cleaned, and normalized DataFrame."""
        # 1. filter for valid features
        valid_features = {name: series for name, series in features.items() if not series.isnull().all()}
        if not valid_features:
            print(f"ERROR: No valid features could be calculated for this symbol.")
            return pd.DataFrame()
        
        df = pd.concat(features.values(), axis=1)
        df.columns = features.keys() # Ensure column names are correct
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        nan_counts = df.isna().sum()
        print(f"DEBUG: NaN counts before filling:\n{nan_counts[nan_counts > 0]}")

        # 2. Handle missing values robustly
        # First, forward-fill to propagate last known values, then back-fill for any remaining NaNs at the start
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.dropna(inplace=True)
        
        
        if df.empty:
            return df # Return empty if all data was dropped

        # 4. Normalize the data
        # Using RobustScaler as it's less sensitive to outliers
        scaler = RobustScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
        
        return df_scaled
