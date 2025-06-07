import pandas_ta as ta
from sklearn.preprocessing import RobustScaler

class AdvancedFeatureEngine:
    def __init__(self):
        self.scalers = {}
        self.lookback_window = 60
        
    def create_comprehensive_features(self, data):
        """Generate multi-timeframe feature set for RNN"""
        features = {}
        
        # 1. Core technical indicators
        features.update(self.calculate_technical_indicators(data))
        
        # 2. Multi-timeframe analysis
        features.update(self.create_multitimeframe_features(data))
        
        # 3. Market microstructure features
        if 'volume' in data.columns:
            features.update(self.calculate_volume_features(data))
        
        # 4. Regime detection features
        features.update(self.detect_market_regime(data))
        
        # 5. Cross-asset features
        market_data = self.get_market_context_data(data.index)
        features.update(self.calculate_market_context_features(market_data))
        
        # 6. Risk features
        features.update(self.calculate_risk_features(data))
        
        return self.normalize_and_structure_features(features)
    
    def calculate_technical_indicators(self, data):
        """Comprehensive technical indicator calculation"""
        indicators = {}
        
        # Trend indicators
        for period in [10, 20, 50]:
            indicators[f'sma_{period}'] = ta.sma(data['close'], length=period)
            indicators[f'ema_{period}'] = ta.ema(data['close'], length=period)
        
        # Momentum indicators
        indicators['rsi_14'] = ta.rsi(data['close'], length=14)
        indicators['rsi_21'] = ta.rsi(data['close'], length=21)
        
        # MACD family
        macd = ta.macd(data['close'])
        indicators['macd_line'] = macd['MACD_12_26_9']
        indicators['macd_signal'] = macd['MACDs_12_26_9']
        indicators['macd_histogram'] = macd['MACDh_12_26_9']
        
        # Volatility indicators
        bb = ta.bbands(data['close'])
        indicators['bb_upper'] = bb['BBU_20_2.0']
        indicators['bb_lower'] = bb['BBL_20_2.0']
        indicators['bb_width'] = bb['BBB_20_2.0']
        indicators['bb_percent'] = bb['BBP_20_2.0']
        
        indicators['atr'] = ta.atr(data['high'], data['low'], data['close'])
        
        # Volume indicators
        indicators['obv'] = ta.obv(data['close'], data['volume'])
        indicators['ad'] = ta.ad(data['high'], data['low'], data['close'], data['volume'])
        
        return indicators
    
    def create_multitimeframe_features(self, data):
        """Multi-timeframe feature engineering"""
        mtf_features = {}
        
        # Resample to different timeframes
        hourly_data = data.resample('1H').agg({
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
        
        # Volatility regime
        volatility = data['close'].pct_change().rolling(20).std()
        vol_quantiles = volatility.rolling(252).quantile([0.33, 0.67])
        
        regime_features['low_vol_regime'] = (volatility < vol_quantiles[0.33]).astype(int)
        regime_features['high_vol_regime'] = (volatility > vol_quantiles[0.67]).astype(int)
        
        # Trend regime using ADX
        adx = ta.adx(data['high'], data['low'], data['close'])['ADX_14']
        regime_features['trending_market'] = (adx > 25).astype(int)
        regime_features['consolidating_market'] = (adx < 20).astype(int)
        
        return regime_features