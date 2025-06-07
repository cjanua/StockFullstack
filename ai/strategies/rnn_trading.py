from backtesting import Backtest, Strategy
import numpy as np
import pandas as pd
import torch

class RNNTradingStrategy(Strategy):
    def init(self):
        # Load pre-trained RNN model
        self.rnn_model = torch.load('trained_rnn_model.pth')
        self.rnn_model.eval()
        
        # Feature engineering pipeline
        self.feature_engine = create_feature_pipeline()
        
        # Portfolio state tracking
        self.portfolio_value_history = []
        self.signals_history = []
        
    def next(self):
        # Minimum data requirement
        if len(self.data) < 60:
            return
        
        # Generate features for RNN
        features = self.feature_engine.create_features(
            self.data.df.iloc[-60:]  # 60-day lookback
        )
        
        # Get RNN prediction
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            prediction = self.rnn_model(features_tensor)
            signal = torch.softmax(prediction, dim=1).numpy()[0]
        
        # Trading logic with risk management
        buy_prob, hold_prob, sell_prob = signal
        
        # Position management
        if buy_prob > 0.7 and not self.position:
            # Calculate position size
            size = self.calculate_position_size(buy_prob)
            self.buy(size=size)
            
        elif sell_prob > 0.7 and self.position:
            self.sell(size=self.position.size)
        
        # Track performance
        self.portfolio_value_history.append(self.equity)
        self.signals_history.append(signal)
    
    def calculate_position_size(self, signal_strength):
        """Dynamic position sizing based on signal confidence"""
        max_position = 0.95  # 95% maximum allocation
        base_size = signal_strength * max_position
        
        # Volatility adjustment
        recent_volatility = self.data.Close[-20:].pct_change().std()
        vol_adjustment = max(0.5, 1.0 - recent_volatility * 10)
        
        return base_size * vol_adjustment

# Comprehensive backtesting workflow
def run_comprehensive_backtest(data, strategy_class):
    """Execute full backtesting pipeline with performance analysis"""
    
    # Primary backtest
    bt = Backtest(data, strategy_class, cash=100000, commission=0.002)
    results = bt.run()
    
    # Walk-forward analysis
    wf_results = perform_walk_forward_analysis(data, strategy_class)
    
    # Statistical significance testing
    benchmark_returns = get_benchmark_returns(data.index[0], data.index[-1])
    significance_tests = test_statistical_significance(
        results._trades['ReturnPct'], 
        benchmark_returns
    )
    
    # Risk analysis
    risk_metrics = calculate_comprehensive_risk_metrics(results)
    
    return {
        'backtest_results': results,
        'walk_forward': wf_results,
        'significance_tests': significance_tests,
        'risk_metrics': risk_metrics
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