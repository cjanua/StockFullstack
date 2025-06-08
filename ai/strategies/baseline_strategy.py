# ai/strategies/rnn_trading.py

import talipp.indicators as ta
from backtesting import Strategy

class BaselineStrategy(Strategy):
    def init(self):
        self.rsi = self.I(ta.RSI, self.data.Close, 14)
        self.sma_short = self.I(ta.SMA, self.data.Close, 10)
        self.sma_long = self.I(ta.SMA, self.data.Close, 50)
    
    def next(self):
        if self.sma_short[-1] > self.sma_long[-1] and self.rsi[-1] < 70:
            self.buy(size=0.5)
        elif self.sma_short[-1] < self.sma_long[-1] and self.rsi[-1] > 30:
            self.sell(size=0.5)