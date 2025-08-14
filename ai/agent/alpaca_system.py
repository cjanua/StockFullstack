# ai/agent/alpaca_system.py
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient

from ai.models.lstm import TradingLSTM


class AlpacaTradingSystem:
    def __init__(self, api_key, secret_key, paper=True):
        self.api_key = api_key
        self.secret_key = secret_key

        # Trading client setup
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)

        # Data stream setup
        self.data_stream = StockDataStream(api_key, secret_key)
        self.current_data = {}

        # Initialize RNN model
        self.rnn_model = TradingLSTM()
        self.load_pretrained_model()

    async def setup_data_streams(self, symbols):
        """Setup real-time data streaming for multiple symbols."""
        for symbol in symbols:
            self.data_stream.subscribe_trades(self.handle_trade, symbol)
            self.data_stream.subscribe_quotes(self.handle_quote, symbol)
            self.data_stream.subscribe_bars(self.handle_bar, symbol)

    async def handle_bar(self, bar):
        """Process real-time minute bars for RNN prediction."""
        symbol = bar.symbol

        # Update feature vector
        self.update_features(symbol, bar)

        # Generate RNN prediction
        if len(self.current_data[symbol]) >= 60:  # Minimum lookback
            prediction = self.generate_signal(symbol)
            await self.execute_trading_decision(symbol, prediction, bar.close)

    async def execute_trading_decision(self, symbol, signal, current_price):
        """Execute trades based on RNN signals with risk management."""
        account = self.trading_client.get_account()

        # Position sizing with Kelly criterion
        position_size = self.calculate_position_size(
            signal,
            float(account.equity),
            self.get_volatility(symbol)
        )

        # Execute trade with bracket orders
        if signal > 0.7:  # Strong buy signal
            await self.place_bracket_order(symbol, position_size, current_price)
        elif signal < 0.3:  # Strong sell signal
            await self.close_position(symbol)

    def calculate_position_size(self, signal_strength, portfolio_value, volatility):
        """Kelly criterion position sizing with risk management."""
        max_risk_per_trade = 0.02  # 2% portfolio risk
        kelly_fraction = (signal_strength - 0.5) / volatility
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%

        return min(
            kelly_fraction * portfolio_value,
            max_risk_per_trade * portfolio_value / volatility
        )
