#region imports
from AlgorithmImports import *

from universe import JanuaryEffectUniverseSelectionModel
from alpha import LongMonthlyAlphaModel
#endregion


class JanuaryEffectInStocksAlgorithm(QCAlgorithm):

    _undesired_symbols_from_previous_deployment = []
    _checked_symbols_from_previous_deployment = False

    def initialize(self):
        self.set_start_date(2023, 3, 1)
        self.set_end_date(2024, 3, 1)
        self.set_cash(1_000_000) 

        self.settings.minimum_order_margin_portfolio_percentage = 0
        self.set_security_initializer(BrokerageModelSecurityInitializer(self.brokerage_model, FuncSecuritySeeder(self.get_last_known_prices)))
        
        self.universe_settings.data_normalization_mode = DataNormalizationMode.RAW
        self.universe_settings.schedule.on(self.date_rules.month_start())
        self.add_universe_selection(JanuaryEffectUniverseSelectionModel(
            self,
            self.universe_settings,
            self.get_parameter("coarse_size", 1_000),
            self.get_parameter("fine_size", 10)
        ))

        self.add_alpha(LongMonthlyAlphaModel())

        self.settings.rebalance_portfolio_on_security_changes = False
        self.settings.rebalance_portfolio_on_insight_changes = False
        self.month = -1
        self.set_portfolio_construction(EqualWeightingPortfolioConstructionModel(self._rebalance_func))

        self.add_risk_management(NullRiskManagementModel())

        self.set_execution(ImmediateExecutionModel())

        self.set_warm_up(timedelta(31))

    def _rebalance_func(self, time):
        if self.month != self.time.month and not self.is_warming_up and self.current_slice.quote_bars.count > 0:
            self.month = self.time.month
            return time
        return None
    
    def on_data(self, data):
        # Exit positions that aren't backed by existing insights.
        # If you don't want this behavior, delete this method definition.
        if not self.is_warming_up and not self._checked_symbols_from_previous_deployment:
            for security_holding in self.portfolio.values():
                if not security_holding.invested:
                    continue
                symbol = security_holding.symbol
                if not self.insights.has_active_insights(symbol, self.utc_time):
                    self._undesired_symbols_from_previous_deployment.append(symbol)
            self._checked_symbols_from_previous_deployment = True
        
        for symbol in self._undesired_symbols_from_previous_deployment:
            if self.is_market_open(symbol):
                self.liquidate(symbol, tag="Holding from previous deployment that's no longer desired")
                self._undesired_symbols_from_previous_deployment.remove(symbol)
