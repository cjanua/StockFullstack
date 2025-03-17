#region imports
from AlgorithmImports import *
#endregion


class LongMonthlyAlphaModel(AlphaModel):

    _securities = []

    def update(self, algorithm: QCAlgorithm, data: Slice) -> List[Insight]:
        return [Insight.price(security.symbol, Expiry.END_OF_MONTH, InsightDirection.UP) for security in self._securities]

    def on_securities_changed(self, algorithm: QCAlgorithm, changes: SecurityChanges) -> None:
        for security in changes.removed_securities:
            if security in self._securities:
                self._securities.remove(security)
        self._securities.extend(changes.added_securities)
