#region imports
from AlgorithmImports import *
#endregion


class JanuaryEffectUniverseSelectionModel(FundamentalUniverseSelectionModel):
    def __init__(self, algorithm: QCAlgorithm, universe_settings: UniverseSettings = None, coarse_size: int = 1_000, fine_size: int = 10) -> None:
        def select(fundamental):
            # Select the securities that have the most dollar volume
            shortlisted = [c for c in sorted(fundamental, key=lambda x: x.dollar_volume, reverse=True)[:coarse_size]]
            
            fine = [i for i in shortlisted if i.earning_reports.basic_average_shares.three_months!=0
                                    and i.earning_reports.basic_eps.twelve_months!=0
                                    and i.valuation_ratios.pe_ratio!=0]
            # Sort securities by market cap
            sorted_by_market_cap = sorted(fine, key = lambda x: x.market_cap, reverse=True)
            # In January, select the securities with the smallest market caps
            if algorithm.time.month == 1:
                return [f.symbol for f in sorted_by_market_cap[-fine_size:]]
            # If it's not January, select the securities with the largest market caps
            return [f.symbol for f in sorted_by_market_cap[:fine_size]]
        
        super().__init__(select, universe_settings)
    
