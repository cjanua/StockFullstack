# region imports
from AlgorithmImports import *
# endregion

class EMAMomentumUniverse(QCAlgorithm):
    
    def Initialize(self):
        # Define backtest window and portfolio cash
        self.SetStartDate(2020, 6, 10)
        self.SetEndDate(2025, 6, 9)
        self.SetCash(100000)

        # Add the assets to be fed into the algorithm and save the symbol objects (to be referred later)
        self.stock1_symbol = self.AddEquity('EWA', Resolution.Daily).Symbol
        self.stock2_symbol = self.AddEquity('EWG', Resolution.Daily).Symbol
        
        # Create two identity indicators (a indicator that repeats the value without any processing)
        self.stock1_identity = Identity("My_Stock1")
        self.stock2_identity = Identity("My_Stock2")

        # Set these indicators to receive the data from Stock1 and Stock2        
        self.RegisterIndicator(self.stock1_symbol, self.stock1_identity, Resolution.Daily)
        self.RegisterIndicator(self.stock2_symbol, self.stock2_identity, Resolution.Daily)

        # create the portfolio as a new indicator
        # this is handy as the portfolio will be updated as new data comes in, without the necessity of updating the values manually
        # as the QCAlgorithm already has a Portfolio attribute, we will call our combined portfolio as series
        self.series = IndicatorExtensions.Minus(self.stock1_identity, IndicatorExtensions.Times(self.stock2_identity, 1.312))

        # We then create a bollinger band with 120 steps for lookback period
        self.bb = BollingerBands(120, 0.8, MovingAverageType.Exponential)
        
        # Define the objectives when going long or going short (long=buy Stock1 and sell Stock2) (short=sell Stock1 and buy Stock2)
        self.long_targets = [PortfolioTarget(self.stock1_symbol, 0.9), PortfolioTarget(self.stock2_symbol, -0.9)]
        self.short_targets = [PortfolioTarget(self.stock1_symbol, -0.9), PortfolioTarget(self.stock2_symbol, 0.9)]

        self.is_invested = None

    def OnData(self, data):

        # for daily bars data is delivered at 00:00 of the day containing the closing price of the previous day (23:59:59)
        if (not data.Bars.ContainsKey(self.stock1_symbol)) or (not data.Bars.ContainsKey(self.stock2_symbol)):
            return

        #update the Bollinger Band value
        self.bb.Update(self.Time, self.series.Current.Value)

        # check if the bolllinger band indicator is ready (filled with 120 steps)
        if not self.bb.IsReady:
            return

        serie = self.series.Current.Value

        self.Plot("OIH Prices", "Open", self.Securities[self.stock2_symbol].Open)
        self.Plot("BABA Prices", "Close", self.Securities[self.stock2_symbol].Close)
        
        self.Plot("Indicators", "Serie", serie)
        self.Plot("Indicators", "Middle", self.bb.MiddleBand.Current.Value)
        self.Plot("Indicators", "Upper", self.bb.UpperBand.Current.Value)
        self.Plot("Indicators", "Lower", self.bb.LowerBand.Current.Value)
        
        # if it is not invested, see if there is an entry point
        if not self.is_invested:
            # if our portfolio is bellow the lower band, enter long
            if serie < self.bb.LowerBand.Current.Value:
                self.SetHoldings(self.long_targets)

                self.Debug('Entering Long')
                self.is_invested = 'long'
            
            # if our portfolio is above the upper band, go short
            if serie > self.bb.UpperBand.Current.Value:
                self.SetHoldings(self.short_targets)
                
                self.Debug('Entering Short')
                self.is_invested = 'short'
        
        # if it is invested in something, check the exiting signal (when it crosses the mean)   
        elif self.is_invested == 'long':
            if serie > self.bb.MiddleBand.Current.Value:
                self.Liquidate()
                self.Debug('Exiting Long')
                self.is_invested = None
                
        elif self.is_invested == 'short':
            if serie < self.bb.MiddleBand.Current.Value:
                self.Liquidate()
                self.Debug('Exiting Short')
                self.is_invested = None