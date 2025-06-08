# ai/monitoring/performance_metrics.py
import pandas as pd
import yfinance as yf
from scipy import stats
import quantstats as qs

from ai.config.settings import config
from backend.alpaca.sdk.clients import AlpacaDataConnector

def get_benchmark_returns(start_date, end_date, ticker='SPY'):
    """
    Fetches historical daily returns for a benchmark ticker.
    """
    # benchmark_sym = yf.Ticker(ticker)

    try:
        benchmark_data = yf.download(ticker, start=start_date, end=end_date)

        if benchmark_data.empty:
            print(f"Could not fetch benchmark data for {ticker}.")
            return pd.Series(dtype=float)
        benchmark_returns = benchmark_data['Adj Close'].pct_change().dropna()

        # apca = AlpacaDataConnector(config)
        # benchmark_data = apca.get_historical_data(
        #     symbols=[ticker],
        #     lookback_days=(end_date - start_date).days,
        # )
        # benchmark_df = benchmark_data[ticker]
        # if ticker not in benchmark_data:
        #     print(f"Could not fetch benchmark data for {ticker}.")
        #     return pd.Series(dtype=float)
        
        # benchmark_returns = benchmark_df['Close'].pct_change().dropna()

        return benchmark_returns
    except Exception as e:
        print(f"Could not download benchmark data for {ticker}: {e}")
        return pd.Series(dtype=float)

def test_statistical_significance(strategy_returns, benchmark_returns):
    """
    Performs statistical tests to compare strategy returns against a benchmark.
    """
    if strategy_returns.empty:
        return {
            't_statistic': None,
            'p_value': None,
            'alpha': None,
            'beta': None,
        }
        
    # Align dates
    aligned_returns = pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns
    }).dropna()

    alpha = qs.stats.greeks(strategy_returns, benchmark_returns)['alpha']
    beta = qs.stats.greeks(strategy_returns, benchmark_returns)['beta']

    # Calculate t-test on alpha
    alpha_series = aligned_returns['strategy'] - aligned_returns['benchmark']
    t_stat, p_value = stats.ttest_1samp(alpha_series, 0)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'alpha (annualized)': alpha * 252,
        'beta': beta,
    }

def calculate_comprehensive_risk_metrics(backtest_results):
    """
    Calculates a comprehensive set of risk and performance metrics using quantstats.
    
    Args:
        backtest_results: The results object from the backtesting.py library.
    """
    equity_curve = backtest_results._equity_curve['Equity']
    returns = equity_curve.pct_change().dropna()
    
    if returns.empty or len(returns) < 2:
        return {"error": "Not enough data to calculate metrics."}

    # quantstats can generate a dictionary of all key metrics
    metrics_dict = qs.reports.metrics(returns, display=False, mode='full')
    
    # The result is a pandas Series; convert it to a dictionary
    return metrics_dict.to_dict()


def analyze_portfolio_performance(backtest_results: dict):
    """
    Aggregates results from multiple backtests into a single portfolio view.

    Args:
        backtest_results: A dictionary where keys are symbols and values are
                          the result objects from the backtesting.py library.

    Returns:
        A dictionary containing portfolio-level performance metrics.
    """
    all_equity_curves = []
    initial_cash_total = 0.0

    # Collect the equity curve from each backtest result
    for symbol, result in backtest_results.items():
        if '_equity_curve' in result and not result['_equity_curve'].empty:
            # The equity curve includes the initial cash. We need to find the
            # actual capital allocated to this strategy to properly sum them.
            equity = result._equity_curve['Equity']
            initial_cash_total += result._strategy._broker._cash
            all_equity_curves.append(equity)

    if not all_equity_curves:
        return {'portfolio_sharpe': 0.0, 'error': 'No equity curves found.'}

    # Create a single DataFrame with all equity curves, fill missing values
    combined_df = pd.concat(all_equity_curves, axis=1).ffill()

    # The total portfolio value at any time is the sum of all individual equities
    portfolio_equity = combined_df.sum(axis=1)

    # Calculate returns based on the total portfolio equity
    portfolio_returns = portfolio_equity.pct_change().dropna()

    if portfolio_returns.empty or len(portfolio_returns) < 2:
        return {'portfolio_sharpe': 0.0, 'error': 'Not enough data for portfolio analysis.'}

    # Use quantstats to get a full dictionary of metrics
    portfolio_metrics = qs.reports.metrics(portfolio_returns, display=False, mode='full').to_dict()

    # Your main script specifically looks for 'portfolio_sharpe'.
    # quantstats calls this 'Sharpe Ratio', so we'll add a convenience key.
    portfolio_metrics['portfolio_sharpe'] = portfolio_metrics.get('Sharpe Ratio', 0.0)

    return portfolio_metrics
