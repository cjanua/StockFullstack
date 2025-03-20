from alpaca.trading.models import PortfolioHistory

def serialize_portfolio_history(history: PortfolioHistory) -> dict:
    """Convert PortfolioHistory object to a serializable dictionary.
    
    Args:
        history (PortfolioHistory): Portfolio history object to serialize
        
    Returns:
        dict: Serialized portfolio history data
    """
    return {
        'timestamp': [int(t) for t in history.timestamp],
        'equity': [str(e) for e in history.equity],
        'profit_loss': [str(pl) for pl in history.profit_loss],
        'profit_loss_pct': [str(plp) if plp is not None else None for plp in history.profit_loss_pct],
        'base_value': str(history.base_value),
        'timeframe': str(history.timeframe)
    }