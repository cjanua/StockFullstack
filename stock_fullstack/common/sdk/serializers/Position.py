from alpaca.trading.models import Position

def serialize_position(pos: Position) -> dict:
   """Convert TradeAccount object to a serializable dictionary matching frontend Account type"""
   return {
        'asset_class': str(pos.asset_class),
        'asset_id': str(pos.asset_id),
        'asset_marginable': bool(pos.asset_marginable),
        'avg_entry_price': str(pos.avg_entry_price),
        'avg_entry_swap_rate': pos.avg_entry_swap_rate,  # Can be None
        'change_today': str(pos.change_today),
        'cost_basis': str(pos.cost_basis),
        'current_price': str(pos.current_price),
        'exchange': str(pos.exchange),
        'lastday_price': str(pos.lastday_price),
        'market_value': str(pos.market_value),
        'qty': str(pos.qty),
        'qty_available': str(pos.qty_available),
        'side': str(pos.side),
        'swap_rate': pos.swap_rate,  # Can be None
        'symbol': str(pos.symbol),
        'unrealized_intraday_pl': str(pos.unrealized_intraday_pl),
        'unrealized_intraday_plpc': str(pos.unrealized_intraday_plpc),
        'unrealized_pl': str(pos.unrealized_pl),
        'unrealized_plpc': str(pos.unrealized_plpc),
        'usd': pos.usd  # Can be None
   }
