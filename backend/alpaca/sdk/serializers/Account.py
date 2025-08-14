from alpaca.trading.models import TradeAccount

def serialize_account(account: TradeAccount) -> dict:
   """Convert TradeAccount object to a serializable dictionary matching frontend Account type"""
   return {
       'id': str(account.id),
       'account_number': str(account.account_number),
       'status': str(account.status),
       'currency': str(account.currency),
       'cash': str(account.cash),
       'portfolio_value': str(account.portfolio_value),
       'non_marginable_buying_power': str(account.non_marginable_buying_power),
       'accrued_fees': str(account.accrued_fees),
       'pending_transfer_in': str(account.pending_transfer_in),
       'pending_transfer_out': str(account.pending_transfer_out),
       'pattern_day_trader': bool(account.pattern_day_trader),
       'trade_suspended_by_user': bool(account.trade_suspended_by_user),
       'trading_blocked': bool(account.trading_blocked),
       'transfers_blocked': bool(account.transfers_blocked),
       'account_blocked': bool(account.account_blocked),
       'created_at': account.created_at.isoformat() if account.created_at else None,
       'shorting_enabled': bool(account.shorting_enabled),
       'long_market_value': str(account.long_market_value),
       'short_market_value': str(account.short_market_value),
       'equity': str(account.equity),
       'last_equity': str(account.last_equity),
       'multiplier': str(account.multiplier),
       'buying_power': str(account.buying_power),
       'initial_margin': str(account.initial_margin),
       'maintenance_margin': str(account.maintenance_margin),
       'sma': str(account.sma),
       'daytrade_count': int(account.daytrade_count),
       'last_maintenance_margin': str(account.last_maintenance_margin),
       'daytrading_buying_power': str(account.daytrading_buying_power),
       'regt_buying_power': str(account.regt_buying_power),
       'options_buying_power': str(account.options_buying_power),
       'options_approved_level': str(account.options_approved_level),
       'options_trading_level': str(account.options_trading_level)
   }
