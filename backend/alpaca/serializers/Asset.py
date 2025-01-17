from typing import Dict, List
from uuid import UUID
from alpaca.trading.models import Asset


def serialize_asset(asset) -> Asset:
    """Convert an Asset object to a serializable dictionary matching frontend Asset type
    
    Args:
        asset: Asset object containing asset data
        
    Returns:
        dict: Serialized asset data with standardized types
    """
    return {
        'asset_class': str(asset.asset_class),
        'attributes': list(asset.attributes),  # Convert to list in case it's a set/tuple
        'easy_to_borrow': bool(asset.easy_to_borrow),
        'exchange': str(asset.exchange),
        'fractionable': bool(asset.fractionable),
        'id': str(asset.id),
        'maintenance_margin_requirement': float(asset.maintenance_margin_requirement),
        'marginable': bool(asset.marginable),
        'min_order_size': asset.min_order_size,  # Can be None
        'min_trade_increment': asset.min_trade_increment,  # Can be None
        'name': str(asset.name),
        'price_increment': asset.price_increment,  # Can be None
        'shortable': bool(asset.shortable),
        'status': str(asset.status),
        'symbol': str(asset.symbol),
        'tradable': bool(asset.tradable)
    }

# Example type definitions for reference:
class AssetClass:
    US_EQUITY = 'us_equity'

class AssetExchange:
    NYSE = 'NYSE'
    NASDAQ = 'NASDAQ'
    OTC = 'OTC'

class AssetStatus:
    ACTIVE = 'active'
    INACTIVE = 'inactive'