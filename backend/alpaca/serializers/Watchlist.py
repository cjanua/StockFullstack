from typing import Dict, List, Union

from backend.alpaca.serializers.Asset import serialize_asset


def serialize_watchlist(watchlist) -> Dict[str, Union[str, List[Dict]]]:
    """Convert a Watchlist object to a serializable dictionary matching frontend Watchlist type
    
    Args:
        watchlist: Watchlist object containing watchlist data
        
    Returns:
        dict: Serialized watchlist data with standardized types containing:
            - account_id (str): UUID of associated account
            - id (str): UUID of watchlist
            - name (str): Name of watchlist
            - created_at (str): ISO format timestamp of creation
            - updated_at (str): ISO format timestamp of last update
            - assets (List[dict]): Optional list of serialized assets
    """
    serialized = {
        'account_id': str(watchlist.account_id),
        'id': str(watchlist.id),
        'name': str(watchlist.name),
        'created_at': watchlist.created_at.isoformat(),
        'updated_at': watchlist.updated_at.isoformat(),
    }
    
    # Handle optional assets field
    if watchlist.assets is not None:
        serialized['assets'] = [serialize_asset(asset) for asset in watchlist.assets]
    else:
        serialized['assets'] = None
        
    return serialized

# Example usage:
"""
watchlist = Watchlist(
    account_id=UUID('123e4567-e89b-12d3-a456-426614174000'),
    id=UUID('123e4567-e89b-12d3-a456-426614174001'),
    name='My Tech Stocks',
    created_at=datetime.now(),
    updated_at=datetime.now(),
    assets=[asset1, asset2]  # List of Asset objects
)

serialized_watchlist = serialize_watchlist(watchlist)
"""