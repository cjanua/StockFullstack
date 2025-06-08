# ai/clean_data/asset_grouping.py
import pandas as pd
import numpy as np

def concatenate_asset_data(symbols, processed_data, add_asset_features=True):
    """
    Concatenate data from multiple assets for group training.
    
    Args:
        symbols: List of asset symbols to combine
        processed_data: Dict of {symbol: DataFrame} with features
        add_asset_features: Whether to add asset-specific identifier features
    
    Returns:
        Combined DataFrame with all assets' data
    """
    combined_datasets = []
    
    for symbol in symbols:
        if symbol not in processed_data:
            print(f"Warning: {symbol} not found in processed_data, skipping...")
            continue
            
        asset_data = processed_data[symbol].copy()
        
        if asset_data.empty:
            print(f"Warning: {symbol} has empty data, skipping...")
            continue
        
        if add_asset_features:
            # Add asset identifier features
            asset_data = add_asset_identifier_features(asset_data, symbol, symbols)
        
        # Add a column to track which asset this data came from
        asset_data['_asset_symbol'] = symbol
        
        combined_datasets.append(asset_data)
        print(f"Added {symbol}: {len(asset_data)} samples")
    
    if not combined_datasets:
        print("ERROR: No valid datasets to combine!")
        return pd.DataFrame()
    
    # Concatenate all datasets
    combined_df = pd.concat(combined_datasets, axis=0, ignore_index=False)
    
    # Sort by timestamp to maintain temporal order
    combined_df = combined_df.sort_index()
    
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Asset distribution:")
    print(combined_df['_asset_symbol'].value_counts())
    
    # Remove the tracking column before returning (keep for debugging if needed)
    # combined_df = combined_df.drop(columns=['_asset_symbol'])
    
    return combined_df

def add_asset_identifier_features(data, symbol, all_symbols):
    """
    Add features that help the model understand which asset it's looking at.
    """
    data = data.copy()
    
    # One-hot encoding for asset type
    for asset in all_symbols:
        data[f'is_{asset.lower()}'] = (symbol == asset).astype(int)
    
    # Asset characteristics (you can customize these based on your knowledge)
    asset_characteristics = {
        # Volatility tier (0=low, 1=medium, 2=high)
        'AAPL': {'vol_tier': 1, 'sector': 'tech', 'cap_size': 'mega'},
        'MSFT': {'vol_tier': 1, 'sector': 'tech', 'cap_size': 'mega'},
        'GOOGL': {'vol_tier': 1, 'sector': 'tech', 'cap_size': 'mega'},
        'AMZN': {'vol_tier': 1, 'sector': 'tech', 'cap_size': 'mega'},
        'TSLA': {'vol_tier': 2, 'sector': 'auto', 'cap_size': 'large'},
        'NVDA': {'vol_tier': 2, 'sector': 'tech', 'cap_size': 'large'},
        'META': {'vol_tier': 2, 'sector': 'tech', 'cap_size': 'large'},
        'SPY': {'vol_tier': 0, 'sector': 'etf', 'cap_size': 'broad'},
        'QQQ': {'vol_tier': 1, 'sector': 'etf', 'cap_size': 'broad'},
        'IWM': {'vol_tier': 1, 'sector': 'etf', 'cap_size': 'small'},
    }
    
    if symbol in asset_characteristics:
        chars = asset_characteristics[symbol]
        data['volatility_tier'] = chars['vol_tier']
        
        # Sector encoding
        data['is_tech'] = (chars['sector'] == 'tech').astype(int)
        data['is_etf'] = (chars['sector'] == 'etf').astype(int)
        data['is_auto'] = (chars['sector'] == 'auto').astype(int)
        
        # Cap size encoding
        data['is_mega_cap'] = (chars['cap_size'] == 'mega').astype(int)
        data['is_large_cap'] = (chars['cap_size'] == 'large').astype(int)
        data['is_small_cap'] = (chars['cap_size'] == 'small').astype(int)
        data['is_broad_market'] = (chars['cap_size'] == 'broad').astype(int)
    else:
        # Default values for unknown assets
        data['volatility_tier'] = 1
        data['is_tech'] = 0
        data['is_etf'] = 0
        data['is_auto'] = 0
        data['is_mega_cap'] = 0
        data['is_large_cap'] = 0
        data['is_small_cap'] = 0
        data['is_broad_market'] = 0
    
    return data

def create_asset_groups():
    """
    Define asset groups for group-based training.
    """
    return {
        'mega_tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
        'high_vol_tech': ['TSLA', 'NVDA', 'META'], 
        'market_etfs': ['SPY', 'QQQ', 'IWM'],
        # Note: Removed problematic assets (XLF, XLE, GLD, TLT)
    }

def prepare_grouped_datasets(processed_data):
    """
    Prepare all grouped datasets for training.
    
    Args:
        processed_data: Dict of {symbol: DataFrame} from feature engineering
    
    Returns:
        Dict of {group_name: combined_DataFrame}
    """
    asset_groups = create_asset_groups()
    grouped_datasets = {}
    
    for group_name, symbols in asset_groups.items():
        print(f"\nPreparing group: {group_name}")
        combined_data = concatenate_asset_data(symbols, processed_data)
        
        if not combined_data.empty:
            grouped_datasets[group_name] = combined_data
            print(f"✅ {group_name}: {len(combined_data)} total samples")
        else:
            print(f"❌ {group_name}: Failed to create dataset")
    
    return grouped_datasets