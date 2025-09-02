def validate_data(df, symbol, min_rows=100):
    """
    Validate that the DataFrame has required columns and sufficient data.

    Args:
        df (pd.DataFrame): DataFrame to validate
        symbol (str): Symbol name
        min_rows (int): Minimum number of rows required

    Returns:
        bool: True if valid, False otherwise
    """
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if df.empty:
        print(f"❌ No data for {symbol}, skipping...")
        return False
    if not all(col in df.columns for col in required_columns):
        print(f"❌ Invalid columns for {symbol}, expected {required_columns}, got {list(df.columns)}, skipping...")
        return False
    if len(df) < min_rows:
        print(f"❌ Insufficient data for {symbol} ({len(df)} rows, minimum {min_rows}), skipping...")
        return False
    return True