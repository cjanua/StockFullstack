# ai/clean_data/preprocessing.py
import numpy as np
import pandas as pd
from typing import List

def split_data_by_periods(data: pd.DataFrame, n_periods: int) -> List[pd.DataFrame]:
    """
    Splits a DataFrame into a number of periods for parallel processing.
    
    Args:
        data: The DataFrame to split.
        n_periods: The number of smaller DataFrames to create.
        
    Returns:
        A list of DataFrames.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' must be a pandas DataFrame.")
    if n_periods <= 0:
        raise ValueError("'n_periods' must be a positive integer.")

    # np.array_split is perfect for this, as it handles cases where
    # the DataFrame doesn't divide equally into n_periods.
    return np.array_split(data, n_periods)