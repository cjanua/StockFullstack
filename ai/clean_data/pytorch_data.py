# ai/clean_data/pytorch_data.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

def create_sequences(feature_data: pd.DataFrame, target_series: pd.DataFrame, sequence_length: int, target_col='close'):
    """Creates sequences and corresponding labels for an LSTM model."""
    xs, ys = [], []

    # Ensure data is aligned and has no gaps
    feature_data, target_series = feature_data.align(target_series, join='inner', axis=0)

    for i in range(len(feature_data) - sequence_length):
        # The sequence of features
        sequence = feature_data.iloc[i: (i + sequence_length)].values
        # The price one step after the sequence ends
        target_price = target_series.iloc[i + sequence_length]
        last_price_in_sequence = target_series.iloc[i + sequence_length - 1]

        xs.append(sequence)
        # Simple classification: 1 if price went up, 0 otherwise
        ys.append(1 if target_price > last_price_in_sequence*1.001 else 0)
        
    return np.array(xs), np.array(ys)

def create_pytorch_dataloaders(features: pd.DataFrame, targets: pd.Series, config):
    """Prepares data and creates PyTorch DataLoaders."""
    X, y = create_sequences(features, targets, config.SEQUENCE_LENGTH)
    
    if len(X) == 0:
        return None # Return None if no sequences could be created

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long) # Use 'long' for CrossEntropyLoss
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return dataloader