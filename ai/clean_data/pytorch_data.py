# ai/clean_data/pytorch_data.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def create_sequences(feature_data: pd.DataFrame, target_series: pd.Series, sequence_length: int, target_col='close'):
    """Creates sequences and corresponding labels for an LSTM model."""
    xs, ys = [], []

    # Ensure data is aligned and has no gaps
    feature_data, target_series = feature_data.align(target_series, join='inner', axis=0)

    # Check if we have enough data
    if len(feature_data) < sequence_length + 1:
        print(
            "WARNING: Not enough data to create sequences."
            f"Need at least {sequence_length + 1} rows, have {len(feature_data)}")
        return np.array([]), np.array([])

    feature_data = feature_data.ffill()
    feature_data = feature_data.bfill()

    print(f"Creating sequences from {len(feature_data)} data points...")
    sequences_created = 0

    for i in range(len(feature_data) - sequence_length):
        # The sequence of features
        sequence = feature_data.iloc[i: (i + sequence_length)].values

        nan_ratio = np.isnan(sequence).sum() / sequence.size
        if nan_ratio > 0.1:
            continue

        # Fill any remaining NaNs with column means
        if np.isnan(sequence).any():
            col_means = np.nanmean(sequence, axis=0)
            nan_mask = np.isnan(sequence)
            sequence[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        # The price one step after the sequence ends
        target_price = target_series.iloc[i + sequence_length]
        last_price_in_sequence = target_series.iloc[i + sequence_length - 1]

        if pd.isna(target_price) or pd.isna(last_price_in_sequence) or last_price_in_sequence == 0:
            continue

        xs.append(sequence)
        pct_change = (target_price - last_price_in_sequence) / last_price_in_sequence

        if pct_change > 0.001:  # Up more than 0.1%
            ys.append(2)
        elif pct_change < -0.001:  # Down more than 0.1%
            ys.append(0)
        else:  # Hold
            ys.append(1)

        sequences_created += 1

    print(f"Successfully created {sequences_created} sequences")

    if sequences_created == 0:
        return np.array([]), np.array([])

    X = np.array(xs)
    y = np.array(ys)

    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(['Down', 'Hold', 'Up'], counts, strict=False))}")

    return X, y


def create_pytorch_dataloaders(
    features: pd.DataFrame, targets: pd.Series, config, batch_size=32, num_workers=4, pin_memory=False
) -> DataLoader:
    """Prepares data and creates PyTorch DataLoaders."""
    X, y = create_sequences(features, targets, config.SEQUENCE_LENGTH)

    if len(X) == 0:
        print("ERROR: No sequences created - check your data preprocessing")
        return None # Return None if no sequences could be created

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long) # Use 'long' for CrossEntropyLoss

    if torch.isnan(X_tensor).any():
        print("WARNING: NaN values found in tensor, replacing with zeros")
        X_tensor = torch.nan_to_num(X_tensor, nan=0.0)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    return dataloader
