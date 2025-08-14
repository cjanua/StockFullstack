# ai/agent/simple_training.py
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from ai.models.lstm import TradingLSTM


def simple_train_lstm_model(processed_data: pd.DataFrame, symbol: str, config, num_epochs=50):
    """Simplified training approach that bypasses DataLoader issues."""
    if 'close' not in processed_data.columns:
        print("Warning: 'close' column not in processed data. Skipping training.")
        return None

    print(f"Simple training for {symbol} with {len(processed_data)} samples")

    # Prepare data manually (no DataLoader)
    targets = processed_data['close'].copy()
    features = processed_data.drop(columns=['close'], errors='ignore')

    # Fill any remaining NaN values
    features = features.ffill().bfill().fillna(0)
    targets = targets.ffill().bfill()

    # Create sequences manually
    sequence_length = 60
    X, y = [], []

    for i in range(len(features) - sequence_length):
        sequence = features.iloc[i:i+sequence_length].values
        target_price = targets.iloc[i + sequence_length]
        last_price = targets.iloc[i + sequence_length - 1]

        if pd.isna(target_price) or pd.isna(last_price) or last_price == 0:
            continue

        # Simple classification
        pct_change = (target_price - last_price) / last_price
        if pct_change > 0.001:
            label = 2  # Up
        elif pct_change < -0.001:
            label = 0  # Down
        else:
            label = 1  # Hold

        X.append(sequence)
        y.append(label)

    if len(X) < 50:
        print(f"Not enough sequences for {symbol}: {len(X)}")
        return None

    # Convert to tensors
    X = np.array(X)
    y = np.array(y)

    print(f"Created {len(X)} sequences for {symbol}")
    print(f"Class distribution: {np.bincount(y)}")

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # Handle any remaining NaN values
    if torch.isnan(X_tensor).any():
        print(f"Replacing {torch.isnan(X_tensor).sum()} NaN values with zeros")
        X_tensor = torch.nan_to_num(X_tensor, nan=0.0)

    # Create model
    input_size = X_tensor.shape[-1]
    model = TradingLSTM(input_size=input_size, output_size=3)

    # Training setup
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Manual batching
    batch_size = 32
    n_batches = len(X_tensor) // batch_size

    epoch_losses = []
    model.train()

    print(f"Training with {n_batches} batches of size {batch_size}")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        valid_batches = 0

        # Shuffle data each epoch
        indices = torch.randperm(len(X_tensor))
        X_shuffled = X_tensor[indices]
        y_shuffled = y_tensor[indices]

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            batch_X = X_shuffled[start_idx:end_idx]
            batch_y = y_shuffled[start_idx:end_idx]

            optimizer.zero_grad()

            try:
                predictions = model(batch_X)
                loss = loss_function(predictions, batch_y)

                if torch.isnan(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                valid_batches += 1

            except Exception as e:
                print(f"Batch error: {e}")
                continue

        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
            epoch_losses.append(avg_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        else:
            epoch_losses.append(float('nan'))
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: No valid batches")

    # Save loss curve
    valid_losses = [e_loss for e_loss in epoch_losses if not np.isnan(e_loss)]
    if len(valid_losses) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(valid_losses) + 1), valid_losses, marker='o')
        plt.title(f'Simple Training Loss Curve for {symbol}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plot_filename = f'model_res/training/simple_training_loss_{symbol}.png'
        Path(plot_filename).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_filename)
        plt.close()
        print(f"📉 Loss curve saved to: {plot_filename}")

    model.eval()

    # Final test
    with torch.no_grad():
        test_batch = X_tensor[:10]  # Test with first 10 samples
        test_output = model(test_batch)
        print(f"Final test successful. Output shape: {test_output.shape}")

    successful_epochs = len(valid_losses)
    print(f"Simple training completed for {symbol}. Successful epochs: {successful_epochs}/{num_epochs}")

    if successful_epochs < 5:
        print(f"WARNING: Only {successful_epochs} successful epochs for {symbol}")
        return None

    return model
