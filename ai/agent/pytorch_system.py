# ai/agent/pytorch_system.py
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from ai.models.lstm import TradingLSTM
from ai.clean_data.pytorch_data import create_pytorch_dataloaders

def train_lstm_model(processed_data: pd.DataFrame, symbol: str, config, num_epochs=10):
    """
    Initializes and trains a TradingLSTM model.
    """
    # 1. Ensure target col
    if 'close' not in processed_data.columns:
        print("Warning: 'close' column not in processed data. Skipping training.")
        return None

    # 2. Prep target / close col
    targets = processed_data['close'].copy()
    features = processed_data.drop(columns=['close'], errors='ignore')

    input_size = features.shape[1]

    # Create the LSTM model
    # We'll make the output size 2 for our simple classification (Up or Down)
    model = TradingLSTM(input_size=input_size, output_size=3)

    # Setup loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Create the data loader
    dataloader = create_pytorch_dataloaders(features, targets, config)
    if dataloader is None:
        print(f"Warning: Could not create dataloader for {symbol}. Skipping training.")
        return None
    
    epoch_losses = []

    # --- Training Loop ---
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for sequences, labels in dataloader:
            if len(sequences) == 0:
                continue

            optimizer.zero_grad()
            predictions = model(sequences)
            loss = loss_function(predictions, labels)

            if torch.isnan(loss):
                print(f"WARNING: NaN loss detected at epoch {epoch+1}")
                continue
            
            # Backpropagate and update weights
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        if num_batches > 0 and epoch % 20 == 0:
            avg_epoch_loss = epoch_loss / num_batches
            epoch_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: No valid batches")
            epoch_losses.append(float('nan'))

    # --- Generate and save the loss curve plot ---
    if epoch_losses and not all(np.isnan(epoch_losses)):
        plt.figure()
        plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o')
        plt.title(f'Training Loss Curve for {symbol}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plot_filename = f'model_res/training/training_loss_{symbol}.png'
        plt_path = Path(plot_filename)
        plt_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_filename)
        plt.close()
        print(f"📉 Training loss curve saved to: {plot_filename}")

    model.eval()

    return model