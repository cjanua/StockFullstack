# ai/agent/pytorch_system.py
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

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
    targets = processed_data['close']
    features = processed_data.drop(columns=['close'])

    # The number of features is the number of columns in our processed data
    if features.shape[1] < config.LSTM_INPUT_SIZE:
        print(f"Warning: Not enough features for training ({processed_data.shape[1]}). Skipping.")
        return None
    feature_subset = processed_data.iloc[:, :config.LSTM_INPUT_SIZE]

    # Create the LSTM model
    # We'll make the output size 2 for our simple classification (Up or Down)
    model = TradingLSTM(output_size=2)

    # Setup loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Create the data loader
    dataloader = create_pytorch_dataloaders(feature_subset, targets, config)
    
    epoch_losses = []

    # --- Training Loop ---
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for sequences, labels in dataloader:
            optimizer.zero_grad()
            
            # Get model predictions
            predictions = model(sequences)
            
            # Calculate loss
            loss = loss_function(predictions, labels)
            
            # Backpropagate and update weights
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        
        avg_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss) # Save the average loss for the epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")

    # --- ADD THIS: Generate and save the loss curve plot ---
    plt.figure()
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o')
    plt.title(f'Training Loss Curve for {symbol}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plot_filename = f'model_res/training/training_loss_{symbol}.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"📉 Training loss curve saved to: {plot_filename}")


    model.eval() # Set model to evaluation mode
    return model