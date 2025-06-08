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
    print(f"Training model for {symbol} with {input_size} features and {len(processed_data)} samples")


    # Create the LSTM model
    # We'll make the output size 2 for our simple classification (Up or Down)
    model = TradingLSTM(input_size=input_size, output_size=3)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=10,
        factor=0.5,
    )
    
    # Create the data loader
    dataloader = create_pytorch_dataloaders(features, targets, config)
    if dataloader is None:
        print(f"Warning: Could not create dataloader for {symbol}. Skipping training.")
        return None

    if len(dataloader.dataset) < 100:
        print(f"Warning: Not enough training samples ({len(dataloader.dataset)}) for {symbol}")
        return None
    
    try:
        # Test the first batch to identify issues
        first_batch = next(iter(dataloader))
        sequences, labels = first_batch
        print(f"First batch shapes - Sequences: {sequences.shape}, Labels: {labels.shape}")
        print(f"Sequences dtype: {sequences.dtype}, Labels dtype: {labels.dtype}")
        print(f"Sequences has NaN: {torch.isnan(sequences).any()}")
        print(f"Sequences has Inf: {torch.isinf(sequences).any()}")
        print(f"Labels range: {labels.min().item()} to {labels.max().item()}")
        
        # Test model forward pass with first batch
        test_output = model(sequences)
        print(f"Model forward pass successful. Output shape: {test_output.shape}")
        print(f"Model output range: {test_output.min().item():.4f} to {test_output.max().item():.4f}")
        
    except Exception as e:
        print(f"ERROR in first batch test: {e}")
        print("DataLoader or model has fundamental issues")
        return None

    
    epoch_losses = []
    best_loss = float('inf')
    patience_counter = 0

    # --- Training Loop ---
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        tot_batches = 0
        valid_batches = 0

        for batch_idx, (sequences, labels) in enumerate(dataloader):
            tot_batches += 1
            if len(sequences) == 0:
                continue
            
            # Check for invalid values
            if torch.isnan(sequences).any():
                print(f"Epoch {epoch+1}, Batch {batch_idx}: NaN in sequences")
                continue
                
            if torch.isinf(sequences).any():
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Inf in sequences")
                continue
                
            if torch.isnan(labels).any():
                print(f"Epoch {epoch+1}, Batch {batch_idx}: NaN in labels")
                continue

            # Check tensor shapes
            if sequences.shape[0] == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Zero batch size")
                continue
                
            if sequences.shape[-1] != input_size:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Shape mismatch. Expected {input_size}, got {sequences.shape[-1]}")
                continue

            optimizer.zero_grad()
            try:
                predictions = model(sequences)
                loss = loss_function(predictions, labels)

                if torch.isnan(loss):
                    print(f"WARNING: NaN loss detected at epoch {epoch+1}")
                    continue
                    
                if torch.isinf(loss):
                    print(f"Epoch {epoch+1}, Batch {batch_idx}: Inf loss")
                    continue
                
                # Backpropagate and update weights
                loss.backward()

                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"Epoch {epoch+1}, Batch {batch_idx}: NaN gradient in {name}")
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    continue


                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                epoch_loss += loss.item()
                valid_batches += 1
                if epoch == 0 and batch_idx < 3:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss {loss.item():.4f} - SUCCESS")


            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue


        if valid_batches > 0:
            avg_epoch_loss = epoch_loss / valid_batches
            epoch_losses.append(avg_epoch_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_epoch_loss)
            
            # Early stopping
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or patience_counter >= 15:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, Valid Batches: {valid_batches}")
                
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch+1}")
                break
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: No valid batches")
            epoch_losses.append(float('nan'))

            # If first 10 epochs have no valid batches, something is fundamentally wrong
            if epoch >= 10:
                print(f"ERROR: No valid batches for {epoch+1} epochs. Training failed.")
                return None



    # --- Generate and save the loss curve plot ---
    if epoch_losses and not all(np.isnan(epoch_losses)):
        valid_losses = [loss for loss in epoch_losses if not np.isnan(loss)]
        if valid_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(valid_losses) + 1), valid_losses, marker='o')
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

    # Final validation with detailed testing
    final_valid_batches = 0
    final_total_batches = 0
    with torch.no_grad():
        for sequences, labels in dataloader:
            final_total_batches += 1
            if torch.isnan(sequences).any() or torch.isinf(sequences).any():
                continue
            try:
                predictions = model(sequences)
                if not torch.isnan(predictions).any():
                    final_valid_batches += 1
            except:
                continue
    
    print(f"Final validation for {symbol}: {final_valid_batches}/{final_total_batches} valid batches")
    
    if final_valid_batches == 0:
        print(f"WARNING: Model for {symbol} cannot process any validation batches!")
        return None
    
    if len([l for l in epoch_losses if not np.isnan(l)]) == 0:
        print(f"WARNING: Model for {symbol} had no successful training epochs!")
        return None
    
    print(f"Training completed for {symbol}. Successful epochs: {len([l for l in epoch_losses if not np.isnan(l)])}")
    return model