# ai/agent/pytorch_system.py
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from ai.models.lstm import CustomTradingLoss, EnsembleLSTM, TradingLSTM
from ai.clean_data.pytorch_data import create_pytorch_dataloaders

def train_lstm_model(
    processed_data: pd.DataFrame, symbol: str, config, num_epochs=10
):
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


    # Create model based on symbol characteristics
    # if symbol in ['TSLA', 'NVDA', 'META']:  # High volatility stocks
    #     model = TradingLSTM(
    #         input_size=input_size, 
    #         hidden_size=160,  # Larger for complex patterns
    #         num_layers=1,
    #         output_size=3,
    #         dropout=0.3  # Higher dropout for volatile stocks
    #     )
    # else:
    model = TradingLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=1, 
        output_size=3,
        dropout=0.2
    )

    loss_function = CustomTradingLoss(directional_weight=0.8, magnitude_weight=0.2)

    optimizer = optim.AdamW([
        {'params': model.lstm.parameters(), 'lr': 0.001},
        {'params': model.attention_weight.parameters(), 'lr': 0.002},
        {'params': model.fc1.parameters(), 'lr': 0.001},
        {'params': model.fc2.parameters(), 'lr': 0.0005},
    ], weight_decay=0.01)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.003,
        epochs=num_epochs,
        steps_per_epoch=50,  # Approximate
        pct_start=0.3,
        anneal_strategy='cos'
    )

    
    # Create the data loader
    dataloader = create_pytorch_dataloaders(features, targets, config)
    if dataloader is None:
        print(f"Warning: Could not create dataloader for {symbol}. Skipping training.")
        return None

    
    epoch_losses = []
    directional_accuracies = []
    best_loss = float('inf')
    patience_counter = 0

    # --- Training Loop ---
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct_directions = 0
        total_samples = 0

        for batch_idx, (sequences, labels) in enumerate(dataloader):
            if len(sequences) == 0:
                continue

            optimizer.zero_grad()
            try:
                predictions = model(sequences)
                loss = loss_function(predictions, labels)

                if torch.isnan(loss):
                    continue
                
                # Backpropagate and update weights
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item() * sequences.size(0)

                _, predicted = torch.max(predictions.data, 1)
                correct_directions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue


        if total_samples > 0:
            avg_loss = epoch_loss / total_samples
            directional_accuracy = correct_directions / total_samples

            epoch_losses.append(avg_loss)
            directional_accuracies.append(directional_accuracy)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, "
                      f"Directional Accuracy: {directional_accuracy:.2%}")

                
            if patience_counter >= 40:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        

    # --- Generate and save the loss curve plot ---
    if epoch_losses:
        valid_losses = [loss for loss in epoch_losses if not np.isnan(loss)]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curve
        ax1.plot(epoch_losses, 'b-', label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Training Loss - {symbol}')
        ax1.grid(True)
        ax1.legend()
        
        # Directional accuracy curve
        ax2.plot(directional_accuracies, 'g-', label='Directional Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'Directional Accuracy - {symbol}')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plot_path = Path(f'model_res/training/enhanced_training_{symbol}.png')
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        
        print(f"📊 Training curves saved to: {plot_path}")



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


def train_ensemble_model(processed_data: pd.DataFrame, symbol: str, config, num_epochs=100):
    """Train ensemble model for improved robustness"""
    
    if 'close' not in processed_data.columns:
        return None
    
    targets = processed_data['close'].copy()
    features = processed_data.drop(columns=['close'], errors='ignore')
    
    input_size = features.shape[1]
    print(f"Training ensemble for {symbol} with {input_size} features")
    
    # Create ensemble model
    model = EnsembleLSTM(input_size=input_size, output_size=3)
    
    # Training setup
    loss_function = CustomTradingLoss(directional_weight=0.85, magnitude_weight=0.15)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Create dataloader
    dataloader = create_pytorch_dataloaders(features, targets, config)
    if dataloader is None:
        return None
    
    # Training loop (simplified for brevity)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        valid_batches = 0
        
        for sequences, labels in dataloader:
            if torch.isnan(sequences).any():
                continue
                
            optimizer.zero_grad()
            predictions = model(sequences)
            loss = loss_function(predictions, labels)
            
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                valid_batches += 1
        
        if valid_batches > 0 and epoch % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/valid_batches:.4f}")
    
    model.eval()
    return model


def perform_walk_forward_validation(data: pd.DataFrame, symbol: str, config, 
                                   train_months=12, test_months=3):
    """Walk-forward validation for robust performance assessment"""
    
    tscv = TimeSeriesSplit(n_splits=5, test_size=int(252/12 * test_months))
    validation_results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
        print(f"\nValidation Fold {fold+1}")
        
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        # Train model on fold
        model = train_lstm_model(train_data, f"{symbol}_fold{fold}", config, num_epochs=50)
        
        if model is None:
            continue
        
        # Evaluate on test set
        test_targets = test_data['close'].copy()
        test_features = test_data.drop(columns=['close'], errors='ignore')
        
        # Create sequences for testing
        from ai.clean_data.pytorch_data import create_sequences
        X_test, y_test = create_sequences(test_features, test_targets, 60)
        
        if len(X_test) > 0:
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test)
                predictions = model(X_tensor)
                _, predicted = torch.max(predictions.data, 1)
                
                accuracy = (predicted.numpy() == y_test).mean()
                validation_results.append({
                    'fold': fold,
                    'accuracy': accuracy,
                    'test_size': len(y_test)
                })
                
                print(f"Fold {fold+1} Accuracy: {accuracy:.2%}")
    
    return validation_results
