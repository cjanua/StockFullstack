# ai/agent/pytorch_system

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit
from clean_data.pytorch_data import create_pytorch_dataloaders
from models.lstm import CustomTradingLoss, create_lstm
import warnings

# Suppress common warnings
warnings.filterwarnings("ignore", message="redis-py works best with hiredis")
warnings.filterwarnings("ignore", message="Can't initialize amdsmi")

def create_lr_scheduler(optimizer, config, dataloader_length=None):
    """Create a learning rate scheduler based on configuration."""
    scheduler_type = getattr(config, 'LR_SCHEDULER_TYPE', 'ReduceLROnPlateau')
    
    if scheduler_type == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=getattr(config, 'LR_SCHEDULER_FACTOR', 0.5),
            patience=getattr(config, 'LR_SCHEDULER_PATIENCE', 10),
            verbose=True,
            min_lr=getattr(config, 'LR_SCHEDULER_MIN_LR', 1e-7),
            cooldown=getattr(config, 'LR_SCHEDULER_COOLDOWN', 5),
            threshold=0.01,
            threshold_mode='rel'
        )
    elif scheduler_type == "OneCycleLR" and dataloader_length:
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.003,
            epochs=config.NUM_EPOCHS,
            steps_per_epoch=dataloader_length,
            pct_start=0.3,
            anneal_strategy='cos'
        )
    elif scheduler_type == "StepLR":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,  # Reduce every 30 epochs
            gamma=0.5      # Reduce by half
        )
    else:
        # Default to ReduceLROnPlateau
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True,
            min_lr=1e-7,
            cooldown=5
        )

def train_lstm_model(
    processed_data: pd.DataFrame, symbol: str, config, num_epochs=10, model_type='standard'
):
    print(f"cuda.is_available(): {torch.cuda.is_available()}")
    """Initializes and trains a TradingLSTM model."""
    # 1. Ensure target col
    if 'close' not in processed_data.columns:
        print("Warning: 'close' column not in processed data. Skipping training.")
        return None
    # 2. Prep target / close col
    targets = processed_data['close'].copy()
    features = processed_data.drop(columns=['close'], errors='ignore')
    input_size = features.shape[1]
    print(f"Training model for {symbol} with {input_size} features and {len(processed_data)} samples")
    model = create_lstm(input_size, model_type)
    loss_function = CustomTradingLoss(
        directional_weight=0.85,
        magnitude_weight=0.15,
        trend_weight=0.1,  # Adjusted for trend sensitivity
    )
    if num_epochs == 0:
        return model
    if model_type == 'ensemble':
        # Different learning rates for ensemble components
        optimizer = optim.AdamW([
            {'params': model.models.parameters(), 'lr': 0.001},
            {'params': model.meta_learner.parameters(), 'lr': 0.002},
            {'params': model.ensemble_weights, 'lr': 0.005},
        ], weight_decay=0.01)
    else:
        # Optimal learning rate range: 3e-4 to 1e-3 (research finding)
        optimizer = optim.AdamW([
            {'params': model.lstm.parameters(), 'lr': 0.0008},  # Slightly lower for LSTM
            {'params': model.attention_weight.parameters(), 'lr': 0.002},
            {'params': model.attention_projection.parameters(), 'lr': 0.002},
            {'params': model.fc1.parameters(), 'lr': 0.001},
            {'params': model.fc2.parameters(), 'lr': 0.0005},
        ], weight_decay=0.01)
    # Create the data loader with larger batch size for better GPU utilization
    batch_size = 128 if torch.cuda.is_available() else 64
    dataloader = create_pytorch_dataloaders(features, targets, config, batch_size=batch_size)
    if dataloader is None:
        print(f"Warning: Could not create dataloader for {symbol}. Skipping training.")
        return None
    # Create flexible learning rate scheduler based on configuration
    scheduler = create_lr_scheduler(optimizer, config, len(dataloader))
    scheduler_type = getattr(config, 'LR_SCHEDULER_TYPE', 'ReduceLROnPlateau')
        # Device setup for GPU acceleration with improved error handling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = model.to(device)
        if device.type == 'cuda':
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Clear any existing cache
            torch.cuda.empty_cache()
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB allocated")
        else:
            print("Using CPU (CUDA not available)")
    except Exception as e:
        print(f"⚠️  Device setup warning: {e}")
        device = torch.device("cpu")
        model = model.to(device)
        print("Falling back to CPU")
    epoch_losses = []
    directional_accuracies = []
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    # --- Training Loop ---
    model.train()
    print(f"Using device (cpu or cuda): {next(model.parameters()).device}")
    
    # Initialize gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda') if (device.type == 'cuda' and torch.cuda.get_device_capability(0)[0] >= 7) else None

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct_directions = 0
        total_samples = 0
        for batch_idx, (sequences, labels) in enumerate(dataloader):
            if len(sequences) == 0:
                continue
                
            try:
                # Transfer data to GPU with non_blocking=True for asynchronous transfer
                sequences = sequences.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Verify tensors are on the same device as model
                model_device = next(model.parameters()).device
                if sequences.device != model_device:
                    sequences = sequences.to(model_device)
                if labels.device != model_device:
                    labels = labels.to(model_device)
                    
            except Exception as e:
                print(f"⚠️  Device transfer error in batch {batch_idx}: {e}")
                # Fallback: ensure everything is on the same device
                target_device = next(model.parameters()).device
                sequences = sequences.to(target_device)
                labels = labels.to(target_device)
            
            # Use zero_grad(set_to_none=True) for better performance
            optimizer.zero_grad(set_to_none=True)
            
            # Use torch.amp.autocast() for mixed precision training if on modern GPU
            try:
                with torch.amp.autocast(device_type=device.type, enabled=(scaler is not None)):
                    # Both model types only return predictions (1 value)
                    outputs = model(sequences)
                    
                    # Calculate price changes for loss function
                    if sequences.shape[1] >= 2:
                        price_changes = sequences[:, -1, 0] - sequences[:, -2, 0]  # Recent price change
                    else:
                        price_changes = None
                        
                    # Calculate trend direction
                    if sequences.shape[1] >= 10:
                        trend_changes = sequences[:, -1, 0] - sequences[:, -10, 0]  # 10-period trend
                        trend_direction = torch.sign(trend_changes)
                    else:
                        trend_direction = None
                    
                    loss = loss_function(outputs, labels, price_changes, trend_direction)

                if torch.isnan(loss):
                    continue

                if scaler:
                    scaler.scale(loss).backward()
                    # Unscale before gradient clipping
                    scaler.unscale_(optimizer)
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # Step scheduler per batch for OneCycleLR
                if scheduler_type == "OneCycleLR":
                    scheduler.step()
                
                # Calculate directional accuracy
                epoch_loss += loss.item() * len(sequences)
                predicted_direction = torch.argmax(outputs, dim=1)
                true_direction = torch.argmax(labels, dim=1) if labels.dim() > 1 else labels
                correct_directions += (predicted_direction == true_direction).sum().item()
                total_samples += len(sequences)

            except Exception as e:
                print(f"Error during model forward/backward pass: {e}")
                continue

        if total_samples > 0:
            avg_loss = epoch_loss / total_samples
            directional_accuracy = correct_directions / total_samples
            epoch_losses.append(avg_loss)
            directional_accuracies.append(directional_accuracy)
            
            # Step scheduler per epoch for ReduceLROnPlateau and StepLR
            if scheduler_type == "ReduceLROnPlateau":
                scheduler.step(avg_loss)
            elif scheduler_type == "StepLR":
                scheduler.step()
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                # Get current learning rate from optimizer
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, "
                      f"Directional Accuracy: {directional_accuracy:.2%}, "
                      f"LR: {current_lr:.6f} ({scheduler_type})")

            max_patience = 25
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in loss.")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    # --- Generate and save the loss curve plot ---
    if epoch_losses:
        valid_losses = [loss for loss in epoch_losses if not np.isnan(loss)]
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        # Loss curve
        ax1.plot(epoch_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Training Loss - {symbol} ({model_type})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        # Directional accuracy
        ax2.plot(directional_accuracies, 'g-', label='Directional Accuracy', linewidth=2)
        ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Baseline')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'Directional Accuracy - {symbol}')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        # Learning rate schedule
        ax3.plot(range(len(valid_losses)), [scheduler.get_last_lr()[0]] * len(valid_losses),
                'orange', label='Learning Rate', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        # Loss vs Accuracy correlation
        ax4.scatter(epoch_losses, directional_accuracies, alpha=0.6, c=range(len(epoch_losses)), cmap='viridis')
        ax4.set_xlabel('Loss')
        ax4.set_ylabel('Directional Accuracy')
        ax4.set_title('Loss vs Accuracy Correlation')
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = Path(f'model_res/training/enhanced_training_{symbol}_{model_type}.png')
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Enhanced training curves saved to: {plot_path}")
    model.eval()
    # Final validation with detailed testing
    final_valid_batches = 0
    final_total_batches = 0
    uncertainty_scores = []
    with torch.no_grad():
        for sequences, _labels in dataloader:
            final_total_batches += 1
            sequences = sequences.to(device)
            if torch.isnan(sequences).any() or torch.isinf(sequences).any():
                continue
            try:
                if hasattr(model, 'predict_with_uncertainty'):
                    predictions, uncertainty = model.predict_with_uncertainty(sequences, n_samples=5)
                    uncertainty_scores.append(uncertainty.mean().item())
                else:
                    predictions = model(sequences)
                    avg_uncertainty = 0
                if not torch.isnan(predictions).any():
                    final_valid_batches += 1
            except Exception as _:
                continue
    print(f"Final validation for {symbol}: {final_valid_batches}/{final_total_batches} valid batches")
    if uncertainty_scores:
        avg_uncertainty = np.mean(uncertainty_scores)
        print(f"Average prediction uncertainty: {avg_uncertainty:.4f}")
    if final_valid_batches == 0:
        print(f"WARNING: Model for {symbol} cannot process any validation batches!")
        return None
    successful_epochs = len([loss for loss in epoch_losses if not np.isnan(loss)])
    final_accuracy = directional_accuracies[-1] if directional_accuracies else 0
    print(f"Training completed for {symbol}:")
    print(f"  - Successful epochs: {successful_epochs}/{num_epochs}")
    print(f"  - Final directional accuracy: {final_accuracy:.2%}")
    print(f"  - Final loss: {epoch_losses[-1]:.4f}")
    print(f"  - Best loss: {best_loss:.4f}")
    return model
def train_ensemble_model(processed_data: pd.DataFrame, symbol: str, config, num_epochs=100):
    """Train ensemble model for improved robustness."""
    return train_lstm_model(processed_data, symbol, config, num_epochs, model_type='ensemble')
def perform_walk_forward_validation(data: pd.DataFrame, symbol: str, config,
                                   train_months=12, test_months=3):
    """Walk-forward validation for robust performance assessment."""
    tscv = TimeSeriesSplit(n_splits=5, test_size=int(252/12 * test_months))
    validation_results = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
        print(f"\nValidation Fold {fold+1}")
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        # Test both standard and ensemble models
        for model_type in ['standard', 'ensemble']:
            model = train_lstm_model(
                train_data, f"{symbol}_fold{fold}_{model_type}",
                config, num_epochs=50, model_type=model_type
            )
            if model is None:
                continue
            # Evaluate on test set
            test_targets = test_data['close'].copy()
            test_features = test_data.drop(columns=['close'], errors='ignore')
            from clean_data.pytorch_data import create_sequences
            X_test, y_test = create_sequences(test_features, test_targets, 60)
            if len(X_test) > 0:
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_test)
                    if hasattr(model, 'predict_with_uncertainty'):
                        predictions, uncertainty = model.predict_with_uncertainty(X_tensor)
                        avg_uncertainty = uncertainty.mean().item()
                    else:
                        predictions = model(X_tensor)
                        avg_uncertainty = 0
                    _, predicted = torch.max(predictions.data, 1)
                    accuracy = (predicted.numpy() == y_test).mean()
                    validation_results.append({
                        'fold': fold,
                        'model_type': model_type,
                        'accuracy': accuracy,
                        'uncertainty': avg_uncertainty,
                        'test_size': len(y_test)
                    })
                    print(f"Fold {fold+1} {model_type.capitalize()} Accuracy: {accuracy:.2%}")
                    if avg_uncertainty > 0:
                        print(f"Fold {fold+1} {model_type.capitalize()} Uncertainty: {avg_uncertainty:.4f}")
    return validation_results
