# ai/agent/pytorch_system.py
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit

from ai.clean_data.pytorch_data import create_pytorch_dataloaders
from ai.models.lstm import CustomTradingLoss, create_lstm


def train_lstm_model(
    processed_data: pd.DataFrame, symbol: str, config, num_epochs=10, model_type='standard'
):
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
    # model = TradingLSTM(
    #     input_size=input_size,
    #     hidden_size=128,
    #     num_layers=1,
    #     output_size=3,
    #     dropout=0.2
    # )

    model = create_lstm(input_size, model_type)

    loss_function = CustomTradingLoss(
        directional_weight=0.85,
        magnitude_weight=0.15,
        trend_weight=0.1,  # Adjusted for trend sensitivity
    )

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
    best_model_state = None

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

                if sequences.shape[1] >= 10:  # Need enough sequence length
                    # Simple trend calculation from sequence
                    price_changes = sequences[:, -1, 0] - sequences[:, -10, 0]  # 10-step price change
                    trend_direction = torch.sign(price_changes)
                else:
                    trend_direction = None

                # Calculate actual price changes for magnitude weighting
                if sequences.shape[1] >= 2:
                    actual_changes = sequences[:, -1, 0] - sequences[:, -2, 0]
                else:
                    actual_changes = None

                loss = loss_function(predictions, labels, actual_changes, trend_direction)

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
                      f"Directional Accuracy: {directional_accuracy:.2%}"
                      f"LR: {scheduler.get_last_lr()[0]:.6f}")

            max_patience = 60 if model_type == 'ensemble' else 40
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
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
            if torch.isnan(sequences).any() or torch.isinf(sequences).any():
                continue
            try:
                if hasattr(model, 'predict_with_uncertainty'):
                    predictions, uncertainty = model.predict_with_uncertainty(sequences, n_samples=5)
                    uncertainty_scores.append(uncertainty.mean().item())
                else:
                    predictions = model(sequences)

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

            from ai.clean_data.pytorch_data import create_sequences
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

