# ai/agent/ensemble_training.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ai.models.lstm import TradingLSTM


class EnsembleModel(nn.Module):
    def __init__(self, input_size, num_models=3):
        super().__init__()
        self.input_size = input_size
        self.models = nn.ModuleList([
            TradingLSTM(input_size=input_size, hidden_size=128, output_size=3),
            TradingLSTM(input_size=input_size, hidden_size=96, output_size=3),
            TradingLSTM(input_size=input_size, hidden_size=160, output_size=3)
        ])
        self.weights = nn.Parameter(torch.ones(num_models) / num_models)

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))

        # Weighted average of predictions
        stacked = torch.stack(outputs, dim=0)
        weights = torch.softmax(self.weights, dim=0)
        weighted_output = torch.sum(stacked * weights.view(-1, 1, 1), dim=0)

        return weighted_output

def train_ensemble_model(processed_data, symbol, config, num_epochs=50):
    """Train ensemble model with multiple architectures."""
    if 'close' not in processed_data.columns:
        return None

    targets = processed_data['close'].copy()
    features = processed_data.drop(columns=['close'], errors='ignore')

    # Data preparation (same as before)
    features = features.ffill().bfill().fillna(0)
    targets = targets.ffill().bfill()

    sequence_length = 60
    X, y = [], []

    for i in range(len(features) - sequence_length):
        sequence = features.iloc[i:i+sequence_length].values
        target_price = targets.iloc[i + sequence_length]
        last_price = targets.iloc[i + sequence_length - 1]

        if pd.isna(target_price) or pd.isna(last_price) or last_price == 0:
            continue

        pct_change = (target_price - last_price) / last_price
        if pct_change > 0.001:
            label = 2
        elif pct_change < -0.001:
            label = 0
        else:
            label = 1

        X.append(sequence)
        y.append(label)

    if len(X) < 50:
        return None

    X = np.array(X)
    y = np.array(y)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    if torch.isnan(X_tensor).any():
        X_tensor = torch.nan_to_num(X_tensor, nan=0.0)

    # Create ensemble model
    input_size = X_tensor.shape[-1]
    ensemble_model = EnsembleModel(input_size=input_size, num_models=3)

    # Training setup
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=0.001)

    batch_size = 32
    n_batches = len(X_tensor) // batch_size

    print(f"Training ensemble for {symbol} with {n_batches} batches")

    ensemble_model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        valid_batches = 0

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
                predictions = ensemble_model(batch_X)
                loss = loss_function(predictions, batch_y)

                if torch.isnan(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(ensemble_model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                valid_batches += 1

            except Exception:
                continue

        if valid_batches > 0 and epoch % 10 == 0:
            avg_loss = epoch_loss / valid_batches
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    ensemble_model.eval()
    return ensemble_model
