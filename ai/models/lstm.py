# ai/models/lstm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai.config.settings import config

class TradingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, output_size=3, dropout=0.2):
        super(TradingLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = config.LSTM_HIDDEN_SIZE
        self.num_layers = config.LSTM_NUM_LAYERS

        # Core LSTM layer optimized for financial time series
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Multi-head attention for pattern recognition
        self.attention_weight = nn.Linear(hidden_size * 2, hidden_size)  # *2 for bidirectional
        self.attention_projection = nn.Linear(hidden_size, 1)


        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layers with regularization
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.fc2 = nn.Linear(self.hidden_size // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size // 2)
    
    def attention(self, lstm_output):
        """Bahdanau attention mechanism"""
        # lstm_output shape: (batch, seq_len, hidden_size * 2)
        
        # Calculate attention scores
        attention_scores = self.attention_weight(lstm_output)  # (batch, seq_len, hidden_size)
        attention_scores = torch.tanh(attention_scores)
        attention_scores = self.attention_projection(attention_scores)  # (batch, seq_len, 1)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention weights
        context = torch.sum(attention_weights * lstm_output, dim=1)  # (batch, hidden_size * 2)
        
        return context, attention_weights

    def forward(self, x):
        """Forward pass with attention mechanism"""
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention mechanism
        context, attention_weights = self.attention(lstm_out)
        
        # Feature extraction with residual connection potential
        features = self.feature_extractor(context)
        
        # Output layers with normalization
        out = F.relu(self.fc1(features))
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return F.softmax(out, dim=1)

class EnsembleLSTM(nn.Module):
    """Ensemble of LSTM models with different architectures"""
    
    def __init__(self, input_size, output_size=3):
        super(EnsembleLSTM, self).__init__()
        
        # Different model architectures
        self.models = nn.ModuleList([
            TradingLSTM(input_size, hidden_size=128, num_layers=1, dropout=0.2),
            TradingLSTM(input_size, hidden_size=96, num_layers=2, dropout=0.3),
            TradingLSTM(input_size, hidden_size=160, num_layers=1, dropout=0.15),
        ])
        
        # Learnable weights for ensemble
        self.ensemble_weights = nn.Parameter(torch.ones(len(self.models)) / len(self.models))
        
    def forward(self, x):
        """Weighted ensemble prediction"""
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Stack outputs and apply weighted average
        stacked = torch.stack(outputs, dim=0)
        weights = F.softmax(self.ensemble_weights, dim=0)
        weighted_output = torch.sum(stacked * weights.view(-1, 1, 1), dim=0)
        
        return weighted_output
    
class CustomTradingLoss(nn.Module):
    """Custom loss function optimized for directional accuracy"""
    
    def __init__(self, directional_weight=0.7, magnitude_weight=0.3):
        super(CustomTradingLoss, self).__init__()
        self.directional_weight = directional_weight
        self.magnitude_weight = magnitude_weight
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, predictions, targets, price_changes=None):
        """
        Combined loss focusing on directional accuracy
        
        Args:
            predictions: Model output (batch_size, 3) - probabilities for down/hold/up
            targets: True labels (batch_size,) - 0/1/2
            price_changes: Actual price changes for magnitude weighting
        """
        # Standard cross-entropy loss
        ce_loss = self.ce_loss(predictions, targets)
        
        if price_changes is not None:
            # Weight loss by magnitude of price change
            # Larger moves are more important to predict correctly
            magnitude_weights = torch.abs(price_changes)
            magnitude_weights = magnitude_weights / magnitude_weights.mean()  # Normalize
            
            # Calculate per-sample loss
            ce_unreduced = F.cross_entropy(predictions, targets, reduction='none')
            weighted_loss = (ce_unreduced * magnitude_weights).mean()
            
            return self.directional_weight * ce_loss + self.magnitude_weight * weighted_loss
        
        return ce_loss
