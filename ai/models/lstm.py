# ai/models/lstm.py
import torch
import torch.nn as nn
from ai.config.settings import config

class TradingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, output_size=3):
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
            dropout=0.2 if num_layers > 1 else 0,
        )
        
        # Multi-head attention for pattern recognition
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True,
        )
        
        # Output layers with regularization
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.fc2 = nn.Linear(self.hidden_size // 2, output_size)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        if x.size(-1) != self.lstm.input_size:
            raise RuntimeError(f"Input feature size mismatch! Model expected {self.lstm.input_size}, but got {x.size(-1)}")

        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use final time step with regularization
        final_hidden = attn_out[:, -1, :]
        out = self.relu(self.fc1(final_hidden))
        out = self.dropout(out)
        return torch.softmax(self.fc2(out), dim=1)