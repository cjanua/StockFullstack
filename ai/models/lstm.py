# ai/models/lstm.py
import torch
import torch.nn as nn
from ai.config.settings import config

class TradingLSTM(nn.Module):
    def __init__(self, input_size=50, hidden_size=128, num_layers=1, output_size=3):
        super(TradingLSTM, self).__init__()
        
        # Core LSTM layer optimized for financial time series
        self.lstm = nn.LSTM(
            input_size=config.LSTM_INPUT_SIZE,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=0.2
        )
        
        # Multi-head attention for pattern recognition
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        
        # Output layers with regularization
        self.fc1 = nn.Linear(config.LSTM_HIDDEN_SIZE, config.LSTM_HIDDEN_SIZE // 2)
        self.fc2 = nn.Linear(config.LSTM_HIDDEN_SIZE // 2, output_size)
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