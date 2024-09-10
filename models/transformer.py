import math

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    # Thanks, ChatGPT!
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices in the encoding
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices in the encoding
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is of shape [batch_size, window_size, d_model]
        # Add positional encodings to the input x
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerModel(nn.Module):
    def __init__(self, num_channels=48, window_size=40, num_outputs=5, d_model=512, num_heads=8, num_encoder_layers=6):
        super(TransformerModel, self).__init__()
        # Linear projection of ECoG input data
        self.input_projection = nn.Linear(num_channels, d_model)

        # Sinusoidal positional encoding
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=window_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Output layer for regression (predicting finger flexion values)
        self.output_layer = nn.Linear(d_model, num_outputs)

    def forward(self, x):
        # x has shape: [batch_size, window_size, num_channels]

        # Project input to the model dimension (d_model)
        x = self.input_projection(x)  # Shape: [batch_size, window_size, d_model]
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)  # Shape: [batch_size, window_size, d_model]
        # We want to predict flexion at the end of the window (i.e., the last time step)
        x = x[:, -1, :]  # Take the output at the last time step

        output = self.output_layer(x)  # Shape: [batch_size, num_outputs]
        return output

