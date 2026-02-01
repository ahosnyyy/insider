"""
LSTM-Autoencoder Model
=======================
Implements the paper's architecture for insider threat detection.

Architecture (from paper):
- Encoder LSTM 1: (batch, 20, 32)
- Encoder LSTM 2: (batch, 16) - bottleneck
- RepeatVector: (batch, 20, 16)
- Decoder LSTM 1: (batch, 20, 16)
- Decoder LSTM 2: (batch, 20, 32)
- TimeDistributed Dense: (batch, 20, n_features)
"""

import torch
import torch.nn as nn
from typing import Tuple


class LSTMAutoencoder(nn.Module):
    """
    LSTM-Autoencoder for sequence reconstruction.
    
    Trained on normal behavior, anomalies show high reconstruction error.
    
    Args:
        n_features: Number of input features (12)
        hidden_dim1: First LSTM hidden size (32)
        hidden_dim2: Second LSTM hidden size / bottleneck (16)
        seq_len: Sequence length (20)
    """
    
    def __init__(
        self,
        n_features: int = 12,
        hidden_dim1: int = 32,
        hidden_dim2: int = 16,
        seq_len: int = 20
    ):
        super().__init__()
        
        self.n_features = n_features
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.seq_len = seq_len
        
        # Encoder
        self.encoder_lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim1,
            batch_first=True
        )
        self.encoder_lstm2 = nn.LSTM(
            input_size=hidden_dim1,
            hidden_size=hidden_dim2,
            batch_first=True
        )
        
        # Decoder
        self.decoder_lstm1 = nn.LSTM(
            input_size=hidden_dim2,
            hidden_size=hidden_dim1,
            batch_first=True
        )
        self.decoder_lstm2 = nn.LSTM(
            input_size=hidden_dim1,
            hidden_size=hidden_dim1,
            batch_first=True
        )
        
        # Output projection (TimeDistributed Dense equivalent)
        self.output_layer = nn.Linear(hidden_dim1, n_features)
        
        # Activation (paper uses ReLU)
        self.relu = nn.ReLU()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence to bottleneck representation.
        
        Args:
            x: Input tensor (batch, seq_len, n_features)
            
        Returns:
            Bottleneck representation (batch, hidden_dim2)
        """
        # First encoder LSTM
        x, _ = self.encoder_lstm1(x)
        x = self.relu(x)
        
        # Second encoder LSTM - only take last hidden state
        _, (hidden, _) = self.encoder_lstm2(x)
        
        # hidden shape: (1, batch, hidden_dim2) -> (batch, hidden_dim2)
        return hidden.squeeze(0)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode bottleneck representation to reconstructed sequence.
        
        Args:
            z: Bottleneck tensor (batch, hidden_dim2)
            
        Returns:
            Reconstructed sequence (batch, seq_len, n_features)
        """
        # Repeat vector: (batch, hidden_dim2) -> (batch, seq_len, hidden_dim2)
        z = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # First decoder LSTM
        x, _ = self.decoder_lstm1(z)
        x = self.relu(x)
        
        # Second decoder LSTM
        x, _ = self.decoder_lstm2(x)
        x = self.relu(x)
        
        # Output projection for each timestep
        output = self.output_layer(x)
        
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: encode then decode.
        
        Args:
            x: Input tensor (batch, seq_len, n_features)
            
        Returns:
            Reconstructed sequence (batch, seq_len, n_features)
        """
        z = self.encode(x)
        return self.decode(z)
    
    def get_reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Calculate reconstruction error (MSE) for each sample.
        
        Args:
            x: Input tensor (batch, seq_len, n_features)
            reduction: 'mean' for per-sample error, 'none' for full tensor
            
        Returns:
            Reconstruction error per sample (batch,) or full tensor
        """
        x_reconstructed = self.forward(x)
        
        if reduction == 'mean':
            # Mean squared error per sample
            return ((x - x_reconstructed) ** 2).mean(dim=(1, 2))
        else:
            return (x - x_reconstructed) ** 2


def create_model(config: dict) -> LSTMAutoencoder:
    """Create model from config."""
    return LSTMAutoencoder(
        n_features=config['model']['n_features'],
        hidden_dim1=config['model']['lstm_units'][0],
        hidden_dim2=config['model']['lstm_units'][1],
        seq_len=config['model']['lookback']
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    model = LSTMAutoencoder(n_features=12, hidden_dim1=32, hidden_dim2=16, seq_len=20)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(32, 20, 12)  # batch=32, seq=20, features=12
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test reconstruction error
    error = model.get_reconstruction_error(x)
    print(f"Reconstruction error shape: {error.shape}")
    print(f"Mean error: {error.mean().item():.4f}")
