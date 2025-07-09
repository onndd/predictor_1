"""
Autoformer Model Implementation for Time Series Forecasting
Based on the paper: Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math
import warnings
warnings.filterwarnings('ignore')

class AutoCorrelation(nn.Module):
    """
    Auto-Correlation mechanism for Autoformer
    """
    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        super(AutoCorrelation, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.correlation = correlation
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values
        
        self.W_q = nn.Linear(d_model, d_keys * n_heads, bias=False)
        self.W_k = nn.Linear(d_model, d_keys * n_heads, bias=False)
        self.W_v = nn.Linear(d_model, d_values * n_heads, bias=False)
        self.W_o = nn.Linear(d_values * n_heads, d_model, bias=False)
        
    def time_delay_agg_training(self, values, corr):
        """
        Time delay aggregation during training
        """
        batch, head, length, d_k = values.shape
        length = values.shape[2]
        
        # Find the delay
        init_length = length
        values = values.unsqueeze(-1)
        corr = corr.unsqueeze(-1)
        
        # Time delay aggregation
        values = values * corr
        values = values.sum(dim=2)
        
        return values
    
    def time_delay_agg_inference(self, values, corr):
        """
        Time delay aggregation during inference
        """
        batch, head, length, d_k = values.shape
        
        # Find the delay
        init_length = length
        values = values.unsqueeze(-1)
        corr = corr.unsqueeze(-1)
        
        # Time delay aggregation
        values = values * corr
        values = values.sum(dim=2)
        
        return values
    
    def forward(self, queries, keys, values, attn_mask):
        """
        Forward pass for auto-correlation
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        queries = self.W_q(queries).view(B, L, H, -1)
        keys = self.W_k(keys).view(B, S, H, -1)
        values = self.W_v(values).view(B, S, H, -1)
        
        # Transpose for attention computation
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Compute correlation
        corr = self.correlation(queries, keys)
        
        # Time delay aggregation
        if self.training:
            V = self.time_delay_agg_training(values, corr)
        else:
            V = self.time_delay_agg_inference(values, corr)
        
        # Reshape and apply output projection
        V = V.transpose(1, 2).contiguous().view(B, L, -1)
        V = self.W_o(V)
        
        return V

class AutoCorrelationLayer(nn.Module):
    """
    Auto-Correlation layer with correlation computation
    """
    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None, d_ff=None, dropout=0.1):
        super(AutoCorrelationLayer, self).__init__()
        
        d_ff = d_ff or 4 * d_model
        self.attention = AutoCorrelation(correlation, d_model, n_heads, d_keys, d_values)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu
        
    def forward(self, x, attn_mask=None):
        # Auto-correlation attention
        new_x = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        
        # Position-wise feed-forward
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm2(x + y)

class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        
    def forward(self, x):
        # Decompose series into trend and seasonal components
        trend = self.avg(x.transpose(-1, -2)).transpose(-1, -2)
        seasonal = x - trend
        return trend, seasonal

class AutoCorrelationBlock(nn.Module):
    """
    Auto-Correlation block with series decomposition
    """
    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None, d_ff=None, dropout=0.1, activation="gelu"):
        super(AutoCorrelationBlock, self).__init__()
        
        self.correlation = correlation
        self.attention = AutoCorrelationLayer(correlation, d_model, n_heads, d_keys, d_values, d_ff, dropout)
        self.decomp1 = SeriesDecomp(25)
        self.decomp2 = SeriesDecomp(25)
        
    def forward(self, x, attn_mask=None):
        # Decompose input
        trend1, seasonal1 = self.decomp1(x)
        
        # Apply auto-correlation attention to seasonal component
        seasonal2 = self.attention(seasonal1, seasonal1, seasonal1, attn_mask=attn_mask)
        
        # Decompose output
        trend2, seasonal3 = self.decomp2(seasonal2)
        
        # Combine trend and seasonal components
        output = trend1 + trend2 + seasonal3
        
        return output

class AutoformerModel(nn.Module):
    """
    Complete Autoformer Model
    """
    def __init__(self, input_size=1, output_size=1, d_model=512, n_heads=8, 
                 e_layers=2, d_layers=1, d_ff=2048, factor=5, dropout=0.1, 
                 activation='gelu', output_attention=False, distil=True, 
                 mix=True, seq_len=200, label_len=50, pred_len=1):
        super(AutoformerModel, self).__init__()
        
        # Encoding
        self.enc_embedding = nn.Linear(input_size, d_model)
        self.dec_embedding = nn.Linear(input_size, d_model)
        
        # Auto-correlation function
        def autocorrelation(queries, keys):
            """
            Simple auto-correlation computation
            """
            B, H, L, D = queries.shape
            _, _, S, _ = keys.shape
            
            # Compute correlation
            corr = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(D)
            
            # Apply softmax
            corr = F.softmax(corr, dim=-1)
            
            return corr
        
        # Encoder
        self.encoder = nn.ModuleList([
            AutoCorrelationBlock(autocorrelation, d_model, n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(e_layers)
        ])
        
        # Decoder
        self.decoder = nn.ModuleList([
            AutoCorrelationBlock(autocorrelation, d_model, n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(d_layers)
        ])
        
        # Output projection
        self.projection = nn.Linear(d_model, output_size, bias=True)
        
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        Forward pass
        
        Args:
            x_enc: Encoder input
            x_mark_enc: Encoder time features
            x_dec: Decoder input
            x_mark_dec: Decoder time features
            enc_self_mask: Encoder self-attention mask
            dec_self_mask: Decoder self-attention mask
            dec_enc_mask: Decoder-encoder attention mask
        """
        # Encoder
        enc_out = self.enc_embedding(x_enc)
        for layer in self.encoder:
            enc_out = layer(enc_out, attn_mask=enc_self_mask)
        
        # Decoder
        dec_out = self.dec_embedding(x_dec)
        for layer in self.decoder:
            dec_out = layer(dec_out, attn_mask=dec_self_mask)
        
        # Output projection
        dec_out = self.projection(dec_out)
        
        return dec_out, None

class AutoformerPredictor:
    """
    Autoformer based predictor for JetX time series
    """
    def __init__(self, sequence_length: int = 200, d_model: int = 512, n_heads: int = 8,
                 e_layers: int = 2, d_layers: int = 1, learning_rate: float = 0.001):
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.learning_rate = learning_rate
        
        # Initialize model
        self.model = AutoformerModel(
            input_size=1,
            output_size=1,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            seq_len=sequence_length,
            label_len=sequence_length // 4,
            pred_len=1
        )
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training state
        self.is_trained = False
        self.training_history = []
        
    def prepare_sequences(self, data: List[float]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare sequences for training
        
        Args:
            data: List of time series values
            
        Returns:
            Tuple of (enc_input, dec_input, enc_target, dec_target)
        """
        enc_inputs = []
        dec_inputs = []
        enc_targets = []
        dec_targets = []
        
        for i in range(len(data) - self.sequence_length - 1):
            # Encoder input
            enc_seq = data[i:i + self.sequence_length]
            enc_inputs.append(enc_seq)
            enc_targets.append(data[i + self.sequence_length])
            
            # Decoder input (use last part of encoder sequence)
            dec_seq = data[i + self.sequence_length - self.sequence_length // 4:i + self.sequence_length]
            dec_inputs.append(dec_seq)
            dec_targets.append(data[i + self.sequence_length])
        
        return (torch.tensor(enc_inputs, dtype=torch.float32).unsqueeze(-1),
                torch.tensor(dec_inputs, dtype=torch.float32).unsqueeze(-1),
                torch.tensor(enc_targets, dtype=torch.float32),
                torch.tensor(dec_targets, dtype=torch.float32))
    
    def train(self, data: List[float], epochs: int = 100, batch_size: int = 32, 
              validation_split: float = 0.2, verbose: bool = True) -> dict:
        """
        Train the Autoformer model
        
        Args:
            data: Training data
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            verbose: Whether to print training progress
            
        Returns:
            Training history
        """
        # Prepare data
        enc_inputs, dec_inputs, enc_targets, dec_targets = self.prepare_sequences(data)
        
        # Split into train and validation
        split_idx = int(len(enc_inputs) * (1 - validation_split))
        enc_train, enc_val = enc_inputs[:split_idx], enc_inputs[split_idx:]
        dec_train, dec_val = dec_inputs[:split_idx], dec_inputs[split_idx:]
        enc_target_train, enc_target_val = enc_targets[:split_idx], enc_targets[split_idx:]
        dec_target_train, dec_target_val = dec_targets[:split_idx], dec_targets[split_idx:]
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_train_loss = 0
            num_batches = 0
            
            for i in range(0, len(enc_train), batch_size):
                batch_enc = enc_train[i:i + batch_size]
                batch_dec = dec_train[i:i + batch_size]
                batch_enc_target = enc_target_train[i:i + batch_size]
                batch_dec_target = dec_target_train[i:i + batch_size]
                
                # Create dummy time features
                batch_enc_mark = torch.zeros_like(batch_enc)
                batch_dec_mark = torch.zeros_like(batch_dec)
                
                self.optimizer.zero_grad()
                predictions, _ = self.model(batch_enc, batch_enc_mark, batch_dec, batch_dec_mark)
                loss = self.criterion(predictions.squeeze(), batch_dec_target)
                loss.backward()
                self.optimizer.step()
                
                total_train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = total_train_loss / num_batches
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_predictions, _ = self.model(enc_val, torch.zeros_like(enc_val), 
                                              dec_val, torch.zeros_like(dec_val))
                val_loss = self.criterion(val_predictions.squeeze(), dec_target_val).item()
            
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {val_loss:.6f}")
        
        self.is_trained = True
        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        return self.training_history
    
    def predict(self, sequence: List[float]) -> float:
        """
        Make a prediction
        
        Args:
            sequence: Input sequence of length sequence_length
            
        Returns:
            Predicted next value
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if len(sequence) != self.sequence_length:
            raise ValueError(f"Sequence must have length {self.sequence_length}")
        
        self.model.eval()
        with torch.no_grad():
            # Prepare encoder and decoder inputs
            enc_input = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            dec_input = torch.tensor(sequence[-self.sequence_length // 4:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            
            # Create dummy time features
            enc_mark = torch.zeros_like(enc_input)
            dec_mark = torch.zeros_like(dec_input)
            
            prediction, _ = self.model(enc_input, enc_mark, dec_input, dec_mark)
        
        return prediction.item()
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'model_config': {
                'sequence_length': self.sequence_length,
                'd_model': self.d_model,
                'n_heads': self.n_heads,
                'e_layers': self.e_layers,
                'd_layers': self.d_layers
            }
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        self.is_trained = checkpoint['is_trained']