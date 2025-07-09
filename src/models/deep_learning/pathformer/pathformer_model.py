"""
Pathformer Model Implementation for Time Series Forecasting
Based on the concept of path-based attention for time series modeling
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math
import warnings
warnings.filterwarnings('ignore')

class PathAttention(nn.Module):
    """
    Path-based Attention mechanism for Pathformer
    """
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout=0.1):
        super(PathAttention, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values
        
        self.W_q = nn.Linear(d_model, d_keys * n_heads, bias=False)
        self.W_k = nn.Linear(d_model, d_keys * n_heads, bias=False)
        self.W_v = nn.Linear(d_model, d_values * n_heads, bias=False)
        self.W_o = nn.Linear(d_values * n_heads, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def compute_path_attention(self, queries, keys, values, path_length=3):
        """
        Compute path-based attention
        """
        B, H, L, D = queries.shape
        _, _, S, _ = keys.shape
        
        # Compute direct attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(D)
        
        # Compute path attention for longer sequences
        if L > path_length:
            path_scores = torch.zeros_like(scores)
            
            for i in range(L):
                for j in range(S):
                    # Compute path through intermediate points
                    if abs(i - j) <= path_length:
                        # Direct connection
                        path_scores[:, :, i, j] = scores[:, :, i, j]
                    else:
                        # Path through intermediate points
                        intermediate_points = []
                        step = (j - i) / path_length
                        
                        for k in range(1, path_length):
                            mid_point = int(i + k * step)
                            if 0 <= mid_point < L:
                                intermediate_points.append(mid_point)
                        
                        if intermediate_points:
                            # Compute path score through intermediates
                            path_score = scores[:, :, i, intermediate_points[0]]
                            for mid_point in intermediate_points[1:]:
                                path_score = path_score * scores[:, :, mid_point, j]
                            path_scores[:, :, i, j] = path_score.mean()
            
            scores = path_scores
        
        # Apply attention
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, values)
        
        return output, attention_weights
    
    def forward(self, queries, keys, values, attn_mask=None):
        """
        Forward pass for path attention
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
        
        # Apply path attention
        output, attention_weights = self.compute_path_attention(queries, keys, values)
        
        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(B, L, -1)
        output = self.W_o(output)
        
        return output, attention_weights

class PathAttentionLayer(nn.Module):
    """
    Path Attention layer
    """
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, d_ff=None, dropout=0.1):
        super(PathAttentionLayer, self).__init__()
        
        d_ff = d_ff or 4 * d_model
        self.attention = PathAttention(d_model, n_heads, d_keys, d_values, dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu
        
    def forward(self, x, attn_mask=None):
        # Path attention
        new_x, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        
        # Position-wise feed-forward
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm2(x + y)

class TemporalPathLayer(nn.Module):
    """
    Temporal Path Layer for capturing temporal dependencies
    """
    def __init__(self, d_model, n_heads, path_length=3, dropout=0.1):
        super(TemporalPathLayer, self).__init__()
        
        self.path_length = path_length
        self.attention = PathAttentionLayer(d_model, n_heads, dropout=dropout)
        self.temporal_conv = nn.Conv1d(d_model, d_model, kernel_size=path_length, padding=path_length//2)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Apply path attention
        x = self.attention(x)
        
        # Apply temporal convolution
        conv_out = self.temporal_conv(x.transpose(-1, -2)).transpose(-1, -2)
        x = x + self.dropout(conv_out)
        x = self.norm(x)
        
        return x

class PathformerModel(nn.Module):
    """
    Complete Pathformer Model
    """
    def __init__(self, input_size=1, output_size=1, d_model=512, n_heads=8, 
                 num_layers=6, path_length=3, dropout=0.1, seq_len=200):
        super(PathformerModel, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Input embedding
        self.input_embedding = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # Path attention layers
        self.path_layers = nn.ModuleList([
            TemporalPathLayer(d_model, n_heads, path_length, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, output_size)
        """
        # Input embedding
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        
        # Apply path attention layers
        for layer in self.path_layers:
            x = layer(x)
        
        # Output projection
        output = self.output_projection(x)
        
        return output

class PathformerPredictor:
    """
    Pathformer based predictor for JetX time series
    """
    def __init__(self, sequence_length: int = 200, d_model: int = 512, n_heads: int = 8,
                 num_layers: int = 6, path_length: int = 3, learning_rate: float = 0.001):
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.path_length = path_length
        self.learning_rate = learning_rate
        
        # Initialize model
        self.model = PathformerModel(
            input_size=1,
            output_size=1,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            path_length=path_length,
            seq_len=sequence_length
        )
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training state
        self.is_trained = False
        self.training_history = []
        
    def prepare_sequences(self, data: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare sequences for training
        
        Args:
            data: List of time series values
            
        Returns:
            Tuple of (sequences, targets)
        """
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            target = data[i + self.sequence_length]
            
            sequences.append(seq)
            targets.append(target)
        
        return torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1), torch.tensor(targets, dtype=torch.float32)
    
    def train(self, data: List[float], epochs: int = 100, batch_size: int = 32, 
              validation_split: float = 0.2, verbose: bool = True) -> dict:
        """
        Train the Pathformer model
        
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
        X, y = self.prepare_sequences(data)
        
        # Split into train and validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_train_loss = 0
            num_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]
                
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                # Use the last timestep prediction
                loss = self.criterion(predictions[:, -1, :].squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = total_train_loss / num_batches
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val)
                val_loss = self.criterion(val_predictions[:, -1, :].squeeze(), y_val).item()
            
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
            x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            prediction = self.model(x)
        
        return prediction[:, -1, :].item()
    
    def predict_sequence(self, sequence: List[float], steps: int = 1) -> List[float]:
        """
        Predict multiple steps ahead
        
        Args:
            sequence: Input sequence
            steps: Number of steps to predict
            
        Returns:
            List of predictions
        """
        predictions = []
        current_seq = sequence.copy()
        
        for _ in range(steps):
            pred = self.predict(current_seq)
            predictions.append(pred)
            current_seq = current_seq[1:] + [pred]
        
        return predictions
    
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
                'num_layers': self.num_layers,
                'path_length': self.path_length
            }
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        self.is_trained = checkpoint['is_trained']