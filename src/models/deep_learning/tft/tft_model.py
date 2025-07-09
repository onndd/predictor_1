"""
Temporal Fusion Transformer (TFT) Model Implementation
Based on the paper: Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network for TFT
    """
    def __init__(self, input_sizes: List[int], hidden_size: int, dropout: float = 0.1):
        super(VariableSelectionNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        
        # Feature linear layers
        self.feature_linear_layers = nn.ModuleList([
            nn.Linear(size, hidden_size) for size in input_sizes
        ])
        
        # Variable selection weights
        self.variable_selection_weights = nn.Linear(hidden_size, len(input_sizes))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            inputs: List of input tensors
            
        Returns:
            Selected and transformed features
        """
        # Transform each input
        transformed_inputs = []
        for i, (input_tensor, linear_layer) in enumerate(zip(inputs, self.feature_linear_layers)):
            transformed = linear_layer(input_tensor)
            transformed_inputs.append(transformed)
        
        # Stack transformed inputs
        transformed_inputs = torch.stack(transformed_inputs, dim=-1)
        
        # Calculate variable selection weights
        flat_inputs = transformed_inputs.mean(dim=-1)
        sparse_weights = F.softmax(self.variable_selection_weights(flat_inputs), dim=-1)
        sparse_weights = self.dropout(sparse_weights)
        
        # Apply variable selection
        processed_inputs = torch.sum(transformed_inputs * sparse_weights.unsqueeze(1), dim=-1)
        
        return processed_inputs

class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention for TFT
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super(InterpretableMultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear transformations
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Scaled dot-product attention
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear transformation
        output = self.w_o(output)
        
        return output, attention_weights

class TemporalVariableSelection(nn.Module):
    """
    Temporal Variable Selection Network
    """
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super(TemporalVariableSelection, self).__init__()
        
        self.hidden_size = hidden_size
        
        # GRU for temporal processing
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        
        # Variable selection
        self.variable_selection = VariableSelectionNetwork([hidden_size], hidden_size, dropout)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Processed temporal features
        """
        # GRU processing
        gru_output, _ = self.gru(x)
        
        # Variable selection
        processed = self.variable_selection([gru_output])
        
        return self.dropout(processed)

class TemporalFusionDecoder(nn.Module):
    """
    Temporal Fusion Decoder
    """
    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 8, 
                 num_layers: int = 2, dropout: float = 0.1):
        super(TemporalFusionDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Temporal variable selection
        self.temporal_variable_selection = TemporalVariableSelection(input_size, hidden_size, dropout)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            InterpretableMultiHeadAttention(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Position-wise feed-forward networks
        self.feed_forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, hidden_size)
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_layers * 2)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input tensor
            mask: Attention mask
            
        Returns:
            Tuple of (output, attention_weights_list)
        """
        # Temporal variable selection
        x = self.temporal_variable_selection(x)
        
        attention_weights_list = []
        
        for i, (attention, feed_forward) in enumerate(zip(self.attention_layers, self.feed_forward_layers)):
            # Self-attention
            attn_output, attn_weights = attention(x, x, x, mask)
            attention_weights_list.append(attn_weights)
            
            # Residual connection and layer norm
            x = self.layer_norms[i * 2](x + self.dropout(attn_output))
            
            # Feed-forward
            ff_output = feed_forward(x)
            
            # Residual connection and layer norm
            x = self.layer_norms[i * 2 + 1](x + self.dropout(ff_output))
        
        return x, attention_weights_list

class TFTModel(nn.Module):
    """
    Complete Temporal Fusion Transformer Model
    """
    def __init__(self, input_size: int = 1, hidden_size: int = 256, num_heads: int = 8,
                 num_layers: int = 2, dropout: float = 0.1, forecast_horizon: int = 1):
        super(TFTModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forecast_horizon = forecast_horizon
        
        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Temporal fusion decoder
        self.temporal_fusion_decoder = TemporalFusionDecoder(
            hidden_size, hidden_size, num_heads, num_layers, dropout
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, forecast_horizon)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            mask: Attention mask
            
        Returns:
            Tuple of (predictions, attention_weights)
        """
        # Input projection
        x = self.input_projection(x)
        x = self.dropout(x)
        
        # Temporal fusion decoder
        x, attention_weights = self.temporal_fusion_decoder(x, mask)
        
        # Output projection (use last timestep)
        predictions = self.output_projection(x[:, -1, :])
        
        return predictions, attention_weights

class TFTPredictor:
    """
    TFT based predictor for JetX time series
    """
    def __init__(self, sequence_length: int = 200, hidden_size: int = 256, 
                 num_heads: int = 8, num_layers: int = 2, learning_rate: float = 0.001):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        
        # Initialize model
        self.model = TFTModel(
            input_size=1,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            forecast_horizon=1
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
        Train the TFT model
        
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
                predictions, _ = self.model(batch_X)
                loss = self.criterion(predictions.squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = total_train_loss / num_batches
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_predictions, _ = self.model(X_val)
                val_loss = self.criterion(val_predictions.squeeze(), y_val).item()
            
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
            prediction, _ = self.model(x)
        
        return prediction.item()
    
    def predict_with_attention(self, sequence: List[float]) -> Tuple[float, torch.Tensor]:
        """
        Make a prediction with attention weights
        
        Args:
            sequence: Input sequence of length sequence_length
            
        Returns:
            Tuple of (prediction, attention_weights)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if len(sequence) != self.sequence_length:
            raise ValueError(f"Sequence must have length {self.sequence_length}")
        
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            prediction, attention_weights = self.model(x)
        
        return prediction.item(), attention_weights
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'model_config': {
                'sequence_length': self.sequence_length,
                'hidden_size': self.hidden_size,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers
            }
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        self.is_trained = checkpoint['is_trained']