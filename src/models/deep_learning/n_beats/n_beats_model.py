"""
N-BEATS (Neural Basis Expansion Analysis for Time Series) Model Implementation
Based on the paper: N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class NBeatsBlock(nn.Module):
    """
    N-BEATS Block implementation
    """
    def __init__(self, input_size: int, theta_size: int, basis_function: str = 'linear', 
                 hidden_size: int = 256, num_layers: int = 4):
        super(NBeatsBlock, self).__init__()
        
        self.input_size = input_size
        self.theta_size = theta_size
        self.basis_function = basis_function
        self.hidden_size = hidden_size
        
        # Fully connected layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_size, theta_size))
        self.layers = nn.Sequential(*layers)
        
        # Basis function
        if basis_function == 'linear':
            self.basis = self._linear_basis
        elif basis_function == 'seasonal':
            self.basis = self._seasonal_basis
        elif basis_function == 'trend':
            self.basis = self._trend_basis
        else:
            raise ValueError(f"Unknown basis function: {basis_function}")
    
    def _linear_basis(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Linear basis function"""
        return theta.unsqueeze(1) * x.unsqueeze(0)
    
    def _seasonal_basis(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Seasonal basis function using Fourier series"""
        # Create Fourier basis
        freqs = torch.arange(1, self.theta_size // 2 + 1, dtype=torch.float32)
        basis = torch.cat([
            torch.cos(2 * np.pi * freqs * x),
            torch.sin(2 * np.pi * freqs * x)
        ], dim=-1)
        return torch.sum(theta.unsqueeze(1) * basis.unsqueeze(0), dim=-1)
    
    def _trend_basis(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Trend basis function using polynomial expansion"""
        # Create polynomial basis
        basis = torch.stack([x ** i for i in range(self.theta_size)], dim=-1)
        return torch.sum(theta.unsqueeze(1) * basis.unsqueeze(0), dim=-1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Tuple of (backcast, forecast)
        """
        theta = self.layers(x)
        
        # Create time indices
        backcast_indices = torch.arange(self.input_size, dtype=torch.float32) / self.input_size
        forecast_indices = torch.arange(self.input_size, 2 * self.input_size, dtype=torch.float32) / self.input_size
        
        # Generate backcast and forecast
        backcast = self.basis(theta, backcast_indices)
        forecast = self.basis(theta, forecast_indices)
        
        return backcast, forecast

class NBeatsStack(nn.Module):
    """
    N-BEATS Stack implementation
    """
    def __init__(self, input_size: int, theta_size: int, basis_function: str = 'linear',
                 num_blocks: int = 3, hidden_size: int = 256, num_layers: int = 4):
        super(NBeatsStack, self).__init__()
        
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, theta_size, basis_function, hidden_size, num_layers)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the stack
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Tuple of (backcast, forecast)
        """
        backcast = x
        forecast = torch.zeros_like(x)
        
        for block in self.blocks:
            block_backcast, block_forecast = block(backcast)
            backcast = backcast - block_backcast
            forecast = forecast + block_forecast
        
        return backcast, forecast

class NBeatsModel(nn.Module):
    """
    Complete N-BEATS Model implementation
    """
    def __init__(self, input_size: int = 200, forecast_size: int = 1, 
                 num_stacks: int = 3, num_blocks: int = 3, hidden_size: int = 256,
                 num_layers: int = 4, basis_functions: List[str] = None):
        super(NBeatsModel, self).__init__()
        
        self.input_size = input_size
        self.forecast_size = forecast_size
        
        if basis_functions is None:
            basis_functions = ['trend', 'seasonal', 'linear']
        
        # Ensure we have enough basis functions
        while len(basis_functions) < num_stacks:
            basis_functions.append('linear')
        
        # Create stacks
        self.stacks = nn.ModuleList([
            NBeatsStack(input_size, forecast_size, basis_functions[i], 
                       num_blocks, hidden_size, num_layers)
            for i in range(num_stacks)
        ])
        
        # Final projection layer
        self.final_projection = nn.Linear(forecast_size * num_stacks, forecast_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Forecast tensor of shape (batch_size, forecast_size)
        """
        forecasts = []
        
        for stack in self.stacks:
            _, forecast = stack(x)
            forecasts.append(forecast)
        
        # Concatenate forecasts from all stacks
        combined_forecast = torch.cat(forecasts, dim=-1)
        
        # Final projection
        final_forecast = self.final_projection(combined_forecast)
        
        return final_forecast

class NBeatsPredictor:
    """
    N-BEATS based predictor for JetX time series
    """
    def __init__(self, sequence_length: int = 200, hidden_size: int = 256, 
                 num_stacks: int = 3, num_blocks: int = 3, learning_rate: float = 0.001):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.learning_rate = learning_rate
        
        # Initialize model
        self.model = NBeatsModel(
            input_size=sequence_length,
            forecast_size=1,
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            hidden_size=hidden_size
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
        
        return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
    
    def train(self, data: List[float], epochs: int = 100, batch_size: int = 32, 
              validation_split: float = 0.2, verbose: bool = True) -> dict:
        """
        Train the N-BEATS model
        
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
                predictions = self.model(batch_X).squeeze()
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = total_train_loss / num_batches
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val).squeeze()
                val_loss = self.criterion(val_predictions, y_val).item()
            
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
            x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            prediction = self.model(x).item()
        
        return prediction
    
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
                'hidden_size': self.hidden_size,
                'num_stacks': self.num_stacks,
                'num_blocks': self.num_blocks
            }
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        self.is_trained = checkpoint['is_trained']