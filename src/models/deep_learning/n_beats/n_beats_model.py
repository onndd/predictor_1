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
        
        # Create time indices on the same device as input
        device = x.device
        backcast_indices = torch.arange(self.input_size, dtype=torch.float32, device=device) / self.input_size
        forecast_indices = torch.arange(self.input_size, 2 * self.input_size, dtype=torch.float32, device=device) / self.input_size
        
        # Generate backcast and forecast
        backcast = self.basis(theta, backcast_indices)
        forecast = self.basis(theta, forecast_indices)
        
        return backcast, forecast

class JetXNBeatsBlock(nn.Module):
    """
    JetX-specific N-BEATS Block with crash pattern basis functions
    """
    def __init__(self, input_size: int, theta_size: int, basis_function: str = 'jetx_trend', 
                 hidden_size: int = 256, num_layers: int = 4, threshold: float = 1.5):
        super(JetXNBeatsBlock, self).__init__()
        
        self.input_size = input_size
        self.theta_size = theta_size
        self.basis_function = basis_function
        self.hidden_size = hidden_size
        self.threshold = threshold
        
        # Fully connected layers with dropout for better generalization
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            if i < num_layers - 1:  # No dropout before last layer
                layers.append(nn.Dropout(0.1))
        
        layers.append(nn.Linear(hidden_size, theta_size))
        self.layers = nn.Sequential(*layers)
        
        # JetX-specific basis functions
        if basis_function == 'jetx_crash':
            self.basis = self._jetx_crash_basis
        elif basis_function == 'jetx_pump':
            self.basis = self._jetx_pump_basis
        elif basis_function == 'jetx_trend':
            self.basis = self._jetx_trend_basis
        else:
            # Fallback to standard basis functions
            if basis_function == 'linear':
                self.basis = self._linear_basis
            elif basis_function == 'seasonal':
                self.basis = self._seasonal_basis
            elif basis_function == 'trend':
                self.basis = self._trend_basis
            else:
                self.basis = self._jetx_trend_basis
    
    def _jetx_crash_basis(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """JetX crash pattern basis function"""
        # Exponential decay pattern typical of JetX crashes
        crash_patterns = []
        for i in range(self.theta_size):
            decay_rate = 0.5 + i * 0.1
            pattern = torch.exp(-decay_rate * x)
            crash_patterns.append(pattern)
        
        basis = torch.stack(crash_patterns, dim=-1)
        return torch.sum(theta.unsqueeze(1) * basis.unsqueeze(0), dim=-1)
    
    def _jetx_pump_basis(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """JetX pump pattern basis function"""
        # Exponential growth pattern typical of JetX pumps
        pump_patterns = []
        for i in range(self.theta_size):
            growth_rate = 0.2 + i * 0.05
            pattern = torch.exp(growth_rate * x)
            pump_patterns.append(pattern)
        
        basis = torch.stack(pump_patterns, dim=-1)
        return torch.sum(theta.unsqueeze(1) * basis.unsqueeze(0), dim=-1)
    
    def _jetx_trend_basis(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """JetX trend basis function optimized for 1.5 threshold"""
        # Polynomial basis centered around threshold
        trend_patterns = []
        for i in range(self.theta_size):
            power = i + 1
            pattern = (x - 0.5) ** power  # Center around middle of sequence
            trend_patterns.append(pattern)
        
        basis = torch.stack(trend_patterns, dim=-1)
        return torch.sum(theta.unsqueeze(1) * basis.unsqueeze(0), dim=-1)
    
    def _linear_basis(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Linear basis function"""
        return theta.unsqueeze(1) * x.unsqueeze(0)
    
    def _seasonal_basis(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Seasonal basis function using Fourier series"""
        freqs = torch.arange(1, self.theta_size // 2 + 1, dtype=torch.float32)
        basis = torch.cat([
            torch.cos(2 * np.pi * freqs * x),
            torch.sin(2 * np.pi * freqs * x)
        ], dim=-1)
        return torch.sum(theta.unsqueeze(1) * basis.unsqueeze(0), dim=-1)
    
    def _trend_basis(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Trend basis function using polynomial expansion"""
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
        
        # Create time indices on the same device as input
        device = x.device
        backcast_indices = torch.arange(self.input_size, dtype=torch.float32, device=device) / self.input_size
        forecast_indices = torch.arange(self.input_size, 2 * self.input_size, dtype=torch.float32, device=device) / self.input_size
        
        # Generate backcast and forecast
        backcast = self.basis(theta, backcast_indices)
        forecast = self.basis(theta, forecast_indices)
        
        return backcast, forecast

class JetXNBeatsStack(nn.Module):
    """
    JetX-specific N-BEATS Stack implementation
    """
    def __init__(self, input_size: int, theta_size: int, basis_function: str = 'jetx_trend',
                 num_blocks: int = 3, hidden_size: int = 256, num_layers: int = 4, threshold: float = 1.5):
        super(JetXNBeatsStack, self).__init__()
        
        self.blocks = nn.ModuleList([
            JetXNBeatsBlock(input_size, theta_size, basis_function, hidden_size, num_layers, threshold)
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

class JetXThresholdLoss(nn.Module):
    """
    JetX-specific loss function that emphasizes threshold prediction accuracy
    """
    def __init__(self, threshold: float = 1.5, crash_weight: float = 2.0, alpha: float = 0.7):
        super(JetXThresholdLoss, self).__init__()
        self.threshold = threshold
        self.crash_weight = crash_weight
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
    
    def forward(self, predictions: dict, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate JetX-specific loss
        
        Args:
            predictions: Dictionary with value, probability, confidence, crash_risk predictions
            targets: Target values
            
        Returns:
            Combined loss
        """
        # Primary value loss
        value_loss = self.mse(predictions['value'].squeeze(), targets)
        
        # Threshold classification loss
        threshold_targets = (targets >= self.threshold).float()
        threshold_loss = self.bce(predictions['probability'].squeeze(), threshold_targets)
        
        # Crash risk loss (emphasize crashes)
        crash_targets = (targets < self.threshold).float()
        crash_loss = self.bce(predictions['crash_risk'].squeeze(), crash_targets)
        
        # Weight crash predictions higher
        crash_weight = torch.where(crash_targets == 1, self.crash_weight, 1.0)
        weighted_crash_loss = torch.mean(crash_weight * F.binary_cross_entropy(
            predictions['crash_risk'].squeeze(), crash_targets, reduction='none'))
        
        # Combine losses
        total_loss = (
            self.alpha * value_loss +
            (1 - self.alpha) * 0.5 * threshold_loss +
            (1 - self.alpha) * 0.3 * crash_loss +
            (1 - self.alpha) * 0.2 * weighted_crash_loss
        )
        
        return total_loss

class JetXPatternDetector:
    """
    JetX pattern detection utility
    """
    def __init__(self, threshold: float = 1.5):
        self.threshold = threshold
    
    def detect_patterns(self, sequence: List[float]) -> dict:
        """
        Detect JetX-specific patterns in sequence
        
        Args:
            sequence: Input sequence
            
        Returns:
            Dictionary with pattern information
        """
        seq = np.array(sequence)
        patterns = {}
        
        # Crash pattern detection
        patterns['crash_probability'] = self._detect_crash_pattern(seq)
        
        # Pump pattern detection
        patterns['pump_probability'] = self._detect_pump_pattern(seq)
        
        # Stability pattern detection
        patterns['stability_score'] = self._detect_stability_pattern(seq)
        
        # Consecutive high pattern
        patterns['consecutive_high'] = self._count_consecutive_above_threshold(seq)
        
        # Volatility pattern
        patterns['volatility_score'] = self._calculate_volatility_score(seq)
        
        return patterns
    
    def _detect_crash_pattern(self, sequence: np.ndarray) -> float:
        """Detect crash pattern likelihood"""
        if len(sequence) < 5:
            return 0.0
        
        # Look for high values followed by rapid decline
        recent = sequence[-5:]
        max_val = float(np.max(recent))
        min_val = float(np.min(recent))
        
        if max_val > 2.0 and min_val < float(self.threshold):
            crash_magnitude = max_val - min_val
            return min(1.0, crash_magnitude / 3.0)
        
        return 0.0
    
    def _detect_pump_pattern(self, sequence: np.ndarray) -> float:
        """Detect pump pattern likelihood"""
        if len(sequence) < 5:
            return 0.0
        
        # Look for rapid increase
        recent = sequence[-5:]
        if len(recent) >= 3:
            increase_rate = (recent[-1] - recent[0]) / len(recent)
            if increase_rate > 0.3:
                return min(1.0, increase_rate / 0.5)
        
        return 0.0
    
    def _detect_stability_pattern(self, sequence: np.ndarray) -> float:
        """Detect stability pattern"""
        if len(sequence) < 5:
            return 0.0
        
        # Low volatility indicates stability
        volatility = float(np.std(sequence[-10:]))
        return max(0.0, 1.0 - volatility / 0.5)
    
    def _count_consecutive_above_threshold(self, sequence: np.ndarray) -> int:
        """Count consecutive values above threshold"""
        count = 0
        for val in reversed(sequence):
            if val >= self.threshold:
                count += 1
            else:
                break
        return count
    
    def _calculate_volatility_score(self, sequence: np.ndarray) -> float:
        """Calculate volatility score"""
        if len(sequence) < 2:
            return 0.0
        
        returns = np.diff(sequence) / sequence[:-1]
        return float(np.std(returns))

class JetXNBeatsModel(nn.Module):
    """
    JetX-optimized N-BEATS Model with crash pattern detection
    """
    def __init__(self, input_size: int = 200, forecast_size: int = 1, 
                 num_stacks: int = 3, num_blocks: int = 3, hidden_size: int = 256,
                 num_layers: int = 4, basis_functions: Optional[List[str]] = None, threshold: float = 1.5):
        super(JetXNBeatsModel, self).__init__()
        
        self.input_size = input_size
        self.forecast_size = forecast_size
        self.threshold = threshold
        
        if basis_functions is None:
            basis_functions = ['jetx_crash', 'jetx_pump', 'jetx_trend']
        
        # Ensure we have enough basis functions
        while len(basis_functions) < num_stacks:
            basis_functions.append('jetx_trend')
        
        # Create JetX-specific stacks
        self.stacks = nn.ModuleList([
            JetXNBeatsStack(input_size, forecast_size, basis_functions[i], 
                           num_blocks, hidden_size, num_layers, threshold)
            for i in range(num_stacks)
        ])
        
        # Multi-output projections
        self.value_projection = nn.Linear(forecast_size * num_stacks, forecast_size)
        self.probability_projection = nn.Linear(forecast_size * num_stacks, 1)
        self.confidence_projection = nn.Linear(forecast_size * num_stacks, 1)
        
        # JetX-specific layers
        self.crash_detector = nn.Sequential(
            nn.Linear(input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.pattern_analyzer = nn.Sequential(
            nn.Linear(input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 3),  # crash, pump, stable
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass with JetX-specific outputs
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Dictionary with value, probability, confidence, crash_risk, pattern predictions
        """
        forecasts = []
        
        for stack in self.stacks:
            _, forecast = stack(x)
            forecasts.append(forecast)
        
        # Concatenate forecasts from all stacks
        combined_forecast = torch.cat(forecasts, dim=-1)
        
        # Multi-output predictions
        value_pred = self.value_projection(combined_forecast)
        probability_pred = torch.sigmoid(self.probability_projection(combined_forecast))
        confidence_pred = torch.sigmoid(self.confidence_projection(combined_forecast))
        
        # JetX-specific predictions
        crash_risk = self.crash_detector(x)
        pattern_pred = self.pattern_analyzer(x)
        
        return {
            'value': value_pred,
            'probability': probability_pred,
            'confidence': confidence_pred,
            'crash_risk': crash_risk,
            'pattern': pattern_pred
        }

class NBeatsPredictor:
    """
    N-BEATS based predictor for JetX time series
    """
    def __init__(self, sequence_length: int = 200, hidden_size: int = 256,
                 num_stacks: int = 3, num_blocks: int = 3, learning_rate: float = 0.001,
                 threshold: float = 1.5, crash_weight: float = 2.0):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.learning_rate = learning_rate
        
        # Initialize the fully-featured JetXNBeatsModel
        self.model = JetXNBeatsModel(
            input_size=sequence_length,
            forecast_size=1,
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            hidden_size=hidden_size,
            threshold=threshold
        )
        
        # Use the custom JetXThresholdLoss
        self.criterion = JetXThresholdLoss(threshold=threshold, crash_weight=crash_weight)
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
                predictions = self.model(batch_X) # Returns a dict
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = total_train_loss / num_batches
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val) # Returns a dict
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
        Make a prediction and return predicted value.
        
        Args:
            sequence: Input sequence of length sequence_length
            
        Returns:
            Predicted value
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if len(sequence) != self.sequence_length:
            raise ValueError(f"Sequence must have length {self.sequence_length}")
        
        try:
            self.model.eval()
            with torch.no_grad():
                x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
                
                # Ensure tensor is on the same device as model
                device = next(self.model.parameters()).device
                x = x.to(device)
                
                predictions = self.model(x)  # Returns a dict
                
                value = predictions['value'].squeeze().item()
                
                # Ensure value is reasonable
                if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                    raise ValueError("Model produced invalid prediction")
                
                return float(value)
                
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def predict_with_confidence(self, sequence: List[float]) -> Tuple[float, float, float]:
        """
        Make a prediction with confidence metrics.
        
        Args:
            sequence: Input sequence of length sequence_length
            
        Returns:
            tuple: (predicted_value, above_threshold_probability, confidence_score)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if len(sequence) != self.sequence_length:
            raise ValueError(f"Sequence must have length {self.sequence_length}")
        
        try:
            self.model.eval()
            with torch.no_grad():
                x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
                
                # Ensure tensor is on the same device as model
                device = next(self.model.parameters()).device
                x = x.to(device)
                
                predictions = self.model(x)  # Returns a dict
                
                value = predictions['value'].squeeze().item()
                probability = predictions['probability'].squeeze().item()
                confidence = predictions['confidence'].squeeze().item()
                
                # Validate outputs
                if any(np.isnan([value, probability, confidence])) or any(np.isinf([value, probability, confidence])):
                    raise ValueError("Model produced invalid predictions")
                
                return float(value), float(probability), float(confidence)
                
        except Exception as e:
            raise RuntimeError(f"Prediction with confidence failed: {str(e)}")
    
    def predict_next_value(self, sequence: List[float]) -> float:
        """
        Compatibility method for ensemble systems - returns only the predicted value
        
        Args:
            sequence: Input sequence
            
        Returns:
            Predicted value
        """
        return self.predict(sequence)
    
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
