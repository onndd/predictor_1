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

# Tqdm import with fallback
try:
    from tqdm.notebook import tqdm as _tqdm
    tqdm = _tqdm
except ImportError:
    try:
        from tqdm import tqdm as _tqdm
        tqdm = _tqdm
    except ImportError:
        # Fallback tqdm implementation
        class FallbackTqdm:
            def __init__(self, iterable, desc="Processing", leave=True):
                self.iterable = iterable
                self.desc = desc
                self.leave = leave
                print(f"{desc}...")
            
            def __iter__(self):
                for i, item in enumerate(self.iterable):
                    yield item
                print(f"{self.desc} completed!")
            
            def set_description(self, desc):
                pass  # Dummy implementation
        
        tqdm = FallbackTqdm

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
        # FIX: forecast_indices should only be forecast_size length (1), not input_size
        forecast_indices = torch.arange(1, dtype=torch.float32, device=device)
        
        # Generate backcast and forecast
        backcast = self.basis(theta, backcast_indices)
        forecast = self.basis(theta, forecast_indices)
        
        return backcast, forecast

class JetXNBeatsBlock(nn.Module):
    """
    JetX-specific N-BEATS Block with crash pattern basis functions
    """
    def __init__(self, input_size: int, theta_size: Optional[int] = None, basis_function: str = 'jetx_trend', 
                 hidden_size: int = 256, num_layers: int = 4, threshold: float = 1.5, forecast_size: int = 1):
        super(JetXNBeatsBlock, self).__init__()
        
        self.input_size = input_size
        self.forecast_size = forecast_size
        
        # Güvenli theta_size hesaplama - minimum değerler garanti et
        if theta_size is None:
            if basis_function in ['jetx_crash', 'jetx_pump']:
                self.theta_size = max(8, min(16, input_size // 20))  # 8-16 arası
            elif basis_function == 'jetx_trend':
                self.theta_size = max(8, min(12, input_size // 25))  # 8-12 arası 
            elif basis_function == 'seasonal':
                self.theta_size = max(10, min(20, input_size // 15))  # 10-20 arası (çift sayı olmalı)
                if self.theta_size % 2 != 0:
                    self.theta_size += 1
            else:
                self.theta_size = max(8, min(16, input_size // 20))
        else:
            self.theta_size = max(8, theta_size)  # Minimum 8 garanti et
        
        
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
        
        layers.append(nn.Linear(hidden_size, self.theta_size))
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
        """JetX crash pattern basis function with numerical stability"""
        batch_size = theta.size(0)
        x_len = x.size(0)
        device = theta.device
        
        # Create basis patterns on the same device
        x_device = x.to(device)
        patterns = []
        for i in range(self.theta_size):
            decay_rate = 0.3 + i * 0.05
            clamped_input = torch.clamp(-decay_rate * x_device, min=-10, max=10)
            pattern = torch.exp(clamped_input)
            patterns.append(pattern)
        
        # Stack patterns: [theta_size, x_len] - FIX: transpose for correct matmul
        basis_matrix = torch.stack(patterns, dim=0).to(device)
        
        # Compute: [batch_size, x_len]
        # theta: [batch_size, theta_size]
        # basis_matrix: [theta_size, x_len]
        result = torch.matmul(theta, basis_matrix)  # [batch_size, x_len]
        
        return torch.clamp(result, min=-1e6, max=1e6)
    
    def _jetx_pump_basis(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """JetX pump pattern basis function with numerical stability"""
        batch_size = theta.size(0)
        x_len = x.size(0)
        device = theta.device
        
        # Create basis patterns on the same device
        x_device = x.to(device)
        patterns = []
        for i in range(self.theta_size):
            growth_rate = 0.1 + i * 0.02
            clamped_input = torch.clamp(growth_rate * x_device, min=-10, max=10)
            pattern = torch.exp(clamped_input)
            patterns.append(pattern)
        
        # Stack patterns: [theta_size, x_len] - FIX: transpose for correct matmul
        basis_matrix = torch.stack(patterns, dim=0).to(device)
        
        # Compute: [batch_size, x_len]
        result = torch.matmul(theta, basis_matrix)  # [batch_size, x_len]
        
        return torch.clamp(result, min=-1e6, max=1e6)
    
    def _jetx_trend_basis(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """JetX trend basis function optimized for 1.5 threshold"""
        batch_size = theta.size(0)
        x_len = x.size(0)
        device = theta.device
        
        # Create polynomial patterns on the same device
        x_device = x.to(device)
        patterns = []
        for i in range(self.theta_size):
            power = i + 1
            pattern = (x_device - 0.5) ** power
            patterns.append(pattern)
        
        # Stack patterns: [theta_size, x_len] - FIX: transpose for correct matmul
        basis_matrix = torch.stack(patterns, dim=0).to(device)
        
        # Compute: [batch_size, x_len]
        result = torch.matmul(theta, basis_matrix)  # [batch_size, x_len]
        
        return result
    
    def _linear_basis(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Linear basis function"""
        batch_size = theta.size(0)
        x_len = x.size(0)
        device = theta.device
        
        # Simple linear basis: just tile theta for each x position
        x_device = x.to(device)
        # theta: [batch_size, theta_size], take first theta coefficient for linear
        linear_coeff = theta[:, 0:1]  # [batch_size, 1]
        
        # Broadcast multiplication: [batch_size, x_len]
        result = linear_coeff * x_device.unsqueeze(0)  # [batch_size, x_len]
        
        return result
    
    def _seasonal_basis(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Seasonal basis function using Fourier series"""
        batch_size = theta.size(0)
        x_len = x.size(0)
        device = theta.device
        
        # Create Fourier basis on the same device
        num_freqs = self.theta_size // 2
        freqs = torch.arange(1, num_freqs + 1, dtype=torch.float32, device=device)
        x_device = x.to(device)
        
        # Create cosine and sine components
        cos_components = torch.cos(2 * np.pi * freqs.unsqueeze(1) * x_device.unsqueeze(0))  # [num_freqs, x_len]
        sin_components = torch.sin(2 * np.pi * freqs.unsqueeze(1) * x_device.unsqueeze(0))  # [num_freqs, x_len]
        
        # Concatenate: [theta_size, x_len]
        basis_matrix = torch.cat([cos_components, sin_components], dim=0)
        
        # Handle case where theta_size is odd
        if self.theta_size % 2 == 1:
            # Add one more cosine component
            extra_cos = torch.cos(2 * np.pi * (num_freqs + 1) * x_device).unsqueeze(0)
            basis_matrix = torch.cat([basis_matrix, extra_cos], dim=0)
        
        # Compute: [batch_size, x_len]
        result = torch.matmul(theta, basis_matrix)  # [batch_size, x_len]
        
        return result
    
    def _trend_basis(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Trend basis function using polynomial expansion"""
        batch_size = theta.size(0)
        x_len = x.size(0)
        device = theta.device
        
        # Create polynomial basis on the same device
        x_device = x.to(device)
        patterns = []
        for i in range(self.theta_size):
            pattern = x_device ** i  # [x_len]
            patterns.append(pattern)
        
        # Stack patterns: [theta_size, x_len]
        basis_matrix = torch.stack(patterns, dim=0).to(device)
        
        # Compute: [batch_size, x_len]
        result = torch.matmul(theta, basis_matrix)  # [batch_size, x_len]
        
        return result
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Tuple of (backcast, forecast)
        """
        batch_size = x.size(0)
        device = x.device
        
        # Forward through layers
        theta = self.layers(x)  # Shape: (batch_size, theta_size)
        
        # Create time indices normalized to [0, 1]
        backcast_indices = torch.arange(self.input_size, dtype=torch.float32, device=device) / self.input_size
        forecast_indices = torch.arange(self.forecast_size, dtype=torch.float32, device=device)
        
        # Generate backcast and forecast for each batch
        backcast = self.basis(theta, backcast_indices)  # Shape: (batch_size, input_size)
        forecast = self.basis(theta, forecast_indices)  # Shape: (batch_size, forecast_size)
        
        # Ensure correct output shapes
        if backcast.dim() == 1:
            backcast = backcast.unsqueeze(0).expand(batch_size, -1)
        if forecast.dim() == 1:
            forecast = forecast.unsqueeze(0).expand(batch_size, -1)
            
        return backcast, forecast

class JetXNBeatsStack(nn.Module):
    """
    JetX-specific N-BEATS Stack implementation
    """
    def __init__(self, input_size: int, theta_size: Optional[int] = None, basis_function: str = 'jetx_trend',
                 num_blocks: int = 3, hidden_size: int = 256, num_layers: int = 4, threshold: float = 1.5, forecast_size: int = 1):
        super(JetXNBeatsStack, self).__init__()
        
        self.forecast_size = forecast_size
        self.blocks = nn.ModuleList([
            JetXNBeatsBlock(input_size, theta_size, basis_function, hidden_size, num_layers, threshold, forecast_size)
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
        # FIX: forecast should be (batch_size, forecast_size) not (batch_size, input_size)
        forecast = torch.zeros(x.size(0), self.forecast_size, device=x.device, dtype=x.dtype)
        
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
        # FIX: The forecast should have forecast_size as its last dimension, not input_size.
        # The forecast size for this block is implicitly 1.
        # The forecast size for this block is implicitly 1.
        forecast = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        
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
        # Loss functions are instantiated in forward pass to guarantee statelessness
    
    def forward(self, predictions: dict, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate JetX-specific loss
        
        Args:
            predictions: Dictionary with value, probability, confidence, crash_risk predictions
            targets: Target values
            
        Returns:
            Combined loss
        """
        # Instantiate loss functions inside forward to ensure they are stateless per call
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()

        # Primary value loss
        value_loss = mse_loss(predictions['value'].squeeze(), targets)
        
        # Threshold classification loss
        threshold_targets = (targets >= self.threshold).float()
        threshold_loss = bce_loss(predictions['probability'].squeeze(), threshold_targets)
        
        # Crash risk loss (emphasize crashes)
        crash_targets = (targets < self.threshold).float()
        
        # Weight crash predictions higher
        crash_weight = torch.where(crash_targets == 1, self.crash_weight, 1.0)
        
        # Calculate weighted BCE loss
        weighted_crash_loss = torch.mean(crash_weight * F.binary_cross_entropy(
            predictions['crash_risk'].squeeze(), crash_targets, reduction='none'))
        
        # Combine losses (simplified and corrected)
        total_loss = (
            self.alpha * value_loss +
            (1 - self.alpha) * 0.5 * threshold_loss +
            (1 - self.alpha) * 0.5 * weighted_crash_loss
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
            JetXNBeatsStack(input_size, None, basis_functions[i], 
                           num_blocks, hidden_size, num_layers, threshold, forecast_size)
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

from src.models.base_predictor import BasePredictor

class NBeatsPredictor(BasePredictor):
    """
    N-BEATS based predictor for JetX time series, inheriting from BasePredictor.
    This version adds a feature projection layer to handle multivariate inputs.
    """
    def __init__(self, model_sequence_length: int, input_size: int, learning_rate: float, device: str = 'cpu', **kwargs):
        # Store hyperparameters before calling super().__init__ because _build_model needs them
        self.hidden_size = kwargs.get('hidden_size', 512)
        self.num_stacks = kwargs.get('num_stacks', 4)
        self.num_blocks = kwargs.get('num_blocks', 4)
        self.threshold = kwargs.get('threshold', 1.5)
        self.crash_weight = kwargs.get('crash_weight', 2.0)
        
        # The sequence length the N-BEATS model itself will see
        self.model_sequence_length = model_sequence_length

        # This layer projects the rich feature vector at each time step to a single value,
        # creating a synthetic univariate time series that N-BEATS can process.
        self.feature_to_univariate = nn.Linear(input_size, 1).to(device)

        # The `sequence_length` for BasePredictor is the length of the sequence fed to the model.
        # The `input_size` for BasePredictor is the number of raw features before projection.
        super().__init__(sequence_length=self.model_sequence_length, input_size=input_size, learning_rate=learning_rate, device=device, **kwargs)

    def _create_loss_function(self, **kwargs) -> nn.Module:
        """
        Override the base loss function to use the JetX-specific loss,
        which is designed to handle the dictionary output of the model.
        """
        print("✅ Using JetXThresholdLoss for N-Beats.")
        return JetXThresholdLoss(
            threshold=self.threshold,
            crash_weight=self.crash_weight
        )

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Override the base optimizer creation to include the parameters of the
        feature_to_univariate projection layer. This is crucial because this layer
        is part of the computation graph but not part of `self.model`.
        Failing to include its parameters in the optimizer leads to stale gradients
        and the 'trying to backward through the graph a second time' error.
        """
        all_params = list(self.model.parameters()) + list(self.feature_to_univariate.parameters())
        print("✅ N-Beats optimizer including projection layer parameters.")
        return torch.optim.Adam(
            all_params,
            lr=self.learning_rate,
            weight_decay=1e-4
        )

    def _build_model(self, input_size: int, **kwargs) -> nn.Module:
        """
        Build the JetX-optimized N-Beats model.
        The model's internal input_size is `model_sequence_length`.
        """
        # The input_size passed here is the number of raw features, which we ignore
        # in favor of the model_sequence_length for the N-BEATS architecture.
        return JetXNBeatsModel(
            input_size=self.model_sequence_length,
            forecast_size=1,
            num_stacks=self.num_stacks,
            num_blocks=self.num_blocks,
            hidden_size=self.hidden_size,
            threshold=self.threshold
        )

    def predict_with_confidence(self, sequence: List[float]) -> Tuple[float, float, float]:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if len(sequence) != self.sequence_length:
            raise ValueError(f"Sequence must have length {self.sequence_length}")
        
        try:
            self.model.eval()
            with torch.no_grad():
                # N-Beats expects a 2D tensor: (batch_size, sequence_length)
                x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                predictions = self.model(x)
                
                value = predictions['value'].squeeze().item()
                probability = predictions['probability'].squeeze().item()
                confidence = predictions['confidence'].squeeze().item()
                
                if any(np.isnan([v]) for v in [value, probability, confidence]):
                    raise ValueError("Model produced invalid predictions (NaN)")
                
                return float(value), float(probability), float(confidence)
                
        except Exception as e:
            raise RuntimeError(f"Prediction with confidence failed: {str(e)}")

    def train(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 100, batch_size: int = 32,
              validation_split: float = 0.2, verbose: bool = True, tqdm_desc: str = "Training") -> dict:
        """
        Custom training loop for N-BEATS to handle the feature projection within the batch loop.
        This avoids the 'trying to backward through the graph a second time' error by re-creating
        the graph for each batch, which is the standard and expected way in PyTorch.
        """
        if X.dim() != 3:
            raise ValueError(f"NBeatsPredictor expects a 3D tensor, but got {X.dim()}D.")

        if len(X) == 0:
            if verbose:
                print("⚠️ Not enough data to train.")
            return {}

        # Move full dataset to device once to avoid repeated transfers
        X = X.to(self.device)
        y = y.to(self.device)

        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        if len(X_train) == 0 or len(X_val) == 0:
            if verbose:
                print("⚠️ Not enough data for training and validation split.")
            return {}

        self.scheduler.T_max = epochs
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15  # Early stopping patience
        
        train_losses, val_losses = [], []
        
        epoch_iterator = tqdm(range(epochs), desc=tqdm_desc, leave=False)
        for epoch in epoch_iterator:
            self.model.train()
            self.feature_to_univariate.train()  # Ensure projection layer is in train mode
            total_train_loss = 0
            
            # Shuffle training data each epoch
            permutation = torch.randperm(X_train.size(0))
            
            for i in range(0, len(X_train), batch_size):
                indices = permutation[i:i + batch_size]
                batch_X, batch_y = X_train[indices], y_train[indices]
                
                self.optimizer.zero_grad()
                
                # Project features for the current batch *inside* the loop
                batch_X_projected = self.feature_to_univariate(batch_X).squeeze(-1)
                
                predictions = self.model(batch_X_projected)
                loss = self.criterion(predictions, batch_y)
                
                loss.backward()
                
                # Clip gradients for both model and projection layer
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.feature_to_univariate.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / (len(X_train) / batch_size)
            
            self.model.eval()
            self.feature_to_univariate.eval()  # Ensure projection layer is in eval mode
            with torch.no_grad():
                # Project validation data for evaluation
                X_val_projected = self.feature_to_univariate(X_val).squeeze(-1)
                val_predictions = self.model(X_val_projected)
                val_loss = self.criterion(val_predictions, y_val).item()
            
            self.scheduler.step()
            
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            epoch_iterator.set_description(f"{tqdm_desc} | Epoch {epoch+1}/{epochs} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        self.is_trained = True
        self.training_history = {'train_losses': train_losses, 'val_losses': val_losses}
        return self.training_history
    
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
                'input_size': self.input_size,
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
