"""
Enhanced PyTorch LSTM Model for JetX Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from tqdm.notebook import tqdm

class JetXLSTMModel(nn.Module):
    """
    Enhanced PyTorch LSTM Model with JetX-specific features
    """
    def __init__(self, input_size: int = 1, hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.2, bidirectional: bool = True, threshold: float = 1.5):
        super(JetXLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.threshold = threshold
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate output size after LSTM
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-output heads
        self.value_head = nn.Linear(hidden_size // 2, 1)
        self.probability_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        self.crash_risk_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_output_size)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Dictionary with predictions
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Self-attention
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Combine LSTM and attention outputs
        combined = lstm_out + attended_out
        
        # Use last timestep
        last_hidden = combined[:, -1, :]
        
        # Feature extraction
        features = self.feature_extractor(last_hidden)
        
        # Multi-output predictions
        value = self.value_head(features)
        probability = self.probability_head(features)
        confidence = self.confidence_head(features)
        crash_risk = self.crash_risk_head(features)
        
        return {
            'value': value,
            'probability': probability,
            'confidence': confidence,
            'crash_risk': crash_risk
        }

from src.models.base_predictor import BasePredictor

class EnhancedLSTMPredictor(BasePredictor):
    """
    Enhanced LSTM Predictor with PyTorch backend, inheriting from BasePredictor.
    """
    def __init__(self, sequence_length: int, input_size: int, learning_rate: float = 0.001, device: str = 'cpu', **kwargs):
        # Store other params needed for _build_model
        self.hidden_size = kwargs.get('hidden_size', 128)
        self.num_layers = kwargs.get('num_layers', 2)
        self.threshold = kwargs.get('threshold', 1.5)
        # Call super().__init__ which will store input_size and call _build_model
        super().__init__(sequence_length=sequence_length, input_size=input_size, learning_rate=learning_rate, device=device, **kwargs)

    def _build_model(self, input_size: int, **kwargs) -> nn.Module:
        """Build the JetX-enhanced LSTM model."""
        return JetXLSTMModel(
            input_size=input_size, # Use the passed input_size
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            threshold=self.threshold
        )

    def _create_loss_function(self, **kwargs) -> nn.Module:
        """Create the multi-output loss function for LSTM."""
        threshold = kwargs.get('threshold', 1.5)
        crash_weight = kwargs.get('crash_weight', 5.0)

        def loss_fn(predictions, targets):
            value_loss = F.mse_loss(predictions['value'].squeeze(), targets)
            prob_targets = (targets >= threshold).float()
            prob_loss = F.binary_cross_entropy(predictions['probability'].squeeze(), prob_targets)
            crash_targets = (targets < threshold).float()
            weights = torch.where(crash_targets == 1, crash_weight, 1.0)
            weighted_crash_loss = F.binary_cross_entropy(
                predictions['crash_risk'].squeeze(), crash_targets, weight=weights, reduction='mean'
            )
            with torch.no_grad():
                accuracy = 1.0 - torch.abs(predictions['value'].squeeze() - targets) / targets.clamp(min=0.1)
            conf_loss = F.mse_loss(predictions['confidence'].squeeze(), accuracy)
            
            return 0.5 * value_loss + 0.3 * prob_loss + 0.1 * weighted_crash_loss + 0.1 * conf_loss
        
        return loss_fn

    def predict_with_confidence(self, sequence: List[float]) -> Tuple[float, float, float]:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # This part needs a refactor on how prediction is done, as we need the feature extractor.
        # For now, this will likely fail as it expects raw sequence.
        # A proper fix involves passing the feature extractor or data manager.
        raise NotImplementedError("Prediction logic needs to be updated for feature-rich inputs.")

    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'is_trained': self.is_trained,
            'model_config': {
                'sequence_length': self.sequence_length,
                'input_size': self.input_size,
                'learning_rate': self.learning_rate,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'threshold': self.threshold
            }
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.is_trained = checkpoint['is_trained']

# Backward compatibility
class LSTMModel(EnhancedLSTMPredictor):
    def __init__(self, seq_length=200, n_features=1, threshold=1.5, **kwargs):
        # This is for backward compatibility, we need to pass input_size
        # We assume n_features is the input_size here.
        super().__init__(sequence_length=seq_length, input_size=n_features, threshold=threshold, **kwargs)
