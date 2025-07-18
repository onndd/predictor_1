"""
Enhanced TFT Model with JetX-specific Multi-Output Predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from .tft_model import *
from tqdm import tqdm

class JetXTFTModel(nn.Module):
    """
    JetX-enhanced TFT Model with multi-output predictions
    """
    def __init__(self, input_size: int = 1, hidden_size: int = 256, num_heads: int = 8,
                 num_layers: int = 2, dropout: float = 0.1, forecast_horizon: int = 1, threshold: float = 1.5):
        super(JetXTFTModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forecast_horizon = forecast_horizon
        self.threshold = threshold
        
        # Enhanced input projection with multiple features
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # JetX-enhanced temporal fusion decoder
        self.temporal_fusion_decoder = JetXTemporalFusionDecoder(
            hidden_size, hidden_size, num_heads, num_layers, dropout, threshold
        )
        
        # Multi-output projections
        self.value_projection = nn.Linear(hidden_size, forecast_horizon)
        self.probability_projection = nn.Linear(hidden_size, 1)
        self.confidence_projection = nn.Linear(hidden_size, 1)
        self.crash_risk_projection = nn.Linear(hidden_size, 1)
        
        # Pattern analysis layer
        self.pattern_analyzer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3),  # crash, pump, stable
            nn.Softmax(dim=-1)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-output predictions
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            mask: Attention mask
            return_attention: If True, includes attention weights in the output dict.
            
        Returns:
            Dictionary with multiple predictions
        """
        # Input projection
        x = self.input_projection(x)
        x = self.dropout(x)
        
        # JetX-enhanced temporal fusion decoder
        x, attention_weights = self.temporal_fusion_decoder(x, mask)
        
        # Use last timestep for predictions
        last_hidden = x[:, -1, :]
        
        # Multi-output predictions
        value_pred = self.value_projection(last_hidden)
        probability_pred = torch.sigmoid(self.probability_projection(last_hidden))
        confidence_pred = torch.sigmoid(self.confidence_projection(last_hidden))
        crash_risk_pred = torch.sigmoid(self.crash_risk_projection(last_hidden))
        pattern_pred = self.pattern_analyzer(last_hidden)
        
        predictions = {
            'value': value_pred,
            'probability': probability_pred,
            'confidence': confidence_pred,
            'crash_risk': crash_risk_pred,
            'pattern': pattern_pred,
        }
        
        if return_attention:
            predictions['attention_weights'] = attention_weights
            
        return predictions

from src.models.base_predictor import BasePredictor

class EnhancedTFTPredictor(BasePredictor):
    """
    Enhanced TFT predictor with JetX-specific features, inheriting from BasePredictor.
    """
    def _build_model(self, **kwargs) -> nn.Module:
        """Build the JetX-enhanced TFT model."""
        return JetXTFTModel(
            input_size=1,
            hidden_size=kwargs.get('hidden_size', 256),
            num_heads=kwargs.get('num_heads', 8),
            num_layers=kwargs.get('num_layers', 2),
            forecast_horizon=1,
            threshold=kwargs.get('threshold', 1.5)
        )

    def _create_loss_function(self, **kwargs) -> nn.Module:
        """Create the JetX-specific loss function for TFT."""
        return JetXTFTLoss(
            threshold=kwargs.get('threshold', 1.5),
            crash_weight=kwargs.get('crash_weight', 5.0)
        )

    def predict_with_confidence(self, sequence: List[float]) -> Tuple[float, float, float]:
        """
        Make a prediction with confidence metrics - required for rolling training system.
        This method is kept for compatibility with the RollingTrainer.
        """
        value, probability, confidence, _ = self.predict_with_attention(sequence)
        return value, probability, confidence

    def predict_with_attention(self, sequence: List[float]) -> Tuple[float, float, float, Optional[List[torch.Tensor]]]:
        """
        Makes a prediction and returns attention weights.
        
        Returns:
            Tuple of (value, probability, confidence, attention_weights)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if len(sequence) != self.sequence_length:
            raise ValueError(f"Sequence must have length {self.sequence_length}")
        
        try:
            self.model.eval()
            with torch.no_grad():
                x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
                predictions = self.model(x, return_attention=True)
                
                value = predictions['value'].squeeze().item()
                probability = predictions['probability'].squeeze().item()
                confidence = predictions['confidence'].squeeze().item()
                attention_weights = predictions.get('attention_weights') # Use .get for safety
                
                # Validate outputs
                import numpy as np
                if any(np.isnan([value, probability, confidence])) or any(np.isinf([value, probability, confidence])):
                    raise ValueError("Model produced invalid predictions")
                
                return float(value), float(probability), float(confidence), attention_weights
                
        except Exception as e:
            raise RuntimeError(f"Enhanced TFT prediction failed: {str(e)}")
    
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

class JetXTFTLoss(nn.Module):
    """
    JetX-specific loss function for TFT
    """
    def __init__(self, threshold: float = 1.5, crash_weight: float = 5.0, alpha: float = 0.6):
        super(JetXTFTLoss, self).__init__()
        self.threshold = threshold
        self.crash_weight = crash_weight
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate JetX-specific loss
        
        Args:
            predictions: Dictionary with model predictions
            targets: Target values
            
        Returns:
            Combined loss
        """
        # Primary value loss
        value_loss = self.mse(predictions['value'].squeeze(-1), targets)
        
        # Threshold probability loss
        threshold_targets = (targets >= self.threshold).float()
        prob_loss = self.bce(predictions['probability'].squeeze(-1), threshold_targets)
        
        # Crash risk loss (weighted)
        crash_targets = (targets < self.threshold).float()
        
        # Apply crash_weight to the BCE loss for crash predictions
        crash_weights = torch.where(crash_targets == 1, self.crash_weight, 1.0)
        weighted_crash_loss = F.binary_cross_entropy(
            predictions['crash_risk'].squeeze(-1),
            crash_targets,
            weight=crash_weights,
            reduction='mean'
        )
        
        # Confidence loss (should be high when prediction is accurate)
        with torch.no_grad():
            prediction_accuracy = 1.0 - torch.abs(predictions['value'].squeeze(-1) - targets) / targets.clamp(min=0.1)
        conf_loss = self.mse(predictions['confidence'].squeeze(-1), prediction_accuracy)
        
        # Combined loss
        total_loss = (
            self.alpha * value_loss +
            (1 - self.alpha) * 0.5 * prob_loss +
            (1 - self.alpha) * 0.3 * weighted_crash_loss +
            (1 - self.alpha) * 0.2 * conf_loss
        )
        
        return total_loss
