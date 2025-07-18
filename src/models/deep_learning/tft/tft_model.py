"""
Temporal Fusion Transformer (TFT) Model Implementation
Based on the paper: Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Union
import warnings
warnings.filterwarnings('ignore')

class JetXFeatureExtractor:
    """
    JetX-specific feature extraction for TFT
    """
    def __init__(self, threshold: float = 1.5):
        self.threshold = threshold
    
    def extract_features(self, sequence: List[float]) -> torch.Tensor:
        """
        Extract JetX-specific features from sequence
        
        Args:
            sequence: Input sequence
            
        Returns:
            Multi-feature tensor
        """
        seq = np.array(sequence)
        features = []
        
        # Original value
        features.append(seq)
        
        # Moving averages
        ma_5 = self._moving_average(seq, 5)
        ma_10 = self._moving_average(seq, 10)
        features.extend([ma_5, ma_10])
        
        # Volatility
        vol_5 = self._volatility(seq, 5)
        vol_10 = self._volatility(seq, 10)
        features.extend([vol_5, vol_10])
        
        # Momentum
        mom_5 = self._momentum(seq, 5)
        mom_10 = self._momentum(seq, 10)
        features.extend([mom_5, mom_10])
        
        # Above threshold ratio
        above_threshold = self._above_threshold_ratio(seq, 20)
        features.append(above_threshold)
        
        # Stack features
        feature_tensor = np.stack(features, axis=-1)
        return torch.tensor(feature_tensor, dtype=torch.float32)
    
    def _moving_average(self, sequence: np.ndarray, window: int) -> np.ndarray:
        """Calculate moving average"""
        result = np.zeros_like(sequence)
        for i in range(len(sequence)):
            start = max(0, i - window + 1)
            result[i] = np.mean(sequence[start:i+1])
        return result
    
    def _volatility(self, sequence: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling volatility"""
        result = np.zeros_like(sequence)
        for i in range(len(sequence)):
            start = max(0, i - window + 1)
            if i >= window - 1:
                returns = np.diff(sequence[start:i+1]) / sequence[start:i]
                result[i] = np.std(returns) if len(returns) > 0 else 0
            else:
                result[i] = 0
        return result
    
    def _momentum(self, sequence: np.ndarray, window: int) -> np.ndarray:
        """Calculate momentum"""
        result = np.zeros_like(sequence)
        for i in range(len(sequence)):
            if i >= window - 1:
                result[i] = (sequence[i] - sequence[i - window + 1]) / sequence[i - window + 1]
            else:
                result[i] = 0
        return result
    
    def _above_threshold_ratio(self, sequence: np.ndarray, window: int) -> np.ndarray:
        """Calculate above threshold ratio"""
        result = np.zeros_like(sequence)
        for i in range(len(sequence)):
            start = max(0, i - window + 1)
            window_seq = sequence[start:i+1]
            result[i] = np.mean(window_seq >= self.threshold)
        return result

class JetXSpecificAttention(nn.Module):
    """
    JetX-specific attention mechanism focusing on crash/pump patterns
    """
    def __init__(self, hidden_size: int, num_heads: int, threshold: float = 1.5):
        super(JetXSpecificAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.threshold = threshold
        
        # Standard attention
        self.attention = InterpretableMultiHeadAttention(hidden_size, num_heads)
        
        # JetX-specific attention weights
        self.crash_attention = nn.Linear(hidden_size, 1)
        self.pump_attention = nn.Linear(hidden_size, 1)
        self.threshold_attention = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with JetX-specific attention
        
        Args:
            x: Hidden representations
            values: Original values for pattern detection
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        # Standard attention
        attended, attention_weights = self.attention(x, x, x)
        
        # JetX-specific attention
        crash_scores = torch.sigmoid(self.crash_attention(x))
        pump_scores = torch.sigmoid(self.pump_attention(x))
        threshold_scores = torch.sigmoid(self.threshold_attention(x))
        
        # Combine attention scores based on value patterns
        value_mask = (values.unsqueeze(-1) >= self.threshold).float()
        
        # Weight attention based on crash/pump patterns
        jetx_attention = (
            crash_scores * (1 - value_mask) +  # Focus on low values for crashes
            pump_scores * value_mask +         # Focus on high values for pumps
            threshold_scores * 0.5             # Always consider threshold
        )
        
        # Apply JetX-specific attention
        jetx_attended = attended * jetx_attention
        
        return jetx_attended, jetx_attention

class JetXTemporalFusionDecoder(nn.Module):
    """
    JetX-enhanced Temporal Fusion Decoder
    """
    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 8, 
                 num_layers: int = 2, dropout: float = 0.1, threshold: float = 1.5):
        super(JetXTemporalFusionDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.threshold = threshold
        
        # Enhanced temporal variable selection
        self.temporal_variable_selection = JetXTemporalVariableSelection(
            input_size, hidden_size, dropout, threshold
        )
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            InterpretableMultiHeadAttention(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # JetX-specific attention layers
        self.jetx_attention_layers = nn.ModuleList([
            JetXSpecificAttention(hidden_size, num_heads, threshold)
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
            for _ in range(num_layers * 3)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with JetX-enhanced processing
        
        Args:
            x: Input tensor
            mask: Attention mask
            
        Returns:
            Tuple of (output, attention_weights_list)
        """
        # Enhanced temporal variable selection
        x = self.temporal_variable_selection(x)
        
        attention_weights_list = []
        
        for i, (attention, jetx_attention, feed_forward) in enumerate(
            zip(self.attention_layers, self.jetx_attention_layers, self.feed_forward_layers)
        ):
            # Standard self-attention
            attn_output, attn_weights = attention(x, x, x, mask)
            attention_weights_list.append(attn_weights)
            
            # Residual connection and layer norm
            x = self.layer_norms[i * 3](x + self.dropout(attn_output))
            
            # JetX-specific attention
            jetx_output, jetx_weights = jetx_attention(x, x[:, :, 0])  # Use first feature as values
            
            # Residual connection and layer norm
            x = self.layer_norms[i * 3 + 1](x + self.dropout(jetx_output))
            
            # Feed-forward
            ff_output = feed_forward(x)
            
            # Residual connection and layer norm
            x = self.layer_norms[i * 3 + 2](x + self.dropout(ff_output))
        
        return x, attention_weights_list

class JetXTemporalVariableSelection(nn.Module):
    """
    JetX-enhanced Temporal Variable Selection Network
    """
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1, threshold: float = 1.5):
        super(JetXTemporalVariableSelection, self).__init__()
        
        self.hidden_size = hidden_size
        self.threshold = threshold
        
        # Enhanced GRU for temporal processing
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        
        # Variable selection with JetX-specific features
        self.variable_selection = VariableSelectionNetwork(
            [hidden_size * 2], hidden_size, dropout
        )
        
        # JetX-specific pattern recognition
        self.pattern_recognition = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3),  # crash, pump, stable
            nn.Softmax(dim=-1)
        )
        
        # Gating layer for patterns - fix tensor dimensions
        self.pattern_gate = nn.Sequential(
            nn.Linear(3, hidden_size), # 3 patterns to hidden_size
            nn.Sigmoid()
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with JetX-enhanced processing
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Processed temporal features
        """
        # Bidirectional GRU processing
        gru_output, _ = self.gru(x)  # [batch_size, seq_len, hidden_size * 2]
        
        # Pattern recognition
        pattern_scores = self.pattern_recognition(gru_output)  # [batch_size, seq_len, 3]
        
        # Variable selection
        processed = self.variable_selection([gru_output])  # [batch_size, seq_len, hidden_size]
        
        # Enhance with pattern information using a gating mechanism
        # Fix tensor dimension mismatch by ensuring gate has same shape as processed
        gate = self.pattern_gate(pattern_scores)  # [batch_size, seq_len, hidden_size]
        
        # Ensure gate has the same dimensions as processed for element-wise multiplication
        # Both should be [batch_size, seq_len, hidden_size]
        if gate.shape != processed.shape:
            # Debug information
            print(f"Gate shape: {gate.shape}, Processed shape: {processed.shape}")
            # Reshape gate to match processed dimensions
            if len(gate.shape) == 3 and len(processed.shape) == 3:
                if gate.shape[0] == processed.shape[0]:  # Same batch size
                    # Use broadcasting or repeat along the sequence dimension
                    gate = gate.expand(processed.shape[0], processed.shape[1], processed.shape[2])
                else:
                    # Fallback: use mean pooling to adjust dimensions
                    gate = torch.mean(gate, dim=1, keepdim=True).expand_as(processed)
            else:
                # Fallback: just use processed without gating
                gate = torch.ones_like(processed)
        
        pattern_enhanced = processed * gate
        
        return self.dropout(pattern_enhanced)

class JetXTFTLoss(nn.Module):
    """
    JetX-specific loss function for TFT
    """
    def __init__(self, threshold: float = 1.5, crash_weight: float = 2.0, alpha: float = 0.6):
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
        value_loss = self.mse(predictions['value'].squeeze(), targets)
        
        # Threshold probability loss
        threshold_targets = (targets >= self.threshold).float()
        prob_loss = self.bce(predictions['probability'].squeeze(), threshold_targets)
        
        # Crash risk loss
        crash_targets = (targets < self.threshold).float()
        crash_loss = self.bce(predictions['crash_risk'].squeeze(), crash_targets)
        
        # Confidence loss (should be high when prediction is accurate)
        prediction_accuracy = 1.0 - torch.abs(predictions['value'].squeeze() - targets) / targets.clamp(min=0.1)
        confidence_loss = self.mse(predictions['confidence'].squeeze(), prediction_accuracy)
        
        # Combined loss
        total_loss = (
            self.alpha * value_loss +
            (1 - self.alpha) * 0.4 * prob_loss +
            (1 - self.alpha) * 0.3 * crash_loss +
            (1 - self.alpha) * 0.3 * confidence_loss
        )
        
        return total_loss

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
        # Handle single input case (common in our TFT implementation)
        if len(inputs) == 1:
            # Simple case: just transform the single input
            transformed = self.feature_linear_layers[0](inputs[0])
            return self.dropout(transformed)
        
        # Transform each input
        transformed_inputs = []
        for i, (input_tensor, linear_layer) in enumerate(zip(inputs, self.feature_linear_layers)):
            transformed = linear_layer(input_tensor)
            transformed_inputs.append(transformed)
        
        # Stack transformed inputs
        transformed_inputs = torch.stack(transformed_inputs, dim=-1)  # [batch, seq, hidden, num_inputs]
        
        # Calculate variable selection weights
        flat_inputs = transformed_inputs.mean(dim=-1)  # [batch, seq, hidden]
        sparse_weights = F.softmax(self.variable_selection_weights(flat_inputs), dim=-1)  # [batch, seq, num_inputs]
        sparse_weights = self.dropout(sparse_weights)
        
        # Apply variable selection
        weights = sparse_weights.unsqueeze(-2)  # [batch, seq, 1, num_inputs]
        
        # Weighted sum across inputs
        processed_inputs = torch.sum(transformed_inputs * weights, dim=-1)  # [batch, seq, hidden]
        
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
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            mask: Attention mask
            return_attention: If True, returns attention weights along with predictions.
            
        Returns:
            - If return_attention is False: predictions (Tensor)
            - If return_attention is True: Tuple of (predictions, attention_weights)
        """
        # Input projection
        x = self.input_projection(x)
        x = self.dropout(x)
        
        # Temporal fusion decoder
        x, attention_weights = self.temporal_fusion_decoder(x, mask)
        
        # Output projection (use last timestep)
        predictions = self.output_projection(x[:, -1, :])
        
        if return_attention:
            return predictions, attention_weights
        return predictions

# This class is now obsolete and has been replaced by EnhancedTFTPredictor
# which inherits from the BasePredictor.
# Keeping the file for now to avoid breaking imports, but the class is removed.
pass
