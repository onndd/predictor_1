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

class EnhancedLSTMPredictor:
    """
    Enhanced LSTM Predictor with PyTorch backend
    """
    def __init__(self, seq_length: int = 200, n_features: int = 1, threshold: float = 1.5,
                 hidden_size: int = 128, num_layers: int = 2, learning_rate: float = 0.001,
                 crash_weight: float = 5.0, device: str = 'cpu'):
        self.seq_length = seq_length
        self.n_features = n_features
        self.threshold = threshold
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.crash_weight = crash_weight
        self.device = device
        
        # Initialize model
        self.model = JetXLSTMModel(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            threshold=threshold
        ).to(self.device)
        
        # Loss function and optimizer
        self.criterion = self._create_loss_function()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        # Training state
        self.is_trained = False
        self.training_history = []
        
    def _create_loss_function(self):
        """Create multi-output loss function"""
        def loss_fn(predictions, targets):
            # Value loss (MSE)
            value_loss = F.mse_loss(predictions['value'].squeeze(), targets)
            
            # Probability loss (BCE)
            prob_targets = (targets >= self.threshold).float()
            prob_loss = F.binary_cross_entropy(predictions['probability'].squeeze(), prob_targets)
            
            # Crash risk loss (weighted)
            crash_targets = (targets < self.threshold).float()
            crash_weights = torch.where(crash_targets == 1, self.crash_weight, 1.0)
            weighted_crash_loss = F.binary_cross_entropy(
                predictions['crash_risk'].squeeze(),
                crash_targets,
                weight=crash_weights,
                reduction='mean'
            )
            
            # Confidence loss (higher confidence for accurate predictions)
            with torch.no_grad():
                accuracy = 1.0 - torch.abs(predictions['value'].squeeze() - targets) / targets.clamp(min=0.1)
            conf_loss = F.mse_loss(predictions['confidence'].squeeze(), accuracy)
            
            # Combined loss
            total_loss = (
                0.5 * value_loss +
                0.3 * prob_loss +
                0.1 * weighted_crash_loss +
                0.1 * conf_loss
            )
            return total_loss
        
        return loss_fn
    
    def prepare_sequences(self, data):
        """Prepare sequences for training"""
        # Handle tuple data
        if data and isinstance(data[0], (tuple, list)):
            processed_data = [float(item[1]) for item in data]
        else:
            processed_data = [float(item) for item in data]
        
        X, y = [], []
        
        for i in range(len(processed_data) - self.seq_length):
            seq = processed_data[i:i+self.seq_length]
            target = processed_data[i+self.seq_length]
            
            X.append(seq)
            y.append(target)
        
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # Add feature dimension
        y = torch.tensor(y, dtype=torch.float32)
        
        print(f"🔧 Enhanced LSTM: Prepared sequences shape: {X.shape}")
        print(f"🔧 Enhanced LSTM: Prepared targets shape: {y.shape}")
        
        return X, y
    
    def train(self, data, epochs=100, batch_size=32, validation_split=0.2, verbose=True, tqdm_desc="Training"):
        """Train the model"""
        # Prepare data
        X, y = self.prepare_sequences(data)
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Update scheduler
        self.scheduler.T_max = epochs
        
        # Training loop
        train_losses = []
        val_losses = []
        
        epoch_iterator = tqdm(range(epochs), desc=tqdm_desc, leave=False)
        for epoch in epoch_iterator:
            # Training
            self.model.train()
            total_train_loss = 0
            num_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size].to(self.device)
                batch_y = y_train[i:i + batch_size].to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_train_loss += loss.item()
                num_batches += 1
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val.to(self.device))
                val_loss = self.criterion(val_predictions, y_val.to(self.device)).item()
            
            # Update learning rate
            self.scheduler.step()
            
            train_losses.append(total_train_loss / num_batches)
            val_losses.append(val_loss)
            
            # Update tqdm description
            epoch_iterator.set_description(f"{tqdm_desc} | Epoch {epoch+1}/{epochs} | Val Loss: {val_loss:.4f}")
        
        self.is_trained = True
        return {'train_losses': train_losses, 'val_losses': val_losses}
    
    def predict_with_confidence(self, sequence):
        """Make prediction with confidence - required for rolling training system"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure correct length
        if len(sequence) < self.seq_length:
            sequence = [sequence[0]] * (self.seq_length - len(sequence)) + list(sequence)
        sequence = sequence[-self.seq_length:]
        
        try:
            self.model.eval()
            with torch.no_grad():
                X = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
                predictions = self.model(X)
                
                value = predictions['value'].squeeze().item()
                probability = predictions['probability'].squeeze().item()
                confidence = predictions['confidence'].squeeze().item()
                
                return float(value), float(probability), float(confidence)
                
        except Exception as e:
            raise RuntimeError(f"Enhanced LSTM prediction failed: {str(e)}")
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'is_trained': self.is_trained
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.is_trained = checkpoint['is_trained']

# Backward compatibility
class LSTMModel(EnhancedLSTMPredictor):
    def __init__(self, seq_length=200, n_features=1, threshold=1.5):
        super().__init__(seq_length, n_features, threshold)
