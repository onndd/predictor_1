"""
Enhanced TFT Model with JetX-specific Multi-Output Predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from .tft_model import *
from tqdm.notebook import tqdm

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
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-output predictions
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            mask: Attention mask
            
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
        
        return {
            'value': value_pred,
            'probability': probability_pred,
            'confidence': confidence_pred,
            'crash_risk': crash_risk_pred,
            'pattern': pattern_pred,
            'attention_weights': attention_weights
        }

class EnhancedTFTPredictor:
    """
    Enhanced TFT predictor with JetX-specific features
    """
    def __init__(self, sequence_length: int = 200, hidden_size: int = 256,
                 num_heads: int = 8, num_layers: int = 2, learning_rate: float = 0.001,
                 threshold: float = 1.5, device: str = 'cpu'):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.device = device
        
        # Initialize enhanced JetX TFT model
        self.model = JetXTFTModel(
            input_size=1,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            forecast_horizon=1,
            threshold=threshold
        ).to(self.device)
        
        # Use JetX-specific loss function
        self.criterion = JetXTFTLoss(threshold=threshold)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01,  # L2 regularization
            eps=1e-8
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
        
    def prepare_sequences(self, data: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare sequences for training, handling both lists of floats and lists of tuples.
        """
        # Check if the data is a list of tuples and extract the second element if so.
        processed_data = []
        if data and len(data) > 0:
            first_item = data[0]
            if isinstance(first_item, (tuple, list)) and len(first_item) > 1:
                processed_data = [float(item[1]) for item in data if isinstance(item, (tuple, list)) and len(item) > 1]
            else:
                processed_data = [float(item) for item in data]
        else:
            processed_data = []
        
        sequences = []
        targets = []
        
        for i in range(len(processed_data) - self.sequence_length):
            seq = processed_data[i:i + self.sequence_length]
            target = processed_data[i + self.sequence_length]
            
            sequences.append(seq)
            targets.append(target)
        
        sequences_tensor = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        
        print(f"🔧 Enhanced TFT: Prepared sequences shape: {sequences_tensor.shape}")
        print(f"🔧 Enhanced TFT: Prepared targets shape: {targets_tensor.shape}")
        
        return sequences_tensor, targets_tensor
    
    def train(self, data: List[float], epochs: int = 100, batch_size: int = 32,
              validation_split: float = 0.2, verbose: bool = True, tqdm_desc: str = "Training") -> dict:
        """
        Train the enhanced TFT model
        """
        # Prepare data
        X, y = self.prepare_sequences(data)
        
        # Split into train and validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Update scheduler T_max
        self.scheduler.T_max = epochs
        
        # Early stopping parameters
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
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
            
            avg_train_loss = total_train_loss / num_batches
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val.to(self.device))
                val_loss = self.criterion(val_predictions, y_val.to(self.device)).item()
            
            # Update learning rate
            self.scheduler.step()
            
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            # Update tqdm description
            epoch_iterator.set_description(f"{tqdm_desc} | Epoch {epoch+1}/{epochs} | Val Loss: {val_loss:.4f}")
            
            # Early stopping
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
        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        return self.training_history
    
    def predict_with_confidence(self, sequence: List[float]) -> Tuple[float, float, float]:
        """
        Make a prediction with confidence metrics - required for rolling training system
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if len(sequence) != self.sequence_length:
            raise ValueError(f"Sequence must have length {self.sequence_length}")
        
        try:
            self.model.eval()
            with torch.no_grad():
                x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
                predictions = self.model(x)
                
                value = predictions['value'].squeeze().item()
                probability = predictions['probability'].squeeze().item()
                confidence = predictions['confidence'].squeeze().item()
                
                # Validate outputs
                import numpy as np
                if any(np.isnan([value, probability, confidence])) or any(np.isinf([value, probability, confidence])):
                    raise ValueError("Model produced invalid predictions")
                
                return float(value), float(probability), float(confidence)
                
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
