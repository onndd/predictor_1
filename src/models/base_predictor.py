"""
Base Predictor for all deep learning models in the JetX Prediction System.
This abstract class provides a common interface and shared logic for training,
prediction, and model persistence.
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

class BasePredictor(ABC):
    """
    Abstract Base Class for all model predictors.
    """
    def __init__(self, sequence_length: int, learning_rate: float, device: str = 'cpu', **kwargs):
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.device = device
        self.model = self._build_model(**kwargs).to(self.device)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_loss_function(**kwargs)
        
        self.is_trained = False
        self.training_history: Dict[str, List[float]] = {}

    @abstractmethod
    def _build_model(self, **kwargs) -> nn.Module:
        """Build the neural network model. Must be implemented by subclasses."""
        pass

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimizer. Can be overridden by subclasses if needed."""
        return torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-4 
        )

    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create the learning rate scheduler. Can be overridden."""
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=100, # Default, will be updated in train method
            eta_min=1e-6
        )

    @abstractmethod
    def _create_loss_function(self, **kwargs) -> nn.Module:
        """Create the loss function. Must be implemented by subclasses."""
        pass

    def prepare_sequences(self, data: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare sequences for training. This method is shared across all predictors.
        """
        processed_data = []
        if data and len(data) > 0:
            first_item = data[0]
            if isinstance(first_item, (tuple, list)) and len(first_item) > 1:
                processed_data = [float(item[1]) for item in data if isinstance(item, (tuple, list)) and len(item) > 1]
            else:
                processed_data = [float(item) for item in data]
        else:
            return torch.empty(0, self.sequence_length, 1), torch.empty(0)

        sequences, targets = [], []
        for i in range(len(processed_data) - self.sequence_length):
            seq = processed_data[i:i + self.sequence_length]
            target = processed_data[i + self.sequence_length]
            sequences.append(seq)
            targets.append(target)
        
        # Reshape for models that need a final feature dimension
        sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
        if len(sequences_tensor.shape) == 2:
             sequences_tensor = sequences_tensor.unsqueeze(-1)

        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        
        print(f"🔧 {self.__class__.__name__}: Prepared sequences shape: {sequences_tensor.shape}")
        print(f"🔧 {self.__class__.__name__}: Prepared targets shape: {targets_tensor.shape}")
        
        return sequences_tensor, targets_tensor

    def train(self, data: List[float], epochs: int = 100, batch_size: int = 32,
              validation_split: float = 0.2, verbose: bool = True, tqdm_desc: str = "Training") -> dict:
        """
        Generic training loop for all models.
        """
        X, y = self.prepare_sequences(data)
        
        if len(X) == 0:
            print("⚠️ Not enough data to train.")
            return {}

        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        if len(X_train) == 0 or len(X_val) == 0:
            print("⚠️ Not enough data for training and validation split.")
            return {}

        self.scheduler.T_max = epochs
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        train_losses, val_losses = [], []
        
        epoch_iterator = tqdm(range(epochs), desc=tqdm_desc, leave=False)
        for epoch in epoch_iterator:
            self.model.train()
            total_train_loss = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size].to(self.device)
                batch_y = y_train[i:i + batch_size].to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / (len(X_train) / batch_size)
            
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val.to(self.device))
                val_loss = self.criterion(val_predictions, y_val.to(self.device)).item()
            
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
                    print(f"\\nEarly stopping at epoch {epoch + 1}")
                break
        
        self.is_trained = True
        self.training_history = {'train_losses': train_losses, 'val_losses': val_losses}
        return self.training_history

    @abstractmethod
    def predict_with_confidence(self, sequence: List[float]) -> Tuple[float, float, float]:
        """
        Make a prediction with confidence metrics. Must be implemented by subclasses.
        """
        pass

    def save_model(self, filepath: str):
        """Save the trained model and its state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'model_config': {
                'sequence_length': self.sequence_length,
                'learning_rate': self.learning_rate
            }
        }, filepath)

    def load_model(self, filepath: str):
        """Load a trained model and its state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        self.is_trained = checkpoint.get('is_trained', False)
        self.model.to(self.device)