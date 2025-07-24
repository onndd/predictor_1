"""
Base Predictor for all deep learning models in the JetX Prediction System.
This abstract class provides a common interface and shared logic for training,
prediction, and model persistence.
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, NamedTuple
from tqdm import tqdm
import torch.nn.functional as F
import psutil
import time
import gc

class AsymmetricAdviceLoss(nn.Module):
    """
    Custom loss function that heavily penalizes incorrect 'Play' advice.
    An incorrect 'Play' is when the model predicts >= threshold, but the actual is < threshold.
    """
    def __init__(self, threshold: float = 1.5, penalty_factor: float = 5.0):
        super().__init__()
        self.threshold = threshold
        self.penalty_factor = penalty_factor
        self.mse_loss = nn.MSELoss(reduction='none') # We want per-element loss to apply weights

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the asymmetric loss.
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        
        base_loss = self.mse_loss(predictions, targets)

        # Condition for incorrect 'Play' advice: prediction >= threshold AND target < threshold
        incorrect_play_condition = (predictions >= self.threshold) & (targets < self.threshold)
        
        # Create a penalty tensor, default is 1.0
        penalty = torch.ones_like(base_loss)
        
        # Apply penalty factor where the condition is met
        penalty[incorrect_play_condition] = self.penalty_factor
        
        # Apply the weights to the base loss
        weighted_loss = base_loss * penalty
        
        return weighted_loss.mean()

class PredictionResult(NamedTuple):
    """
    Represents the result of a prediction, including advice.
    """
    prediction: float
    confidence: float
    uncertainty: float
    advice: str
    advice_accuracy: float

class GPUMemoryManager:
    """GPU Memory monitoring and management utility"""
    
    def __init__(self, max_memory_gb: float = 12.0):
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024  # Convert to bytes
        self.device_name = None
        if torch.cuda.is_available():
            self.device_name = torch.cuda.get_device_name()
            print(f"🎮 GPU Detected: {self.device_name}")
            print(f"🎯 Memory Limit: {max_memory_gb}GB")
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory usage info"""
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "free": 0, "total": 0}
        
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        free = total - reserved
        
        return {
            "allocated_gb": allocated / 1024**3,
            "reserved_gb": reserved / 1024**3,
            "free_gb": free / 1024**3,
            "total_gb": total / 1024**3,
            "utilization": (allocated / total) * 100
        }
    
    def is_memory_safe(self, required_memory_gb: float = 0) -> bool:
        """Check if memory usage is within safe limits"""
        if not torch.cuda.is_available():
            return True
        
        current = self.get_memory_info()
        projected = current["allocated_gb"] + required_memory_gb
        return projected < (self.max_memory_bytes / 1024**3 * 0.9)  # 90% threshold
    
    def cleanup_memory(self):
        """Aggressive memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def get_optimal_batch_size(self, base_batch_size: int, model_size_gb: float) -> int:
        """Calculate optimal batch size based on available memory"""
        if not torch.cuda.is_available():
            return base_batch_size
        
        memory_info = self.get_memory_info()
        available_gb = memory_info["free_gb"] * 0.8  # Use 80% of available memory
        
        # Estimate memory per batch (rough calculation)
        memory_per_batch = model_size_gb * 1.5  # Model + gradients + activations
        
        if memory_per_batch > 0:
            optimal_batches = int(available_gb / memory_per_batch)
            optimal_batch_size = max(1, min(base_batch_size, optimal_batches))
        else:
            optimal_batch_size = base_batch_size
        
        return optimal_batch_size

class BasePredictor(ABC):
    """
    Abstract Base Class for all model predictors with GPU memory optimization.
    """
    def __init__(self, sequence_length: int, input_size: int, learning_rate: float, device: str = 'cpu', **kwargs):
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.device = device
        
        # GPU Memory Management
        self.memory_manager = GPUMemoryManager(max_memory_gb=kwargs.get('max_memory_gb', 12.0))
        self.use_mixed_precision = kwargs.get('use_mixed_precision', True) and torch.cuda.is_available()
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 1)
        
        # Build model and move to device
        self.model = self._build_model(input_size=input_size, **kwargs).to(self.device)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_loss_function(**kwargs)
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        
        self.is_trained = False
        self.training_history: Dict[str, List[float]] = {}
        
        # Log GPU optimization settings
        if torch.cuda.is_available():
            print(f"🚀 GPU Optimizations Enabled:")
            print(f"   - Mixed Precision: {self.use_mixed_precision}")
            print(f"   - Gradient Accumulation: {self.gradient_accumulation_steps}")
            print(f"   - Memory Limit: {kwargs.get('max_memory_gb', 12.0)}GB")

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

    def _create_loss_function(self, **kwargs) -> nn.Module:
        """
        Create the loss function.
        Uses AsymmetricAdviceLoss if relevant parameters are provided,
        otherwise defaults to a standard MSE loss for backward compatibility
        or for models that define their own loss.
        """
        loss_threshold = kwargs.get('loss_threshold')
        loss_penalty_factor = kwargs.get('loss_penalty_factor')

        if loss_threshold is not None and loss_penalty_factor is not None:
            print(f"✅ Using AsymmetricAdviceLoss with threshold={loss_threshold} and penalty={loss_penalty_factor}")
            return AsymmetricAdviceLoss(threshold=loss_threshold, penalty_factor=loss_penalty_factor)
        
        # Fallback for models that don't use the advice-driven loss.
        # Subclasses can still override this method.
        print("⚠️ Using default nn.MSELoss. For advice-driven training, provide loss_threshold and loss_penalty_factor in config.")
        return nn.MSELoss()

    def train(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 100, batch_size: int = 32,
              validation_split: float = 0.2, verbose: bool = True, tqdm_desc: str = "Training") -> dict:
        """
        Memory-efficient training loop with GPU optimization.
        
        Args:
            X (torch.Tensor): Input sequences tensor.
            y (torch.Tensor): Target values tensor.
            epochs (int): Number of training epochs.
            batch_size (int): Training batch size.
            validation_split (float): Fraction of data for validation.
            verbose (bool): Whether to print progress.
            tqdm_desc (str): Description for the progress bar.
        
        Returns:
            dict: Training history with GPU metrics.
        """
        if len(X) == 0:
            print("⚠️ Not enough data to train.")
            return {}

        # Memory optimization: Pre-transfer data to GPU (once)
        if self.device.startswith('cuda'):
            print("🚀 Pre-transferring data to GPU for memory efficiency...")
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.memory_manager.cleanup_memory()

        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        if len(X_train) == 0 or len(X_val) == 0:
            print("⚠️ Not enough data for training and validation split.")
            return {}

        # Dynamic batch size optimization
        if self.device.startswith('cuda'):
            initial_memory = self.memory_manager.get_memory_info()
            optimal_batch_size = self.memory_manager.get_optimal_batch_size(
                batch_size, model_size_gb=0.5  # Rough estimate
            )
            if optimal_batch_size != batch_size:
                print(f"📊 Optimizing batch size: {batch_size} → {optimal_batch_size}")
                batch_size = optimal_batch_size

        self.scheduler.T_max = epochs
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        train_losses, val_losses = [], []
        gpu_metrics = []
        
        # Training loop with memory monitoring
        epoch_iterator = tqdm(range(epochs), desc=tqdm_desc, leave=False)
        for epoch in epoch_iterator:
            self.model.train()
            total_train_loss = 0
            accumulated_loss = 0
            
            # Gradient accumulation for memory efficiency
            for i in range(0, len(X_train), batch_size):
                batch_end = min(i + batch_size, len(X_train))
                batch_X = X_train[i:batch_end]
                batch_y = y_train[i:batch_end]
                
                # Mixed precision forward pass
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(batch_X)
                        loss = self.criterion(predictions, batch_y)
                        loss = loss / self.gradient_accumulation_steps
                else:
                    predictions = self.model(batch_X)
                    loss = self.criterion(predictions, batch_y)
                    loss = loss / self.gradient_accumulation_steps
                
                accumulated_loss += loss.item()
                
                # Mixed precision backward pass
                if self.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Gradient accumulation step
                if (i // batch_size + 1) % self.gradient_accumulation_steps == 0:
                    if self.use_mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    total_train_loss += accumulated_loss
                    accumulated_loss = 0
                
                # Memory monitoring every 10 batches
                if (i // batch_size) % 10 == 0 and self.device.startswith('cuda'):
                    memory_info = self.memory_manager.get_memory_info()
                    if not self.memory_manager.is_memory_safe():
                        print(f"⚠️ Memory warning: {memory_info['allocated_gb']:.1f}GB used")
                        self.memory_manager.cleanup_memory()
            
            avg_train_loss = total_train_loss / (len(X_train) // (batch_size * self.gradient_accumulation_steps))
            
            # Validation with memory efficiency
            self.model.eval()
            with torch.no_grad():
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        val_predictions = self.model(X_val)
                        val_loss = self.criterion(val_predictions, y_val).item()
                else:
                    val_predictions = self.model(X_val)
                    val_loss = self.criterion(val_predictions, y_val).item()
            
            self.scheduler.step()
            
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            # GPU metrics collection
            if self.device.startswith('cuda'):
                memory_info = self.memory_manager.get_memory_info()
                gpu_metrics.append({
                    'epoch': epoch + 1,
                    'memory_gb': memory_info['allocated_gb'],
                    'utilization': memory_info['utilization']
                })
                
                epoch_iterator.set_description(
                    f"{tqdm_desc} | Epoch {epoch+1}/{epochs} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"GPU: {memory_info['allocated_gb']:.1f}GB ({memory_info['utilization']:.0f}%)"
                )
            else:
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
            
            # Periodic memory cleanup
            if (epoch + 1) % 10 == 0 and self.device.startswith('cuda'):
                self.memory_manager.cleanup_memory()
        
        # Final memory cleanup
        if self.device.startswith('cuda'):
            self.memory_manager.cleanup_memory()
            final_memory = self.memory_manager.get_memory_info()
            print(f"🎯 Training completed. Final GPU memory: {final_memory['allocated_gb']:.1f}GB")
        
        self.is_trained = True
        self.training_history = {
            'train_losses': train_losses, 
            'val_losses': val_losses,
            'gpu_metrics': gpu_metrics
        }
        return self.training_history

    def predict_for_testing(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        A method for getting raw model outputs for a batch of test data.
        Can be overridden by subclasses that require special input processing.
        By default, it passes the input directly to the model.
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(X.to(self.device))

    @abstractmethod
    def predict_with_confidence(self, sequence: List[float]) -> Tuple[float, float, float]:
        """
        Make a prediction with confidence metrics. Must be implemented by subclasses.
        """
        pass

    def get_prediction_with_advice(self, sequence: List[float], confidence_threshold: float = 0.85) -> PredictionResult:
        """
        CONSERVATIVE Advice System - Generates a prediction with strict multi-criteria validation.
        
        CRITICAL CHANGE: Default confidence_threshold increased from 0.6 to 0.85 to prevent money loss!
        Now uses multiple criteria to ensure only high-confidence, low-risk "Play" advice.

        Args:
            sequence (List[float]): The input sequence for prediction.
            confidence_threshold (float): The minimum confidence required to advise 'Play' (default: 0.85).

        Returns:
            PredictionResult: An object containing the prediction, confidence, uncertainty,
                              advice, and the accuracy of the advice.
        """
        prediction, confidence, uncertainty = self.predict_with_confidence(sequence)

        # MULTI-CRITERIA VALIDATION SYSTEM
        # All criteria must be TRUE for "Play" advice
        
        # Criterion 1: Basic threshold check (stricter than before)
        basic_threshold_met = prediction >= 1.5 and confidence >= confidence_threshold
        
        # Criterion 2: Uncertainty check (low uncertainty required)
        uncertainty_acceptable = uncertainty <= 0.3  # Low uncertainty required
        
        # Criterion 3: Conservative bias (apply penalty to confidence)
        conservative_confidence = confidence * 0.9  # 10% penalty for safety
        conservative_threshold_met = conservative_confidence >= (confidence_threshold - 0.05)
        
        # Criterion 4: Volatility check based on sequence
        volatility_safe = self._check_sequence_volatility(sequence)
        
        # Criterion 5: Pattern reliability check
        pattern_reliable = self._check_pattern_reliability(sequence, prediction)
        
        # FINAL DECISION: ALL criteria must be met
        all_criteria_met = (
            basic_threshold_met and 
            uncertainty_acceptable and 
            conservative_threshold_met and
            volatility_safe and
            pattern_reliable
        )
        
        if all_criteria_met:
            advice = "Play"
            # Adjusted accuracy accounting for all factors
            advice_accuracy = min(confidence * 0.9, 0.95)  # Cap at 95%, apply conservative penalty
        else:
            advice = "Do Not Play"
            # High accuracy for conservative advice
            advice_accuracy = max(0.8, 1.0 - uncertainty)
        
        return PredictionResult(
            prediction=prediction,
            confidence=confidence,
            uncertainty=uncertainty,
            advice=advice,
            advice_accuracy=advice_accuracy
        )
    
    def _check_sequence_volatility(self, sequence: List[float]) -> bool:
        """
        Check if sequence volatility is acceptable for Play advice.
        High volatility = risky = Do Not Play
        """
        if len(sequence) < 5:
            return False  # Not enough data = risky
        
        # Calculate volatility (standard deviation of recent values)
        recent_values = sequence[-10:]  # Last 10 values
        volatility = np.std(recent_values)
        
        # Volatility threshold (adjust based on your risk tolerance)
        max_acceptable_volatility = 0.5
        
        return volatility <= max_acceptable_volatility
    
    def _check_pattern_reliability(self, sequence: List[float], prediction: float) -> bool:
        """
        Check if the current pattern is reliable based on sequence characteristics.
        """
        if len(sequence) < 5:
            return False  # Not enough data = unreliable
        
        # Check for trend consistency
        recent_values = sequence[-5:]
        
        # Avoid predictions during rapid changes
        rapid_changes = sum(1 for i in range(1, len(recent_values)) 
                           if abs(recent_values[i] - recent_values[i-1]) > 1.0)
        
        if rapid_changes >= 2:  # Too many rapid changes = unreliable
            return False
        
        # Check if prediction aligns with recent trend
        if len(recent_values) >= 3:
            recent_avg = np.mean(recent_values[-3:])
            prediction_deviation = abs(prediction - recent_avg)
            
            # Prediction shouldn't deviate too much from recent average
            if prediction_deviation > 1.5:  # Large deviation = unreliable
                return False
        
        return True

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
                'input_size': self.input_size,
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
