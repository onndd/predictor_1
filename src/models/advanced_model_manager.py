"""
Advanced Model Manager for JetX Prediction System
Integrates all deep learning models with existing statistical models
"""

import os
import pickle
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import new deep learning models
try:
    from .deep_learning.n_beats.n_beats_model import NBeatsPredictor
    from .deep_learning.tft.tft_model import TFTPredictor
    from .deep_learning.informer.informer_model import InformerPredictor
    from .deep_learning.autoformer.autoformer_model import AutoformerPredictor
    from .deep_learning.pathformer.pathformer_model import PathformerPredictor
    HAS_DEEP_MODELS = True
except ImportError as e:
    print(f"Warning: Deep learning models not available: {e}")
    HAS_DEEP_MODELS = False

# Import existing models
try:
    from .enhanced_light_models import create_enhanced_light_models, LightModelEnsemble
    from .hybrid_predictor import HybridPredictor
    from .crash_detector import CrashDetector
    HAS_EXISTING_MODELS = True
except ImportError as e:
    print(f"Warning: Existing models not available: {e}")
    HAS_EXISTING_MODELS = False

class AdvancedModelManager:
    """
    Advanced Model Manager for JetX Prediction System
    
    Features:
    - Manages both deep learning and statistical models
    - Automatic model selection based on performance
    - Lazy loading of heavy models
    - Ensemble predictions
    - Model performance tracking
    """
    
    def __init__(self, models_dir: str = "trained_models", db_path: str = "jetx_data.db"):
        self.models_dir = models_dir
        self.db_path = db_path
        self.create_directories()
        
        # Model registry
        self.models = {}
        self.model_performances = {}
        self.model_metadata = {}
        self.model_configs = {}
        
        # Training state
        self.is_initialized = False
        self.auto_train_heavy_models = False  # Default: manual training
        
        # Model configurations
        self.setup_model_configs()
        
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.models_dir,
            os.path.join(self.models_dir, "deep_learning"),
            os.path.join(self.models_dir, "statistical"),
            os.path.join(self.models_dir, "ensemble"),
            os.path.join(self.models_dir, "metadata"),
            os.path.join(self.models_dir, "backup")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def setup_model_configs(self):
        """Setup model configurations"""
        self.model_configs = {
            # Deep Learning Models
            'n_beats': {
                'sequence_length': 200,
                'hidden_size': 256,
                'num_stacks': 3,
                'num_blocks': 3,
                'learning_rate': 0.001,
                'is_heavy': True
            },
            'tft': {
                'sequence_length': 200,
                'hidden_size': 256,
                'num_heads': 8,
                'num_layers': 2,
                'learning_rate': 0.001,
                'is_heavy': True
            },
            'informer': {
                'sequence_length': 200,
                'd_model': 512,
                'n_heads': 8,
                'e_layers': 2,
                'd_layers': 1,
                'learning_rate': 0.001,
                'is_heavy': True
            },
            'autoformer': {
                'sequence_length': 200,
                'd_model': 512,
                'n_heads': 8,
                'e_layers': 2,
                'd_layers': 1,
                'learning_rate': 0.001,
                'is_heavy': True
            },
            'pathformer': {
                'sequence_length': 200,
                'd_model': 512,
                'n_heads': 8,
                'num_layers': 6,
                'path_length': 3,
                'learning_rate': 0.001,
                'is_heavy': True
            },
            # Statistical Models (Light)
            'light_ensemble': {
                'is_heavy': False
            },
            'hybrid_predictor': {
                'is_heavy': False
            },
            'crash_detector': {
                'is_heavy': False
            }
        }
    
    def initialize_models(self, data: List[float], auto_train_heavy: bool = False):
        """
        Initialize all models
        
        Args:
            data: Training data
            auto_train_heavy: Whether to automatically train heavy models
        """
        print("Initializing Advanced Model Manager...")
        
        self.auto_train_heavy_models = auto_train_heavy
        
        # Initialize light models first
        self.initialize_light_models(data)
        
        # Initialize heavy models if requested
        if auto_train_heavy:
            self.initialize_heavy_models(data)
        else:
            self.initialize_heavy_models_lazy()
        
        self.is_initialized = True
        print("Model initialization completed.")
    
    def initialize_light_models(self, data: List[float]):
        """Initialize light statistical models"""
        print("Initializing light models...")
        
        if HAS_EXISTING_MODELS:
            # Light ensemble
            try:
                light_models = create_enhanced_light_models()
                self.models['light_ensemble'] = LightModelEnsemble(light_models)
                print("✓ Light ensemble initialized")
            except Exception as e:
                print(f"✗ Light ensemble failed: {e}")
            
            # Hybrid predictor
            try:
                self.models['hybrid_predictor'] = HybridPredictor(self.models_dir)
                print("✓ Hybrid predictor initialized")
            except Exception as e:
                print(f"✗ Hybrid predictor failed: {e}")
            
            # Crash detector
            try:
                self.models['crash_detector'] = CrashDetector()
                print("✓ Crash detector initialized")
            except Exception as e:
                print(f"✗ Crash detector failed: {e}")
    
    def initialize_heavy_models_lazy(self):
        """Initialize heavy models without training (lazy loading)"""
        print("Initializing heavy models (lazy loading)...")
        
        if not HAS_DEEP_MODELS:
            print("Deep learning models not available")
            return
        
        # Initialize model instances without training
        for model_name, config in self.model_configs.items():
            if config.get('is_heavy', False):
                try:
                    model = self.create_model_instance(model_name, config)
                    self.models[model_name] = model
                    print(f"✓ {model_name} initialized (not trained)")
                except Exception as e:
                    print(f"✗ {model_name} failed: {e}")
    
    def initialize_heavy_models(self, data: List[float]):
        """Initialize and train heavy models"""
        print("Initializing and training heavy models...")
        
        if not HAS_DEEP_MODELS:
            print("Deep learning models not available")
            return
        
        for model_name, config in self.model_configs.items():
            if config.get('is_heavy', False):
                try:
                    print(f"Training {model_name}...")
                    model = self.create_model_instance(model_name, config)
                    
                    # Train the model
                    history = model.train(data, epochs=50, verbose=False)
                    
                    # Save the model
                    model_path = os.path.join(self.models_dir, "deep_learning", f"{model_name}.pth")
                    model.save_model(model_path)
                    
                    self.models[model_name] = model
                    self.model_performances[model_name] = history
                    print(f"✓ {model_name} trained and saved")
                    
                except Exception as e:
                    print(f"✗ {model_name} training failed: {e}")
    
    def create_model_instance(self, model_name: str, config: Dict) -> Any:
        """Create a model instance based on name and config"""
        if model_name == 'n_beats':
            return NBeatsPredictor(**config)
        elif model_name == 'tft':
            return TFTPredictor(**config)
        elif model_name == 'informer':
            return InformerPredictor(**config)
        elif model_name == 'autoformer':
            return AutoformerPredictor(**config)
        elif model_name == 'pathformer':
            return PathformerPredictor(**config)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def train_heavy_model(self, model_name: str, data: List[float], epochs: int = 100) -> Dict:
        """
        Train a specific heavy model
        
        Args:
            model_name: Name of the model to train
            data: Training data
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        if not HAS_DEEP_MODELS:
            raise ValueError("Deep learning models not available")
        
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.model_configs[model_name]
        if not config.get('is_heavy', False):
            raise ValueError(f"{model_name} is not a heavy model")
        
        print(f"Training {model_name}...")
        
        # Create and train model
        model = self.create_model_instance(model_name, config)
        history = model.train(data, epochs=epochs, verbose=True)
        
        # Save model
        model_path = os.path.join(self.models_dir, "deep_learning", f"{model_name}.pth")
        model.save_model(model_path)
        
        # Update registry
        self.models[model_name] = model
        self.model_performances[model_name] = history
        
        print(f"✓ {model_name} training completed")
        return history
    
    def load_trained_model(self, model_name: str) -> bool:
        """
        Load a pre-trained model
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Success status
        """
        model_path = os.path.join(self.models_dir, "deep_learning", f"{model_name}.pth")
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False
        
        try:
            config = self.model_configs[model_name]
            model = self.create_model_instance(model_name, config)
            model.load_model(model_path)
            self.models[model_name] = model
            print(f"✓ {model_name} loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")
            return False
    
    def predict_with_model(self, model_name: str, sequence: List[float]) -> Optional[float]:
        """
        Make prediction with a specific model
        
        Args:
            model_name: Name of the model to use
            sequence: Input sequence
            
        Returns:
            Prediction or None if failed
        """
        if model_name not in self.models:
            print(f"Model {model_name} not available")
            return None
        
        try:
            model = self.models[model_name]
            
            # Check if model is trained
            if hasattr(model, 'is_trained') and not model.is_trained:
                print(f"Model {model_name} is not trained")
                return None
            
            # Make prediction
            prediction = model.predict(sequence)
            return prediction
            
        except Exception as e:
            print(f"Prediction failed for {model_name}: {e}")
            return None
    
    def ensemble_predict(self, sequence: List[float], weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Make ensemble prediction using all available models
        
        Args:
            sequence: Input sequence
            weights: Optional weights for each model
            
        Returns:
            Dictionary with predictions and ensemble result
        """
        predictions = {}
        valid_predictions = []
        model_names = []
        
        # Get predictions from all models
        for model_name in self.models:
            pred = self.predict_with_model(model_name, sequence)
            if pred is not None:
                predictions[model_name] = pred
                valid_predictions.append(pred)
                model_names.append(model_name)
        
        if not valid_predictions:
            return {
                'predictions': predictions,
                'ensemble_prediction': None,
                'confidence': 0.0,
                'model_count': 0
            }
        
        # Calculate ensemble prediction
        if weights is None:
            # Equal weights
            weights = {name: 1.0 / len(valid_predictions) for name in model_names}
        
        # Weighted average
        ensemble_pred = sum(pred * weights[name] for pred, name in zip(valid_predictions, model_names))
        
        # Calculate confidence based on prediction variance
        variance = np.var(valid_predictions)
        confidence = max(0.0, 1.0 - variance / 10.0)  # Normalize confidence
        
        return {
            'predictions': predictions,
            'ensemble_prediction': ensemble_pred,
            'confidence': confidence,
            'model_count': len(valid_predictions),
            'weights': weights
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        status = {}
        
        for model_name in self.model_configs:
            config = self.model_configs[model_name]
            is_loaded = model_name in self.models
            is_trained = False
            
            if is_loaded:
                model = self.models[model_name]
                if hasattr(model, 'is_trained'):
                    is_trained = model.is_trained
                else:
                    is_trained = True  # Assume statistical models are always ready
            
            status[model_name] = {
                'loaded': is_loaded,
                'trained': is_trained,
                'is_heavy': config.get('is_heavy', False),
                'performance': self.model_performances.get(model_name, None)
            }
        
        return status
    
    def save_all_models(self):
        """Save all models"""
        print("Saving all models...")
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'save_model'):
                    model_path = os.path.join(self.models_dir, "deep_learning", f"{model_name}.pth")
                    model.save_model(model_path)
                    print(f"✓ {model_name} saved")
            except Exception as e:
                print(f"✗ Failed to save {model_name}: {e}")
        
        # Save metadata
        metadata = {
            'model_performances': self.model_performances,
            'model_metadata': self.model_metadata,
            'model_configs': self.model_configs
        }
        
        metadata_path = os.path.join(self.models_dir, "metadata", "manager_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print("✓ Metadata saved")
    
    def load_all_models(self):
        """Load all saved models"""
        print("Loading all models...")
        
        # Load metadata
        metadata_path = os.path.join(self.models_dir, "metadata", "manager_metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.model_performances = metadata.get('model_performances', {})
                self.model_metadata = metadata.get('model_metadata', {})
                self.model_configs.update(metadata.get('model_configs', {}))
        
        # Load deep learning models
        deep_models_dir = os.path.join(self.models_dir, "deep_learning")
        if os.path.exists(deep_models_dir):
            for model_name in self.model_configs:
                if self.model_configs[model_name].get('is_heavy', False):
                    self.load_trained_model(model_name)
        
        print("✓ All models loaded")
    
    def retrain_models(self, data: List[float], model_names: Optional[List[str]] = None):
        """
        Retrain specified models or all models
        
        Args:
            data: Training data
            model_names: List of model names to retrain (None for all)
        """
        if model_names is None:
            model_names = list(self.model_configs.keys())
        
        print(f"Retraining models: {model_names}")
        
        for model_name in model_names:
            if model_name in self.model_configs:
                config = self.model_configs[model_name]
                
                if config.get('is_heavy', False):
                    # Retrain heavy model
                    try:
                        self.train_heavy_model(model_name, data, epochs=50)
                    except Exception as e:
                        print(f"Failed to retrain {model_name}: {e}")
                else:
                    # Retrain light model (if supported)
                    if model_name in self.models:
                        model = self.models[model_name]
                        if hasattr(model, 'retrain'):
                            try:
                                model.retrain(data)
                                print(f"✓ {model_name} retrained")
                            except Exception as e:
                                print(f"Failed to retrain {model_name}: {e}")
    
    def get_best_models(self, top_k: int = 3) -> List[str]:
        """
        Get the best performing models based on validation loss
        
        Args:
            top_k: Number of top models to return
            
        Returns:
            List of model names
        """
        model_scores = []
        
        for model_name, performance in self.model_performances.items():
            if performance and 'val_losses' in performance:
                # Use the best validation loss
                best_val_loss = min(performance['val_losses'])
                model_scores.append((model_name, best_val_loss))
        
        # Sort by validation loss (lower is better)
        model_scores.sort(key=lambda x: x[1])
        
        return [model_name for model_name, _ in model_scores[:top_k]]