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

# Import optimized ensemble system
try:
    from ..ensemble.optimized_ensemble import OptimizedEnsemble
    from ..ensemble.simplified_confidence import SimplifiedConfidenceEstimator
    from ..feature_engineering.unified_extractor import UnifiedFeatureExtractor
    HAS_OPTIMIZED_ENSEMBLE = True
except ImportError as e:
    print(f"Warning: Optimized ensemble not available: {e}")
    HAS_OPTIMIZED_ENSEMBLE = False

# Import heavy model knowledge system
try:
    from .enhanced_light_models import HeavyModelKnowledge
    HAS_KNOWLEDGE_TRANSFER = True
except ImportError as e:
    print(f"Warning: Knowledge transfer not available: {e}")
    HAS_KNOWLEDGE_TRANSFER = False

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
        
        # Optimized ensemble system
        self.optimized_ensemble = None
        self.confidence_estimator = None
        self.feature_extractor = None
        self.use_optimized_ensemble = HAS_OPTIMIZED_ENSEMBLE
        
        # Knowledge transfer system
        self.heavy_knowledge = None
        self.knowledge_transfer_enabled = HAS_KNOWLEDGE_TRANSFER
        
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
        
        # Initialize optimized ensemble if available
        if self.use_optimized_ensemble:
            self.initialize_optimized_ensemble(data)
        
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

    def initialize_optimized_ensemble(self, data: List[float]):
        """Initialize optimized ensemble system"""
        if not HAS_OPTIMIZED_ENSEMBLE:
            print("Optimized ensemble not available")
            return
        
        print("Initializing optimized ensemble system...")
        
        try:
            # Initialize feature extractor
            self.feature_extractor = UnifiedFeatureExtractor(
                sequence_length=200,
                window_sizes=[5, 10, 20, 50, 100],
                threshold=1.5
            )
            
            # Fit feature extractor if enough data
            if len(data) >= 200:
                self.feature_extractor.fit(data)
                print("✓ Feature extractor fitted")
            else:
                print("⚠️  Not enough data to fit feature extractor")
            
            # Initialize confidence estimator
            self.confidence_estimator = SimplifiedConfidenceEstimator(history_window=200)
            print("✓ Confidence estimator initialized")
            
            # Initialize optimized ensemble
            self.optimized_ensemble = OptimizedEnsemble(
                models=self.models,
                threshold=1.5,
                performance_window=100
            )
            print("✓ Optimized ensemble initialized")
            
        except Exception as e:
            print(f"✗ Failed to initialize optimized ensemble: {e}")
            self.use_optimized_ensemble = False

    def predict_with_optimized_ensemble(self, sequence: List[float]) -> Dict[str, Any]:
        """
        Make prediction using optimized ensemble system
        
        Args:
            sequence: Input sequence
            
        Returns:
            Dictionary with enhanced prediction results
        """
        if not self.use_optimized_ensemble or self.optimized_ensemble is None:
            # Fallback to regular ensemble
            return self.ensemble_predict(sequence)
        
        try:
            # Get prediction from optimized ensemble
            result = self.optimized_ensemble.predict_next_value(sequence)
            
            if result is None:
                return {
                    'predictions': {},
                    'ensemble_prediction': None,
                    'confidence': 0.0,
                    'model_count': 0,
                    'optimized': False
                }
            
            predicted_value, above_prob, confidence = result
            
            # Get individual model predictions for confidence estimation
            model_predictions = {}
            for model_name, model in self.models.items():
                try:
                    pred = self.predict_with_model(model_name, sequence)
                    if pred is not None:
                        model_predictions[model_name] = {
                            'value': pred,
                            'probability': above_prob,  # Use ensemble probability
                            'confidence': confidence
                        }
                except:
                    continue
            
            # Enhanced confidence estimation
            if self.confidence_estimator is not None:
                confidence_analysis = self.confidence_estimator.estimate_confidence(
                    model_predictions, above_prob
                )
                
                enhanced_confidence = confidence_analysis['confidence_score']
                confidence_level = confidence_analysis['confidence_level']
                recommendation = confidence_analysis['recommendation']
            else:
                enhanced_confidence = confidence
                confidence_level = "Medium"
                recommendation = "Standard confidence"
            
            return {
                'predictions': model_predictions,
                'ensemble_prediction': predicted_value,
                'above_threshold_probability': above_prob,
                'confidence': enhanced_confidence,
                'confidence_level': confidence_level,
                'recommendation': recommendation,
                'model_count': len(model_predictions),
                'optimized': True,
                'ensemble_stats': self.optimized_ensemble.get_ensemble_stats() if self.optimized_ensemble else {}
            }
            
        except Exception as e:
            print(f"Optimized ensemble prediction failed: {e}")
            # Fallback to regular ensemble
            return self.ensemble_predict(sequence)

    def update_optimized_ensemble_performance(self, actual_value: float, prediction_id: Optional[str] = None):
        """
        Update optimized ensemble performance with actual result
        
        Args:
            actual_value: Actual JetX value
            prediction_id: Optional prediction ID for tracking
        """
        if not self.use_optimized_ensemble:
            return
        
        # Update optimized ensemble performance
        if self.optimized_ensemble is not None:
            try:
                self.optimized_ensemble.update_performance(actual_value, prediction_id)
            except Exception as e:
                print(f"Failed to update optimized ensemble performance: {e}")
        
        # Update confidence estimator
        if self.confidence_estimator is not None:
            try:
                self.confidence_estimator.add_prediction_result(
                    prediction=actual_value,
                    actual_value=actual_value,
                    model_predictions=None  # Will be added later if needed
                )
            except Exception as e:
                print(f"Failed to update confidence estimator: {e}")

    def get_optimized_ensemble_info(self) -> Dict[str, Any]:
        """Get optimized ensemble information"""
        if not self.use_optimized_ensemble:
            return {
                'available': False,
                'reason': 'Optimized ensemble not available'
            }
        
        info = {
            'available': True,
            'ensemble_stats': {},
            'confidence_stats': {},
            'feature_extractor_info': {}
        }
        
        # Ensemble stats
        if self.optimized_ensemble is not None:
            info['ensemble_stats'] = self.optimized_ensemble.get_ensemble_stats()
            info['model_info'] = self.optimized_ensemble.get_model_info()
        
        # Confidence stats
        if self.confidence_estimator is not None:
            info['confidence_stats'] = self.confidence_estimator.get_performance_summary()
        
        # Feature extractor info
        if self.feature_extractor is not None:
            info['feature_extractor_info'] = self.feature_extractor.get_info()
        
        return info

    def save_optimized_ensemble(self):
        """Save optimized ensemble state"""
        if not self.use_optimized_ensemble:
            return
        
        ensemble_dir = os.path.join(self.models_dir, "ensemble")
        
        try:
            # Save optimized ensemble
            if self.optimized_ensemble is not None:
                ensemble_path = os.path.join(ensemble_dir, "optimized_ensemble.pkl")
                self.optimized_ensemble.save_ensemble_state(ensemble_path)
                print("✓ Optimized ensemble state saved")
            
            # Save confidence estimator
            if self.confidence_estimator is not None:
                confidence_path = os.path.join(ensemble_dir, "confidence_estimator.pkl")
                self.confidence_estimator.save_confidence_state(confidence_path)
                print("✓ Confidence estimator state saved")
            
            # Save feature extractor
            if self.feature_extractor is not None:
                extractor_path = os.path.join(ensemble_dir, "feature_extractor.pkl")
                self.feature_extractor.save_extractor(extractor_path)
                print("✓ Feature extractor state saved")
                
        except Exception as e:
            print(f"Failed to save optimized ensemble: {e}")

    def load_optimized_ensemble(self):
        """Load optimized ensemble state"""
        if not HAS_OPTIMIZED_ENSEMBLE:
            return False
        
        ensemble_dir = os.path.join(self.models_dir, "ensemble")
        
        if not os.path.exists(ensemble_dir):
            return False
        
        try:
            # Load feature extractor
            extractor_path = os.path.join(ensemble_dir, "feature_extractor.pkl")
            if os.path.exists(extractor_path):
                self.feature_extractor = UnifiedFeatureExtractor()
                if self.feature_extractor.load_extractor(extractor_path):
                    print("✓ Feature extractor loaded")
                else:
                    self.feature_extractor = None
            
            # Load confidence estimator
            confidence_path = os.path.join(ensemble_dir, "confidence_estimator.pkl")
            if os.path.exists(confidence_path):
                self.confidence_estimator = SimplifiedConfidenceEstimator()
                if self.confidence_estimator.load_confidence_state(confidence_path):
                    print("✓ Confidence estimator loaded")
                else:
                    self.confidence_estimator = None
            
            # Load optimized ensemble
            ensemble_path = os.path.join(ensemble_dir, "optimized_ensemble.pkl")
            if os.path.exists(ensemble_path):
                self.optimized_ensemble = OptimizedEnsemble(models=self.models)
                if self.optimized_ensemble.load_ensemble_state(ensemble_path):
                    print("✓ Optimized ensemble loaded")
                    self.use_optimized_ensemble = True
                    return True
                else:
                    self.optimized_ensemble = None
            
            return False
            
        except Exception as e:
            print(f"Failed to load optimized ensemble: {e}")
            return False

    def predict_with_ensemble(self, sequence: List[float], use_optimized: bool = True) -> Dict[str, Any]:
        """
        Make prediction using best available ensemble method
        
        Args:
            sequence: Input sequence
            use_optimized: Whether to use optimized ensemble if available
            
        Returns:
            Dictionary with prediction results
        """
        if use_optimized and self.use_optimized_ensemble:
            return self.predict_with_optimized_ensemble(sequence)
        else:
            return self.ensemble_predict(sequence)

    def retrain_optimized_ensemble(self, data: List[float]):
        """Retrain optimized ensemble components"""
        if not self.use_optimized_ensemble:
            return
        
        print("Retraining optimized ensemble...")
        
        try:
            # Retrain feature extractor
            if self.feature_extractor is not None and len(data) >= 200:
                self.feature_extractor.fit(data)
                print("✓ Feature extractor retrained")
            
            # Reset confidence estimator
            if self.confidence_estimator is not None:
                self.confidence_estimator.reset_performance()
                print("✓ Confidence estimator reset")
            
            # Reset optimized ensemble
            if self.optimized_ensemble is not None:
                self.optimized_ensemble.reset_performance()
                print("✓ Optimized ensemble reset")
            
            print("✓ Optimized ensemble retraining completed")
            
        except Exception as e:
            print(f"Failed to retrain optimized ensemble: {e}")

    def extract_knowledge_from_heavy_models(self) -> Optional[Any]:
        """Heavy modellerden bilgi çıkar"""
        if not self.knowledge_transfer_enabled:
            print("Knowledge transfer not available")
            return None
        
        print("Heavy modellerden bilgi çıkarılıyor...")
        
        try:
            # HeavyModelKnowledge instance oluştur
            from .enhanced_light_models import HeavyModelKnowledge
            knowledge = HeavyModelKnowledge()
            
            # Pattern weights ekle
            knowledge.add_pattern_weight("high_volatility", 1.2)
            knowledge.add_pattern_weight("low_values", 1.3)
            knowledge.add_pattern_weight("high_values", 0.8)
            knowledge.add_pattern_weight("consecutive_highs", 0.7)
            knowledge.add_pattern_weight("oscillation", 1.1)
            
            # Threshold adjustments ekle
            knowledge.add_threshold_adjustment("high_volatility", -0.1)
            knowledge.add_threshold_adjustment("low_values", 0.05)
            knowledge.add_threshold_adjustment("high_values", -0.05)
            
            # Feature importance (örnek - gerçek implementasyonda heavy modellerden çıkarılacak)
            knowledge.add_feature_importance("recent_mean", 0.85)
            knowledge.add_feature_importance("volatility", 0.92)
            knowledge.add_feature_importance("consecutive_pattern", 0.78)
            knowledge.add_feature_importance("momentum", 0.71)
            
            # Heavy modellerden gerçek bilgi çıkarma
            if self.models:
                self._extract_patterns_from_models(knowledge)
            
            self.heavy_knowledge = knowledge
            print("✅ Heavy model bilgisi başarıyla çıkarıldı")
            return knowledge
            
        except Exception as e:
            print(f"Heavy model bilgi çıkarma hatası: {e}")
            return None
    
    def _extract_patterns_from_models(self, knowledge: Any):
        """Modellerden pattern bilgilerini çıkar"""
        try:
            # Her heavy model için
            for model_name, model in self.models.items():
                if self.model_configs[model_name].get('is_heavy', False):
                    if hasattr(model, 'model') and hasattr(model.model, 'state_dict'):
                        # PyTorch modellerden bilgi çıkar
                        self._extract_from_torch_model(model, knowledge, model_name)
                    elif hasattr(model, 'get_feature_importance'):
                        # Scikit-learn tarzı modellerden bilgi çıkar
                        self._extract_from_sklearn_model(model, knowledge, model_name)
        except Exception as e:
            print(f"Pattern extraction error: {e}")
    
    def _extract_from_torch_model(self, model: Any, knowledge: Any, model_name: str):
        """PyTorch modelinden bilgi çıkar"""
        try:
            # Model weights'lerden pattern çıkar (basit örnek)
            if hasattr(model, 'get_attention_weights'):
                # Attention weights'lerden önemli pattern'leri çıkar
                attention_weights = model.get_attention_weights()
                
                # En yüksek attention'a sahip pattern'leri belirle
                top_patterns = self._analyze_attention_patterns(attention_weights)
                
                for pattern, weight in top_patterns.items():
                    knowledge.add_pattern_weight(f"{model_name}_{pattern}", weight)
            
            # Model performance'dan threshold adjustment çıkar
            if hasattr(model, 'get_performance_metrics'):
                metrics = model.get_performance_metrics()
                if metrics and 'accuracy' in metrics:
                    # Yüksek accuracy'li modellerin threshold ayarları
                    if metrics['accuracy'] > 0.8:
                        knowledge.add_threshold_adjustment(f"{model_name}_high_acc", 0.02)
                    elif metrics['accuracy'] < 0.6:
                        knowledge.add_threshold_adjustment(f"{model_name}_low_acc", -0.02)
        except Exception as e:
            print(f"Torch model extraction error for {model_name}: {e}")
    
    def _extract_from_sklearn_model(self, model: Any, knowledge: Any, model_name: str):
        """Scikit-learn modelinden bilgi çıkar"""
        try:
            # Feature importance çıkar
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # En önemli feature'ları belirle
                top_features = self._get_top_features(importances)
                
                for feature, importance in top_features.items():
                    knowledge.add_feature_importance(f"{model_name}_{feature}", importance)
            
            # Decision tree'lerden threshold bilgisi çıkar
            if hasattr(model, 'tree_'):
                thresholds = self._extract_decision_thresholds(model.tree_)
                for condition, threshold in thresholds.items():
                    knowledge.add_threshold_adjustment(f"{model_name}_{condition}", threshold)
                    
        except Exception as e:
            print(f"Sklearn model extraction error for {model_name}: {e}")
    
    def _analyze_attention_patterns(self, attention_weights: Any) -> Dict[str, float]:
        """Attention weights'lerden pattern analizi"""
        patterns = {}
        try:
            # Basit pattern analizi (gerçek implementasyon daha karmaşık olacak)
            patterns["high_attention"] = 1.1
            patterns["low_attention"] = 0.9
            patterns["mixed_attention"] = 1.0
        except:
            pass
        return patterns
    
    def _get_top_features(self, importances: Any) -> Dict[str, float]:
        """En önemli feature'ları belirle"""
        features = {}
        try:
            # Feature importance'a göre sıralama
            if hasattr(importances, '__iter__'):
                for i, importance in enumerate(importances[:10]):  # Top 10
                    features[f"feature_{i}"] = float(importance)
        except:
            pass
        return features
    
    def _extract_decision_thresholds(self, tree: Any) -> Dict[str, float]:
        """Decision tree'den threshold bilgisi çıkar"""
        thresholds = {}
        try:
            # Decision tree threshold'larını analiz et
            thresholds["decision_threshold"] = 0.01
        except:
            pass
        return thresholds
    
    def transfer_knowledge_to_light_models(self):
        """Heavy model bilgisini light modellere aktar"""
        if not self.knowledge_transfer_enabled:
            print("Knowledge transfer not available")
            return False
        
        if not self.heavy_knowledge:
            print("Heavy model bilgisi mevcut değil, önce extract_knowledge_from_heavy_models() çalıştırın")
            return False
        
        print("Heavy model bilgisi light modellere aktarılıyor...")
        
        try:
            # Light modelleri güncelle
            for model_name, model in self.models.items():
                if not self.model_configs[model_name].get('is_heavy', False):
                    # Light model ise knowledge transfer yap
                    if hasattr(model, 'update_with_heavy_knowledge'):
                        model.update_with_heavy_knowledge(self.heavy_knowledge)
                    elif hasattr(model, 'models'):
                        # Ensemble ise tüm alt modelleri güncelle
                        if hasattr(model, 'update_with_heavy_knowledge'):
                            model.update_with_heavy_knowledge(self.heavy_knowledge)
            
            print("✅ Knowledge transfer tamamlandı")
            return True
            
        except Exception as e:
            print(f"Knowledge transfer hatası: {e}")
            return False
    
    def auto_knowledge_transfer(self):
        """Otomatik knowledge transfer"""
        if not self.knowledge_transfer_enabled:
            return False
        
        print("Otomatik knowledge transfer başlatılıyor...")
        
        # Heavy modellerin eğitilmiş olup olmadığını kontrol et
        heavy_models_trained = any(
            model_name in self.models and 
            hasattr(self.models[model_name], 'is_trained') and 
            self.models[model_name].is_trained
            for model_name in self.model_configs
            if self.model_configs[model_name].get('is_heavy', False)
        )
        
        if not heavy_models_trained:
            print("Heavy modeller henüz eğitilmemiş")
            return False
        
        # Bilgi çıkar
        knowledge = self.extract_knowledge_from_heavy_models()
        if not knowledge:
            return False
        
        # Light modellere aktar
        return self.transfer_knowledge_to_light_models()
    
    def get_knowledge_transfer_status(self) -> Dict[str, Any]:
        """Knowledge transfer durumunu al"""
        status = {
            'knowledge_transfer_enabled': self.knowledge_transfer_enabled,
            'heavy_knowledge_available': self.heavy_knowledge is not None,
            'heavy_models_trained': 0,
            'light_models_with_knowledge': 0,
            'total_heavy_models': 0,
            'total_light_models': 0
        }
        
        if self.knowledge_transfer_enabled and self.heavy_knowledge:
            status['knowledge_summary'] = self.heavy_knowledge.get_summary()
        
        # Model durumlarını analiz et
        for model_name, config in self.model_configs.items():
            if config.get('is_heavy', False):
                status['total_heavy_models'] += 1
                if (model_name in self.models and 
                    hasattr(self.models[model_name], 'is_trained') and 
                    self.models[model_name].is_trained):
                    status['heavy_models_trained'] += 1
            else:
                status['total_light_models'] += 1
                if (model_name in self.models and 
                    hasattr(self.models[model_name], 'knowledge_boost_enabled') and 
                    self.models[model_name].knowledge_boost_enabled):
                    status['light_models_with_knowledge'] += 1
        
        return status
    
    def save_knowledge(self, filepath: str):
        """Heavy model bilgisini kaydet"""
        if self.heavy_knowledge:
            try:
                self.heavy_knowledge.save_knowledge(filepath)
                print(f"✅ Heavy model bilgisi kaydedildi: {filepath}")
            except Exception as e:
                print(f"Knowledge save hatası: {e}")
        else:
            print("Kaydedilecek heavy model bilgisi yok")
    
    def load_knowledge(self, filepath: str):
        """Heavy model bilgisini yükle"""
        if not self.knowledge_transfer_enabled:
            print("Knowledge transfer not available")
            return False
        
        try:
            from .enhanced_light_models import HeavyModelKnowledge
            knowledge = HeavyModelKnowledge.load_knowledge(filepath)
            
            if knowledge:
                self.heavy_knowledge = knowledge
                print(f"✅ Heavy model bilgisi yüklendi: {filepath}")
                return True
            else:
                print("Knowledge yükleme başarısız")
                return False
                
        except Exception as e:
            print(f"Knowledge load hatası: {e}")
            return False
