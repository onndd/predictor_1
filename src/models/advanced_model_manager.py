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
DEEP_MODELS_ERROR = None
try:
    from .deep_learning.n_beats.n_beats_model import NBeatsPredictor
    from .deep_learning.tft.enhanced_tft_model import EnhancedTFTPredictor
    from .deep_learning.informer.informer_model import InformerPredictor
    from .deep_learning.autoformer.autoformer_model import AutoformerPredictor
    from .deep_learning.pathformer.pathformer_model import PathformerPredictor
    HAS_DEEP_MODELS = True
except ImportError as e:
    DEEP_MODELS_ERROR = f"Deep learning models not available: {e}"
    print(f"Warning: {DEEP_MODELS_ERROR}")
    HAS_DEEP_MODELS = False

# Import existing models
EXISTING_MODELS_ERROR = None
try:
    from .enhanced_light_models import create_enhanced_light_models, LightModelEnsemble
    from .hybrid_predictor import HybridPredictor
    from .crash_detector import CrashDetector
    HAS_EXISTING_MODELS = True
except ImportError as e:
    EXISTING_MODELS_ERROR = f"Existing models not available: {e}"
    print(f"Warning: {EXISTING_MODELS_ERROR}")
    HAS_EXISTING_MODELS = False

# Import optimized ensemble system
try:
    from src.ensemble.optimized_ensemble import OptimizedEnsemble
    from src.feature_engineering.unified_extractor import UnifiedFeatureExtractor
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

from src.config.settings import get_model_config, get_all_models

class AdvancedModelManager:
    """
    Advanced Model Manager for JetX Prediction System
    """
    
    def __init__(self, models_dir: str = "trained_models", db_path: str = "jetx_data.db",
                 min_prediction_value: float = 0.0, max_prediction_value: float = 1000.0):
        self.models_dir = models_dir
        self.db_path = db_path
        self.min_prediction_value = min_prediction_value
        self.max_prediction_value = max_prediction_value
        self.create_directories()
        
        self.models: Dict[str, Any] = {}
        self.model_performances: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Any] = {}
        
        self.is_initialized = False
        self.auto_train_heavy_models = False
        self.dependency_error = DEEP_MODELS_ERROR
        self.light_model_dependency_error = EXISTING_MODELS_ERROR
        
        self.optimized_ensemble: Optional[OptimizedEnsemble] = None
        self.feature_extractor: Optional[UnifiedFeatureExtractor] = None
        self.use_optimized_ensemble = HAS_OPTIMIZED_ENSEMBLE
        
        self.heavy_knowledge: Optional[HeavyModelKnowledge] = None
        self.knowledge_transfer_enabled = HAS_KNOWLEDGE_TRANSFER
        
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
    
    def initialize_models(self, data: List[float], auto_train_heavy: bool = False):
        """Initialize all models"""
        print("Initializing Advanced Model Manager...")
        self.auto_train_heavy_models = auto_train_heavy
        self.initialize_light_models(data)
        if auto_train_heavy:
            self.initialize_heavy_models(data)
        else:
            self.initialize_heavy_models_lazy()
        if self.use_optimized_ensemble:
            self.initialize_optimized_ensemble(data)
        self.is_initialized = True
        print("Model initialization completed.")
    
    def initialize_light_models(self, data: List[float]):
        """Initialize light statistical models"""
        if not HAS_EXISTING_MODELS: return
        try:
            light_models = create_enhanced_light_models()
            light_ensemble = LightModelEnsemble(threshold=1.5)
            for name, model in light_models.items():
                light_ensemble.add_model(name, model)
            self.models['light_ensemble'] = light_ensemble
            self.models['hybrid_predictor'] = HybridPredictor(self.models_dir)
            self.models['crash_detector'] = CrashDetector()
            print("✓ Light models initialized")
        except Exception as e:
            print(f"✗ Light models initialization failed: {e}")
    
    def initialize_heavy_models_lazy(self):
        """Initialize heavy models without training"""
        if not HAS_DEEP_MODELS: return
        for model_name in get_all_models():
            config = get_model_config(model_name)
            if config.get('is_heavy', False):
                try:
                    model = self.create_model_instance(model_name, config.get('params', {}))
                    self.models[model_name] = model
                except Exception as e:
                    print(f"✗ {model_name} lazy init failed: {e}")
    
    def initialize_heavy_models(self, data: List[float]):
        """Initialize and train heavy models"""
        if not HAS_DEEP_MODELS: return
        for model_name in get_all_models():
            config = get_model_config(model_name)
            if config.get('is_heavy', False):
                self.train_heavy_model(model_name, data, epochs=50)

    def create_model_instance(self, model_name: str, config: Dict) -> Any:
        """Create a model instance"""
        model_map = {
            'n_beats': NBeatsPredictor, 'tft': EnhancedTFTPredictor,
            'informer': InformerPredictor, 'autoformer': AutoformerPredictor,
            'pathformer': PathformerPredictor
        }
        if model_name not in model_map:
            raise ValueError(f"Unknown model: {model_name}")
        return model_map[model_name](**config)
    
    def train_heavy_model(self, model_name: str, data: List[float], epochs: int = 100) -> Dict:
        """Train a specific heavy model"""
        config = get_model_config(model_name)
        if not config or not config.get('is_heavy'):
            raise ValueError(f"Invalid or not a heavy model: {model_name}")
        
        print(f"Training {model_name}...")
        model = self.create_model_instance(model_name, config.get('params', {}))
        history = model.train(data, epochs=epochs, verbose=True)
        
        model_path = os.path.join(self.models_dir, "deep_learning", f"{model_name}.pth")
        model.save_model(model_path)
        
        self.models[model_name] = model
        self.model_performances[model_name] = history
        print(f"✓ {model_name} training completed")
        return history
    
    def load_trained_model(self, model_name: str) -> bool:
        """Load a pre-trained model"""
        model_path = os.path.join(self.models_dir, "deep_learning", f"{model_name}.pth")
        if not os.path.exists(model_path): return False
        try:
            config = get_model_config(model_name)
            model = self.create_model_instance(model_name, config.get('params', {}))
            model.load_model(model_path)
            self.models[model_name] = model
            return True
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")
            return False
    
    def predict_with_model(self, model_name: str, sequence: List[float]) -> Optional[float]:
        """Make prediction with a specific model"""
        if model_name not in self.models: return None
        try:
            model = self.models[model_name]
            if hasattr(model, 'is_trained') and not model.is_trained: return None
            
            config = get_model_config(model_name).get('params', {})
            seq_len = config.get('sequence_length', 200)
            if len(sequence) != seq_len: return None
            
            prediction = model.predict(sequence)
            if isinstance(prediction, (int, float)) and np.isfinite(prediction):
                return float(np.clip(prediction, self.min_prediction_value, self.max_prediction_value))
            return None
        except Exception:
            return None
    
    def ensemble_predict(self, sequence: List[float], weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Make ensemble prediction"""
        predictions = {name: self.predict_with_model(name, sequence) for name in self.models}
        valid_preds = {k: v for k, v in predictions.items() if v is not None}
        
        if not valid_preds:
            return {'ensemble_prediction': None, 'confidence': 0.0, 'model_count': 0}
        
        if weights is None:
            weights = {name: 1.0 / len(valid_preds) for name in valid_preds}
        
        ensemble_pred = sum(v * weights.get(k, 0) for k, v in valid_preds.items())
        variance = float(np.var(list(valid_preds.values())))
        confidence = max(0.0, 1.0 - variance / 10.0)
        
        return {
            'predictions': valid_preds, 'ensemble_prediction': ensemble_pred,
            'confidence': confidence, 'model_count': len(valid_preds), 'weights': weights
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        status = {}
        for model_name in get_all_models():
            config = get_model_config(model_name)
            is_loaded = model_name in self.models
            is_trained = is_loaded and getattr(self.models.get(model_name), 'is_trained', True)
            status[model_name] = {
                'loaded': is_loaded, 'trained': is_trained,
                'is_heavy': config.get('is_heavy', False),
                'performance': self.model_performances.get(model_name)
            }
        return status
    
    def save_all_models(self):
        """Save all models"""
        for name, model in self.models.items():
            if hasattr(model, 'save_model'):
                path = os.path.join(self.models_dir, "deep_learning", f"{name}.pth")
                model.save_model(path)
        
        metadata_path = os.path.join(self.models_dir, "metadata", "manager_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump({'model_performances': self.model_performances, 'model_metadata': self.model_metadata}, f)
    
    def load_all_models(self):
        """Load all saved models"""
        metadata_path = os.path.join(self.models_dir, "metadata", "manager_metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.model_performances = metadata.get('model_performances', {})
                self.model_metadata = metadata.get('model_metadata', {})
        
        for model_name in get_all_models():
            if get_model_config(model_name).get('is_heavy', False):
                self.load_trained_model(model_name)
    
    def retrain_models(self, data: List[float], model_names: Optional[List[str]] = None):
        """Retrain specified models or all models"""
        if model_names is None: model_names = get_all_models()
        for name in model_names:
            if get_model_config(name).get('is_heavy', False):
                self.train_heavy_model(name, data, epochs=50)

    def get_best_models(self, top_k: int = 3) -> List[str]:
        """Get the best performing models"""
        scores = [
            (name, min(p['val_losses'])) for name, p in self.model_performances.items() 
            if p and 'val_losses' in p and p['val_losses']
        ]
        scores.sort(key=lambda x: x[1])
        return [name for name, _ in scores[:top_k]]

    def initialize_optimized_ensemble(self, data: List[float]):
        """Initialize optimized ensemble system"""
        if not HAS_OPTIMIZED_ENSEMBLE: return
        try:
            self.feature_extractor = UnifiedFeatureExtractor()
            if len(data) >= 200: self.feature_extractor.fit(data)
            self.optimized_ensemble = OptimizedEnsemble(models=self.models, threshold=1.5)
            print("✓ Optimized ensemble initialized")
        except Exception as e:
            print(f"✗ Failed to initialize optimized ensemble: {e}")
            self.use_optimized_ensemble = False

    def predict_with_optimized_ensemble(self, sequence: List[float]) -> Dict[str, Any]:
        """Make prediction using optimized ensemble system"""
        if not self.use_optimized_ensemble or not self.optimized_ensemble:
            return self.ensemble_predict(sequence)
        try:
            result = self.optimized_ensemble.predict_next_value(sequence)
            if result is None: return {'ensemble_prediction': None, 'confidence': 0.0}
            pred_val, above_prob, conf = result
            return {
                'ensemble_prediction': pred_val, 'above_threshold_probability': above_prob,
                'confidence': conf, 'optimized': True
            }
        except Exception as e:
            print(f"Optimized ensemble prediction failed: {e}")
            return self.ensemble_predict(sequence)

    def get_dependency_error(self) -> Optional[str]:
        """Returns the dependency error message if any"""
        errors = [e for e in [self.dependency_error, self.light_model_dependency_error] if e]
        return "\n".join(errors) if errors else None


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
