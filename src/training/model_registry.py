"""
Model Registry for tracking trained models during a session.
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

# This will be fixed once settings.py is updated
import numpy as np

try:
    from src.config.settings import PATHS
except ImportError:
    PATHS = {'models_dir': 'trained_models'}

class NumpyJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for numpy types.
    Converts numpy integers, floats, and arrays to native Python types.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

class ModelRegistry:
    """
    A simple registry to keep track of models trained during a session.
    """
    def __init__(self):
        self.models: List[Dict[str, Any]] = []
        print("✅ ModelRegistry initialized.")

    def register_model(self, model_name: str, model_type: str, config: Dict, performance: Dict, model_path: str, metadata_path: str):
        """Registers a new model."""
        registration_info = {
            'model_name': model_name,
            'model_type': model_type,
            'config': config,
            'performance': performance,
            'model_path': model_path,
            'metadata_path': metadata_path,
            'timestamp': datetime.now().isoformat()
        }
        self.models.append(registration_info)
        print(f"✅ Model registered: {model_name} (MAE: {performance.get('mae', 'N/A'):.4f})")

    def list_models(self) -> List[Dict]:
        """Lists all registered models."""
        return self.models

    def get_best_model(self, model_type: str, metric: str = 'mae') -> Optional[Dict]:
        """Returns the best model of a specific type based on a metric."""
        if not self.models:
            return None
        
        type_models = [m for m in self.models if m['model_type'] == model_type]
        if not type_models:
            return None

        # For metrics like 'mae' or 'rmse', lower is better.
        # For metrics like 'accuracy', higher is better.
        reverse = metric not in ['mae', 'rmse']
        
        return sorted(type_models, key=lambda x: x['performance'].get(metric, float('inf')), reverse=reverse)[0]

    def get_best_overall_model(self, metric: str = 'f1') -> Optional[Dict]:
        """Returns the best model across all types based on a metric."""
        if not self.models:
            return None
        
        # For metrics like 'mae' or 'rmse', lower is better.
        # For metrics like 'f1', 'recall', 'accuracy', higher is better.
        reverse = metric not in ['mae', 'rmse']
        
        # Handle cases where metric might be missing
        default_value = float('-inf') if reverse else float('inf')
        
        return sorted(self.models, key=lambda x: x['performance'].get(metric, default_value), reverse=reverse)[0]

    def export_to_json(self, filename: str = None) -> str:
        """Exports the current registry to a JSON file."""
        if filename is None:
            filename = f"model_registry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(PATHS['models_dir'], filename)
        
        try:
            os.makedirs(PATHS['models_dir'], exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(self.models, f, indent=4, cls=NumpyJSONEncoder)
            return filepath
        except Exception as e:
            print(f"❌ Failed to export model registry: {e}")
            return ""