"""
Refactored Rolling Window Trainer for the JetX Prediction System.
This module provides a generic, reusable trainer for any model type.
"""

import numpy as np
import torch
import os
import json
import traceback
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple

from src.training.model_registry import ModelRegistry
from src.config.settings import PATHS

# A factory to get model classes dynamically
def get_model_predictor(model_type: str) -> Any:
    """Dynamically imports and returns a model predictor class using a mapping."""
    
    # Model adını ve import yolunu içeren bir sözlük
    MODEL_MAP = {
        'N-Beats': 'src.models.deep_learning.n_beats.n_beats_model.NBeatsPredictor',
        'TFT': 'src.models.deep_learning.tft.enhanced_tft_model.EnhancedTFTPredictor',
        'LSTM': 'src.models.sequential.enhanced_lstm_pytorch.EnhancedLSTMPredictor',
        # Yeni modeller buraya kolayca eklenebilir
    }

    if model_type not in MODEL_MAP:
        raise ImportError(f"Model type '{model_type}' is not supported or defined in MODEL_MAP.")

    try:
        module_path, class_name = MODEL_MAP[model_type].rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        print(f"❌ Could not import model {model_type} from {MODEL_MAP[model_type]}: {e}")
        raise

class RollingTrainer:
    """
    A refactored, generic rolling window training system.
    """
    def __init__(self, model_registry: ModelRegistry, chunks: List[List[float]], model_type: str, config: Dict[str, Any], device: str = 'cpu'):
        self.model_registry = model_registry
        self.chunks = chunks
        self.model_type = model_type
        self.config = config
        self.device = device
        self.models_dir = PATHS['models_dir']
        os.makedirs(self.models_dir, exist_ok=True)

    def _get_model_instance(self) -> Any:
        """Creates an instance of the specified model type with its config."""
        print(f"🔧 Instantiating model: {self.model_type}")
        model_class = get_model_predictor(self.model_type)
        
        # Prepare config by filtering only relevant keys for the model constructor
        # This is a more robust way to pass parameters
        import inspect
        sig = inspect.signature(model_class.__init__)
        model_keys = {p.name for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD}
        
        # Special mapping for differing names (e.g., lstm_units vs hidden_size)
        param_mapping = {
            'lstm_units': 'hidden_size',
            'seq_length': 'sequence_length'
        }
        
        filtered_config = {}
        for key, value in self.config.items():
            mapped_key = param_mapping.get(key, key)
            if mapped_key in model_keys:
                filtered_config[mapped_key] = value
        
        # Ensure n_features is set for LSTM if not present
        if self.model_type == 'LSTM' and 'n_features' not in filtered_config:
            filtered_config['n_features'] = 1

        # Pass the device to the model constructor
        if 'device' in inspect.signature(model_class.__init__).parameters:
            filtered_config['device'] = self.device

        return model_class(**filtered_config)

    def _test_model(self, model: Any, test_data: List[float]) -> Optional[Dict[str, Any]]:
        """Generic model testing function."""
        try:
            sequence_length = self.config['sequence_length']
            if len(test_data) < sequence_length + 1:
                print("⚠️ Not enough test data for a full evaluation.")
                return None

            predictions, actuals = [], []
            for i in range(sequence_length, len(test_data)):
                raw_sequence = test_data[i-sequence_length:i]
                raw_actual = test_data[i]

                # FIX: Process sequence to handle tuples, ensuring correct data shape
                # FIX: Process sequence to handle tuples or floats individually
                sequence = [float(item[1]) if isinstance(item, (tuple, list)) else float(item) for item in raw_sequence]
                
                if isinstance(raw_actual, (tuple, list)):
                    actual = float(raw_actual[1])
                else:
                    actual = float(raw_actual)
                
                pred_value, _, _ = model.predict_with_confidence(sequence)
                predictions.append(pred_value)
                actuals.append(actual)

            if not predictions:
                return None

            mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
            rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals))**2))
            
            pred_above = [p >= self.config.get('threshold', 1.5) for p in predictions]
            actual_above = [a >= self.config.get('threshold', 1.5) for a in actuals]
            accuracy = np.mean([p == a for p, a in zip(pred_above, actual_above)])
            
            return {'mae': mae, 'rmse': rmse, 'accuracy': accuracy}
        except Exception as e:
            print(f"❌ Model testing error: {e}")
            traceback.print_exc()
            return None

    def _save_model(self, model: Any, cycle: int) -> Tuple[Optional[str], Optional[str]]:
        """Saves the model and its metadata."""
        try:
            model_name = f"{self.model_type}_cycle_{cycle}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = os.path.join(self.models_dir, f"{model_name}.pth")
            model.save_model(model_path)
            
            metadata = {
                'model_name': model_name,
                'model_type': self.model_type,
                'cycle': cycle,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            print(f"💾 Model saved: {model_path}")
            return model_path, metadata_path
        except Exception as e:
            print(f"❌ Model save error: {e}")
            return None, None

    def execute_rolling_training(self) -> List[Dict[str, Any]]:
        """Executes the generic rolling window training loop."""
        print(f"🚀 Starting rolling training for {self.model_type}...")
        
        cycle_results = []
        # Wrap the main loop with tqdm for a cycle progress bar
        for cycle in tqdm(range(len(self.chunks) - 1), desc=f"Rolling Training: {self.model_type}"):
            print(f"\n🔄 Cycle {cycle + 1}/{len(self.chunks) - 1}")
            
            train_data = [item for sublist in self.chunks[:cycle + 1] for item in sublist]
            test_data = self.chunks[cycle + 1]
            
            print(f"  - Training size: {len(train_data)}")
            print(f"  - Test size: {len(test_data)}")

            try:
                model = self._get_model_instance()
                # The model instance is now created with the device parameter,
                # and its internal methods will handle moving data to the device.
                
                # Pass the tqdm description to the train method
                train_params = self.config.get('train_params', {})
                train_params['tqdm_desc'] = f"Cycle {cycle + 1}"
                
                model.train(data=train_data, **train_params)
                
                performance = self._test_model(model, test_data)
                if not performance:
                    print("❌ Model testing failed, skipping cycle.")
                    continue
                
                model_path, metadata_path = self._save_model(model, cycle + 1)
                if not model_path or not metadata_path:
                    print("❌ Model saving failed, skipping cycle.")
                    continue

                # Register and store results
                self.model_registry.register_model(
                    model_name=os.path.basename(model_path).replace('.pth', ''),
                    model_type=self.model_type,
                    config=self.config,
                    performance=performance,
                    model_path=model_path,
                    metadata_path=metadata_path
                )
                cycle_results.append({
                    'cycle': cycle + 1,
                    'performance': performance,
                    'model_path': model_path
                })
                
                print(f"  ✅ Cycle {cycle + 1} completed: MAE={performance['mae']:.4f}, Acc={performance['accuracy']:.4f}")

            except Exception as e:
                print(f"❌ Cycle {cycle + 1} failed: {e}")
                traceback.print_exc()
        
        print(f"\n🎉 Rolling training finished for {self.model_type}.")
        return cycle_results