"""
Refactored Rolling Window Trainer for the JetX Prediction System.
This module provides a generic, reusable trainer for any model type,
supporting both rolling window and incremental training strategies.
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
from src.evaluation.metrics import calculate_threshold_metrics # YENİ İMPORT

# A factory to get model classes dynamically
def get_model_predictor(model_type: str) -> Any:
    """Dynamically imports and returns a model predictor class using a mapping."""
    
    MODEL_MAP = {
        'N-Beats': 'src.models.deep_learning.n_beats.n_beats_model.NBeatsPredictor',
        'TFT': 'src.models.deep_learning.tft.enhanced_tft_model.EnhancedTFTPredictor',
        'LSTM': 'src.models.sequential.enhanced_lstm_pytorch.EnhancedLSTMPredictor',
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

def _convert_to_native_types(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively converts numpy types in a dictionary to native Python types for JSON serialization."""
    if isinstance(data, dict):
        return {key: _convert_to_native_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_convert_to_native_types(item) for item in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    return data

class RollingTrainer:
    """
    A refactored, generic training system supporting multiple strategies.
    """
    def __init__(self, model_registry: ModelRegistry, chunks: List[List[float]],
                 model_type: str, config: Dict[str, Any], device: str = 'cpu',
                 incremental: bool = False, enable_checkpointing: bool = True):
        self.model_registry = model_registry
        self.chunks = chunks
        self.model_type = model_type
        self.config = config
        self.device = device
        self.incremental = incremental
        self.enable_checkpointing = enable_checkpointing
        self.models_dir = PATHS['models_dir']
        self.checkpoint_dir = os.path.join(self.models_dir, 'checkpoints')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _get_checkpoint_path(self) -> str:
        """Generates the path for the checkpoint file."""
        strategy = "incremental" if self.incremental else "rolling"
        return os.path.join(self.checkpoint_dir, f"checkpoint_{self.model_type}_{strategy}.ckpt")

    def _save_checkpoint(self, model: Any, optimizer: Any, cycle: int):
        """Saves the training state."""
        if not self.enable_checkpointing:
            return
        
        checkpoint_path = self._get_checkpoint_path()
        state = {
            'cycle': cycle,
            'model_state_dict': model.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config
        }
        torch.save(state, checkpoint_path)
        print(f"  💾 Checkpoint saved for cycle {cycle} at {checkpoint_path}")

    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Loads training state from the latest checkpoint."""
        if not self.enable_checkpointing:
            return None
            
        checkpoint_path = self._get_checkpoint_path()
        if os.path.exists(checkpoint_path):
            try:
                print(f"  🔍 Found checkpoint: {checkpoint_path}. Attempting to resume...")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Check if config matches
                if checkpoint.get('config') != self.config:
                    print("  ⚠️  Configuration has changed. Starting training from scratch.")
                    self._cleanup_checkpoints() # Remove old checkpoint
                    return None
                
                return checkpoint
            except Exception as e:
                print(f"  ❌ Error loading checkpoint: {e}. Starting from scratch.")
                return None
        return None
        
    def _cleanup_checkpoints(self):
        """Removes the checkpoint file after successful completion."""
        if not self.enable_checkpointing:
            return
        checkpoint_path = self._get_checkpoint_path()
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"🧹 Cleaned up checkpoint file: {checkpoint_path}")

    def _get_model_instance(self) -> Any:
        """Creates an instance of the specified model type with its config."""
        print(f"🔧 Instantiating model: {self.model_type}")
        model_class = get_model_predictor(self.model_type)
        
        import inspect
        sig = inspect.signature(model_class.__init__)
        model_keys = {p.name for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD}
        
        param_mapping = {'lstm_units': 'hidden_size', 'seq_length': 'sequence_length'}
        
        filtered_config = {}
        for key, value in self.config.items():
            mapped_key = param_mapping.get(key, key)
            if mapped_key in model_keys:
                filtered_config[mapped_key] = value
        
        if self.model_type == 'LSTM' and 'n_features' not in filtered_config:
            filtered_config['n_features'] = 1

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
                
                sequence = [float(item[1]) if isinstance(item, (tuple, list)) else float(item) for item in raw_sequence]
                actual = float(raw_actual[1]) if isinstance(raw_actual, (tuple, list)) else float(raw_actual)
                
                pred_value, _, _ = model.predict_with_confidence(sequence)
                predictions.append(pred_value)
                actuals.append(actual)

            if not predictions: return None

            mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
            rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals))**2))
            
            # Sınıflandırma metriklerini hesapla
            classification_metrics = calculate_threshold_metrics(
                y_true=actuals,
                y_pred=predictions,
                threshold=self.config.get('threshold', 1.5)
            )
            
            # Regresyon metriklerini ekle
            classification_metrics['mae'] = mae
            classification_metrics['rmse'] = rmse
            
            return classification_metrics
        except Exception as e:
            print(f"❌ Model testing error: {e}")
            traceback.print_exc()
            return None

    def _save_model(self, model: Any, model_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Saves the model and its metadata to specific paths."""
        try:
            model_path = os.path.join(self.models_dir, f"{model_name}.pth")
            metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
            
            model.save_model(model_path)
            
            metadata = {
                'model_name': model_name,
                'model_type': self.model_type,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            print(f"💾 Model saved: {model_path}")
            return model_path, metadata_path
        except Exception as e:
            print(f"❌ Model save error: {e}")
            return None, None

    def execute_rolling_training(self) -> List[Dict[str, Any]]:
        """Executes the training loop based on the chosen strategy."""
        if self.incremental:
            return self._execute_incremental_training()
        else:
            return self._execute_windowed_training()

    def _execute_windowed_training(self) -> List[Dict[str, Any]]:
        """Executes the classic rolling window training loop with checkpointing."""
        print(f"🚀 Starting ROLLING WINDOW training for {self.model_type}...")
        cycle_results = []
        
        start_cycle = 0
        checkpoint = self._load_checkpoint()
        if checkpoint:
            start_cycle = checkpoint.get('cycle', -1) + 1
            print(f"  ✅ Resuming from cycle {start_cycle}")

        for cycle in tqdm(range(start_cycle, len(self.chunks) - 1), desc=f"Rolling Training: {self.model_type}"):
            print(f"\n🔄 Cycle {cycle + 1}/{len(self.chunks) - 1}")
            train_data = [item for sublist in self.chunks[:cycle + 1] for item in sublist]
            test_data = self.chunks[cycle + 1]
            
            sequence_length = self.config.get('sequence_length', 200)
            print(f"  - Training size: {len(train_data)}, Test size: {len(test_data)}, Required sequence_length: {sequence_length}")

            if len(train_data) <= sequence_length:
                print(f"  ⚠️  Cycle {cycle + 1} atlandı: Eğitim verisi ({len(train_data)}) sequence_length'ten ({sequence_length}) küçük veya eşit.")
                continue

            try:
                model = self._get_model_instance()
                train_params = self.config.get('train_params', {})
                train_params['tqdm_desc'] = f"Cycle {cycle + 1}"
                
                print(f"  - Model eğitimi başlıyor...")
                model.train(data=train_data, **train_params)
                
                print(f"  - Model testi başlıyor...")
                performance = self._test_model(model, test_data)
                if not performance:
                    print(f"  - ⚠️  Cycle {cycle + 1}: Test başarısız, döngü atlanıyor.")
                    continue

                model_name = f"{self.model_type}_cycle_{cycle+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                model_path, metadata_path = self._save_model(model, model_name)
                if not model_path:
                    print(f"  - ⚠️  Cycle {cycle + 1}: Model kaydetme başarısız, döngü atlanıyor.")
                    continue

                if metadata_path:
                    with open(metadata_path, 'r+') as f:
                        metadata = json.load(f)
                        performance_native = _convert_to_native_types(performance)
                        metadata.update({'performance': performance_native, 'cycle': cycle + 1})
                        f.seek(0)
                        json.dump(metadata, f, indent=4)
                        f.truncate()

                self.model_registry.register_model(model_name, self.model_type, self.config, performance, model_path, metadata_path)
                cycle_results.append({'cycle': cycle + 1, 'performance': performance, 'model_path': model_path})
                print(f"  ✅ Cycle {cycle + 1} completed: MAE={performance['mae']:.4f}, F1={performance['f1']:.4f}, Recall={performance['recall']:.4f}")
                
                # Save checkpoint after a successful cycle
                self._save_checkpoint(model, model.optimizer, cycle)

            except Exception as e:
                print(f"❌ Cycle {cycle + 1} failed for model {self.model_type} with config {self.config}")
                print(f"  - Hata: {e}")
                traceback.print_exc()
        
        self._cleanup_checkpoints()
        print(f"\n🎉 Rolling window training finished for {self.model_type}.")
        return cycle_results

    def _execute_incremental_training(self) -> List[Dict[str, Any]]:
        """Executes the incremental training loop with checkpointing."""
        print(f"🚀 Starting INCREMENTAL training for {self.model_type}...")
        
        model = self._get_model_instance()
        start_cycle = 0
        
        checkpoint = self._load_checkpoint()
        if checkpoint:
            try:
                model.model.load_state_dict(checkpoint['model_state_dict'])
                model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_cycle = checkpoint.get('cycle', -1) + 1
                print(f"  ✅ Model state loaded. Resuming from cycle {start_cycle}.")
            except Exception as e:
                print(f"  ❌ Checkpoint state could not be loaded, starting fresh. Error: {e}")

        cycle_results = []
        for cycle in tqdm(range(start_cycle, len(self.chunks) - 1), desc=f"Incremental Training: {self.model_type}"):
            print(f"\n🔄 Cycle {cycle + 1}/{len(self.chunks) - 1}")

            train_data = self.chunks[cycle]
            test_data = self.chunks[cycle + 1]
            print(f"  - Training size: {len(train_data)}, Test size: {len(test_data)}")

            try:
                train_params = self.config.get('train_params', {})
                train_params['tqdm_desc'] = f"Cycle {cycle + 1}"
                model.train(data=train_data, **train_params)
                
                performance = self._test_model(model, test_data)
                if not performance: continue

                model_name = f"{self.model_type}_incremental"
                saved_path, metadata_path = self._save_model(model, model_name)
                if not saved_path: continue

                self.model_registry.register_model(model_name, self.model_type, self.config, performance, saved_path, metadata_path)
                cycle_results.append({'cycle': cycle + 1, 'performance': performance, 'model_path': saved_path})
                print(f"  ✅ Cycle {cycle + 1} completed: MAE={performance['mae']:.4f}, Acc={performance['accuracy']:.4f}")

                # Save checkpoint after successful cycle
                self._save_checkpoint(model, model.optimizer, cycle)

            except Exception as e:
                print(f"❌ Cycle {cycle + 1} failed: {e}")
                traceback.print_exc()
        
        self._cleanup_checkpoints()
        print(f"\n🎉 Incremental training finished for {self.model_type}.")
        return cycle_results