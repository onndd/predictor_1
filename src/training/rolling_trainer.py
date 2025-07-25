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
import shap
import matplotlib.pyplot as plt

from src.training.model_registry import ModelRegistry
from src.config.settings import PATHS
from src.evaluation.metrics import calculate_threshold_metrics
from src.data_processing.manager import DataManager
from src.explainability.shap_explainer import ShapExplainer
from src.explainability.optimized_shap_explainer import OptimizedShapExplainer
from src.feature_engineering.unified_extractor import UnifiedFeatureExtractor
from src.evaluation.reporting import explain_test_results
from src.config.settings import CONFIG

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
        self.data_manager = DataManager(use_cache=False) # Rolling trainer için cache kullanmıyoruz
        
        # Initialize optimized SHAP explainer
        self.optimized_shap = OptimizedShapExplainer(CONFIG)
        
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

    def _get_model_instance(self, input_size: int) -> Any:
        """Creates an instance of the specified model type with GPU optimization config."""
        print(f"🔧 Instantiating model: {self.model_type} with input_size: {input_size}")
        model_class = get_model_predictor(self.model_type)
        
        # Add input_size to the config for the model constructor
        model_config = self.config.copy()
        model_config['input_size'] = input_size
        
        # Add the global model_sequence_length from the main config to the model's config
        from src.config.settings import CONFIG
        training_config = CONFIG.get('training', {})
        if 'model_sequence_length' in training_config:
            model_config['model_sequence_length'] = training_config['model_sequence_length']

        # GPU Optimization: Inject GPU settings into model config
        gpu_optimization = CONFIG.get('gpu_optimization', {})
        if gpu_optimization and self.device.startswith('cuda'):
            model_config['gpu_optimization'] = gpu_optimization
            print(f"🚀 GPU Optimization enabled for {self.model_type}:")
            print(f"   - Max Memory: {gpu_optimization.get('max_memory_gb', 12.0)}GB")
            print(f"   - Mixed Precision: {gpu_optimization.get('use_mixed_precision', True)}")
            print(f"   - Gradient Accumulation: {gpu_optimization.get('gradient_accumulation_steps', 2)}")

        # Filter config to only pass parameters expected by the model's __init__
        import inspect
        sig = inspect.signature(model_class.__init__)
        allowed_keys = {p.name for p in sig.parameters.values()}
        
        filtered_config = {k: v for k, v in model_config.items() if k in allowed_keys}
        
        # Ensure device is passed if expected
        if 'device' in allowed_keys:
            filtered_config['device'] = self.device
            
        return model_class(**filtered_config)

    def _test_model(self, model: Any, X_test: torch.Tensor, y_test: torch.Tensor) -> Optional[Dict[str, Any]]:
        """Generic model testing function using pre-featured data."""
        try:
            if len(X_test) == 0:
                print("⚠️ Not enough test data for a full evaluation.")
                return None

            # Use the new `predict_for_testing` method which handles model-specific logic
            outputs = model.predict_for_testing(X_test)
            predictions = outputs['value'].squeeze().cpu().numpy()
            actuals = y_test.cpu().numpy()

            if predictions.ndim == 0:
                predictions = np.expand_dims(predictions, 0)

            mae = np.mean(np.abs(predictions - actuals))
            rmse = np.sqrt(np.mean((predictions - actuals)**2))
            
            classification_metrics = calculate_threshold_metrics(
                y_true=actuals.tolist(),
                y_pred=predictions.tolist(),
                threshold=self.config.get('threshold', 1.5)
            )
            
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
                json.dump(metadata, f, indent=4, cls=NumpyJSONEncoder)
            
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
        
        # CRITICAL FIX: Ensure minimum cycles for reliable training
        min_cycles = 10  # Minimum 10 cycles for reliable results
        available_cycles = len(self.chunks) - 1
        
        if available_cycles < min_cycles:
            print(f"⚠️ WARNING: Only {available_cycles} cycles available, but {min_cycles} recommended for reliable training!")
            print(f"⚠️ This may result in poor precision. Consider collecting more data.")
        
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
            
            # sequence_length'i artık model config'den değil, genel training config'den almalıyız.
            # DataManager bunu zaten bildiği için, doğrudan onun içindeki değeri kullanmak en doğrusu.
            sequence_length = self.data_manager.feature_extractor.model_sequence_length
            print(f"DEBUG: RollingTrainer is using sequence_length: {sequence_length}")
            print(f"  - Training size: {len(train_data)}, Test size: {len(test_data)}, Required sequence_length: {sequence_length}")

            if len(train_data) <= sequence_length:
                print(f"  ⚠️  Cycle {cycle + 1} atlandı: Eğitim verisi ({len(train_data)}) sequence_length'ten ({sequence_length}) küçük veya eşit.")
                continue

            try:
                # Prepare data using the new feature-rich pipeline
                X_train, y_train = self.data_manager.prepare_sequences(train_data, sequence_length)
                X_test, y_test = self.data_manager.prepare_sequences(test_data, sequence_length)

                if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                    print(f"  ⚠️  Cycle {cycle + 1} atlandı: Yeterli dizi oluşturulamadı.")
                    continue
                
                # Get the number of features from the data
                input_size = X_train.shape[2]
                
                # Instantiate the model with the correct input_size
                model = self._get_model_instance(input_size=input_size)
                
                train_params = self.config.get('train_params', {})
                train_params['tqdm_desc'] = f"Cycle {cycle + 1}"
                
                print(f"  - Model eğitimi başlıyor (Özellik Sayısı: {input_size})...")
                model.train(X=X_train, y=y_train, **train_params)
                
                print(f"  - Model testi başlıyor...")
                performance = self._test_model(model, X_test, y_test)
                if not performance:
                    print(f"  - ⚠️  Cycle {cycle + 1}: Test başarısız, döngü atlanıyor.")
                    continue

                model_name = f"{self.model_type}_cycle_{cycle+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                model_path, metadata_path = self._save_model(model, model_name)
                if not model_path:
                    print(f"  - ⚠️  Cycle {cycle + 1}: Model kaydetme başarısız, döngü atlanıyor.")
                    continue

                # --- Optimized SHAP Explanation Generation ---
                shap_plot_path = self._generate_optimized_shap_explanation(model, X_train, model_name, cycle)
                
                if metadata_path:
                    with open(metadata_path, 'r+') as f:
                        metadata = json.load(f)
                        metadata.update({'performance': performance, 'cycle': cycle + 1, 'shap_plot_path': shap_plot_path})
                        f.seek(0)
                        json.dump(metadata, f, indent=4, cls=NumpyJSONEncoder)
                        f.truncate()

                self.model_registry.register_model(model_name, self.model_type, self.config, performance, model_path, metadata_path)
                cycle_results.append({'cycle': cycle + 1, 'performance': performance, 'model_path': model_path, 'shap_plot_path': shap_plot_path})
                
                print("\n  📊 Test Sonuçları:")
                for key, value in performance.items():
                    # Metrik isimlerini daha okunabilir hale getir
                    metric_name = key.replace('_', ' ').title()
                    print(f"    - {metric_name}: {value:.4f}")
                
                # Detaylı açıklama göster
                print("\n" + "="*60)
                print(explain_test_results(performance))
                print("="*60)
                
                print(f"  ✅ Cycle {cycle + 1} tamamlandı. Model: {model_name}\n")

                # Save checkpoint after a successful cycle
                self._save_checkpoint(model, model.optimizer, cycle)
                
                # Memory cleanup every few cycles
                if (cycle + 1) % CONFIG.get('memory_optimization', {}).get('cleanup_frequency', 3) == 0:
                    self._perform_memory_cleanup()

            except Exception as e:
                config_str = json.dumps(self.config, indent=4, cls=NumpyJSONEncoder)
                print(f"❌ Cycle {cycle + 1} failed for model {self.model_type} with config {config_str}")
                print(f"  - Hata: {e}")
                traceback.print_exc()
        
        self._cleanup_checkpoints()
        print(f"\n🎉 Rolling window training finished for {self.model_type}.")
        return cycle_results
    
    def _perform_memory_cleanup(self):
        """Perform aggressive memory cleanup"""
        import gc
        import psutil
        
        # Get memory usage before cleanup
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Garbage collection
        gc.collect()
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Clear matplotlib cache
        plt.close('all')
        
        # Get memory usage after cleanup
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_freed = initial_memory - final_memory
        
        print(f"  🧹 Memory cleanup: {initial_memory:.1f}MB → {final_memory:.1f}MB (freed {memory_freed:.1f}MB)")

    def _generate_optimized_shap_explanation(self, model: Any, X_train: torch.Tensor, 
                                           model_name: str, cycle: int) -> Optional[str]:
        """Generate memory-efficient SHAP explanation using optimized explainer"""
        try:
            # Get feature names from data manager
            feature_names = self.data_manager.feature_extractor.get_feature_names()
            
            # Use optimized SHAP explainer
            shap_plot_path = self.optimized_shap.generate_explanation(
                model=model,
                X_train=X_train,
                feature_names=feature_names,
                model_name=model_name
            )
            
            # Memory cleanup after SHAP generation
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return shap_plot_path
            
        except Exception as e:
            print(f"  - ❌ Optimized SHAP generation failed: {e}")
            return None

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
                
                # Veriyi DataManager ile hazırla
                X_train, y_train = self.data_manager.prepare_sequences(train_data, sequence_length)
                
                # Modeli eğit
                model.train(X=X_train, y=y_train, **train_params)
                
                performance = self._test_model(model, test_data)
                if not performance: continue

                model_name = f"{self.model_type}_incremental"
                saved_path, metadata_path = self._save_model(model, model_name)
                if not saved_path: continue

                self.model_registry.register_model(model_name, self.model_type, self.config, performance, saved_path, metadata_path)
                cycle_results.append({'cycle': cycle + 1, 'performance': performance, 'model_path': saved_path})
                print(f"  ✅ Cycle {cycle + 1} completed: MAE={performance['mae']:.4f}, Threshold Acc={performance.get('threshold_accuracy', 0):.4f}")

                # Save checkpoint after successful cycle
                self._save_checkpoint(model, model.optimizer, cycle)

            except Exception as e:
                print(f"❌ Cycle {cycle + 1} failed: {e}")
                traceback.print_exc()
        
        self._cleanup_checkpoints()
        print(f"\n🎉 Incremental training finished for {self.model_type}.")
        return cycle_results
