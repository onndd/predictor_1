"""
Master Trainer - High-level training orchestrator for the JetX Prediction System.

This module provides a simplified interface for running the complete training pipeline,
serving as a wrapper around PipelineManager with additional convenience methods.
"""

import os
import sys
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional

# Ensure src path is available for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.training.pipeline_manager import PipelineManager
from src.config.settings import get_all_models, get_light_models, get_heavy_models, CONFIG
from src.utils.pre_flight_check import PreFlightChecker

class MasterTrainer:
    """
    Master training orchestrator that provides a high-level interface
    for training multiple models with different strategies.
    """
    
    def __init__(self, device: str = 'cpu', enable_pre_flight_check: bool = True):
        self.device = device
        self.enable_pre_flight_check = enable_pre_flight_check
        self.pre_flight_checker = PreFlightChecker() if enable_pre_flight_check else None
        self.training_results = {}
        
        print("🚀 MasterTrainer initialized")
        print(f"   - Device: {device}")
        print(f"   - Pre-flight check: {enable_pre_flight_check}")

    def run_pre_flight_check(self) -> bool:
        """
        Run comprehensive pre-flight checks before training.
        
        Returns:
            bool: True if all checks pass, False otherwise
        """
        if not self.enable_pre_flight_check or not self.pre_flight_checker:
            print("⚠️ Pre-flight check skipped")
            return True
            
        print("🔍 Running pre-flight checks...")
        try:
            check_results = self.pre_flight_checker.run_all_checks()
            
            if check_results['overall_status']:
                print("✅ All pre-flight checks passed!")
                return True
            else:
                print("❌ Pre-flight checks failed:")
                for check_name, result in check_results['checks'].items():
                    if not result['status']:
                        print(f"   - {check_name}: {result['message']}")
                return False
                
        except Exception as e:
            print(f"❌ Pre-flight check error: {e}")
            return False

    def train_all_models(self, force_training: bool = False) -> Dict[str, List[str]]:
        """
        Train all available models using the pipeline manager.
        
        Args:
            force_training: If True, skip pre-flight checks
            
        Returns:
            Dictionary with training results for each model type
        """
        if not force_training and not self.run_pre_flight_check():
            print("❌ Training aborted due to failed pre-flight checks")
            return {}
            
        models_to_train = get_all_models()
        print(f"🎯 Training all models: {models_to_train}")
        
        return self._execute_training_pipeline(models_to_train)

    def train_light_models(self, force_training: bool = False) -> Dict[str, List[str]]:
        """
        Train only light models (faster training, lower resource usage).
        
        Args:
            force_training: If True, skip pre-flight checks
            
        Returns:
            Dictionary with training results for light models
        """
        if not force_training and not self.run_pre_flight_check():
            print("❌ Training aborted due to failed pre-flight checks")
            return {}
            
        models_to_train = get_light_models()
        print(f"⚡ Training light models: {models_to_train}")
        
        return self._execute_training_pipeline(models_to_train)

    def train_heavy_models(self, force_training: bool = False) -> Dict[str, List[str]]:
        """
        Train only heavy models (longer training, higher resource usage).
        
        Args:
            force_training: If True, skip pre-flight checks
            
        Returns:
            Dictionary with training results for heavy models
        """
        if not force_training and not self.run_pre_flight_check():
            print("❌ Training aborted due to failed pre-flight checks")
            return {}
            
        models_to_train = get_heavy_models()
        print(f"🏋️ Training heavy models: {models_to_train}")
        
        return self._execute_training_pipeline(models_to_train)

    def train_specific_models(self, model_names: List[str], force_training: bool = False) -> Dict[str, List[str]]:
        """
        Train specific models by name.
        
        Args:
            model_names: List of model names to train
            force_training: If True, skip pre-flight checks
            
        Returns:
            Dictionary with training results for specified models
        """
        if not force_training and not self.run_pre_flight_check():
            print("❌ Training aborted due to failed pre-flight checks")
            return {}
            
        available_models = get_all_models()
        invalid_models = [name for name in model_names if name not in available_models]
        
        if invalid_models:
            print(f"❌ Invalid model names: {invalid_models}")
            print(f"Available models: {available_models}")
            return {}
            
        print(f"🎯 Training specific models: {model_names}")
        return self._execute_training_pipeline(model_names)

    def _execute_training_pipeline(self, models_to_train: List[str]) -> Dict[str, List[str]]:
        """
        Execute the training pipeline for specified models.
        
        Args:
            models_to_train: List of model names to train
            
        Returns:
            Dictionary with training results
        """
        if not models_to_train:
            print("⚠️ No models to train")
            return {}
            
        print("="*60)
        print(f"🚀 MASTER TRAINER: Starting training pipeline")
        print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Models to train: {models_to_train}")
        print(f"💻 Device: {self.device}")
        print("="*60)
        
        try:
            # Initialize pipeline manager
            pipeline_manager = PipelineManager(
                models_to_train=models_to_train,
                device=self.device
            )
            
            # Execute training
            trained_model_paths = pipeline_manager.run_pipeline()
            
            # Store results
            self.training_results[datetime.now().strftime('%Y%m%d_%H%M%S')] = {
                'models_trained': models_to_train,
                'trained_paths': trained_model_paths,
                'device_used': self.device,
                'timestamp': datetime.now().isoformat()
            }
            
            # Summary
            print("\n" + "="*60)
            print("🎉 MASTER TRAINER: Training pipeline completed!")
            print(f"✅ Successfully trained {len(trained_model_paths)} models")
            print(f"📁 Model paths: {trained_model_paths}")
            print("="*60)
            
            return {
                'trained_models': models_to_train,
                'model_paths': trained_model_paths,
                'device': self.device,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"\n❌ MASTER TRAINER: Training pipeline failed!")
            print(f"Error: {str(e)}")
            traceback.print_exc()
            
            return {
                'trained_models': [],
                'model_paths': [],
                'device': self.device,
                'status': 'failed',
                'error': str(e)
            }

    def get_training_history(self) -> Dict[str, Any]:
        """
        Get the complete training history.
        
        Returns:
            Dictionary with all training sessions
        """
        return self.training_results

    def get_latest_training_result(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent training result.
        
        Returns:
            Dictionary with latest training results or None
        """
        if not self.training_results:
            return None
            
        latest_key = max(self.training_results.keys())
        return self.training_results[latest_key]

    def quick_train(self, model_type: str = 'light', device: str = None) -> Dict[str, List[str]]:
        """
        Quick training method for common use cases.
        
        Args:
            model_type: 'light', 'heavy', or 'all'
            device: Override device setting
            
        Returns:
            Training results
        """
        if device:
            self.device = device
            
        print(f"🚀 Quick training: {model_type} models on {self.device}")
        
        if model_type == 'light':
            return self.train_light_models(force_training=True)
        elif model_type == 'heavy':
            return self.train_heavy_models(force_training=True)
        elif model_type == 'all':
            return self.train_all_models(force_training=True)
        else:
            print(f"❌ Invalid model_type: {model_type}. Use 'light', 'heavy', or 'all'")
            return {}

# Convenience functions for easy usage
def train_all(device: str = 'cpu') -> Dict[str, List[str]]:
    """Quick function to train all models."""
    trainer = MasterTrainer(device=device)
    return trainer.train_all_models()

def train_light(device: str = 'cpu') -> Dict[str, List[str]]:
    """Quick function to train light models."""
    trainer = MasterTrainer(device=device)
    return trainer.train_light_models()

def train_heavy(device: str = 'cpu') -> Dict[str, List[str]]:
    """Quick function to train heavy models."""
    trainer = MasterTrainer(device=device)
    return trainer.train_heavy_models()

def quick_train(model_type: str = 'light', device: str = 'cpu') -> Dict[str, List[str]]:
    """Quick function for common training scenarios."""
    trainer = MasterTrainer(device=device)
    return trainer.quick_train(model_type, device)

if __name__ == "__main__":
    # Example usage
    print("🚀 Master Trainer - JetX Prediction System")
    print("Available quick functions:")
    print("  - train_all(device='cpu')")
    print("  - train_light(device='cpu')")
    print("  - train_heavy(device='cpu')")
    print("  - quick_train(model_type='light', device='cpu')")
