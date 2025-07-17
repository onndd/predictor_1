"""
Master Trainer for the JetX Prediction System

This module orchestrates the entire training, validation, and reporting process.
It is designed to be called from an execution environment like a Colab notebook.
"""

import os
import sys
import traceback
from datetime import datetime
from typing import Dict, List, Any

# Ensure src path is available for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config.settings import get_aggressive_training_profiles, PATHS
from src.data_processing.loader import load_data_from_sqlite
from src.training.rolling_trainer import RollingTrainer
from src.training.model_registry import ModelRegistry

class MasterTrainer:
    """
    Orchestrates the end-to-end model training process.
    """
    def __init__(self, models_to_train: List[str] = None, device: str = 'cpu'):
        """
        Initializes the MasterTrainer.

        Args:
            models_to_train: A list of model names to train. If None, trains all models from config.
            device: The device to run training on ('cpu' or 'cuda').
        """
        self.training_profiles = get_aggressive_training_profiles()
        self.device = device
        print(f"🔌 MasterTrainer initialized to run on device: {self.device}")
        
        if models_to_train:
            self.models_to_train = [m for m in models_to_train if m in self.training_profiles]
        else:
            self.models_to_train = list(self.training_profiles.keys())
            
        self.model_registry = ModelRegistry()
        self.db_path = PATHS['database']
        self.results = {}

    def _load_and_prepare_data(self, chunk_size: int = 1000) -> List[List[float]]:
        """Loads data from SQLite and prepares it in rolling chunks."""
        print("📊 Loading and preparing JetX data...")
        try:
            data = load_data_from_sqlite(self.db_path)
            if not data or len(data) < chunk_size:
                print("❌ Not enough data to start training.")
                return []
            
            chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size) if len(data[i:i + chunk_size]) >= chunk_size]
            print(f"✅ Data loaded. Created {len(chunks)} rolling chunks.")
            return chunks
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return []

    def run(self):
        """
        Executes the entire automated training pipeline.
        """
        print("🚀 MASTER TRAINER: Starting automated training pipeline.")
        print("="*60)

        rolling_chunks = self._load_and_prepare_data()
        if not rolling_chunks:
            print("❌ Pipeline stopped due to data loading failure.")
            return

        for model_name in self.models_to_train:
            print(f"\n\n--- Training Model: {model_name} ---")
            profile = self.training_profiles[model_name]
            
            print("⚙️ Using Training Profile:")
            for key, value in profile.items():
                print(f"   - {key}: {value}")

            try:
                trainer = RollingTrainer(
                    model_registry=self.model_registry,
                    chunks=rolling_chunks,
                    model_type=model_name,
                    config=profile,
                    device=self.device
                )
                
                model_results = trainer.execute_rolling_training()
                self.results[model_name] = model_results
                print(f"✅ Successfully completed training for {model_name}.")

            except Exception as e:
                print(f"❌ An error occurred during training for {model_name}: {e}")
                traceback.print_exc()
        
        self._finalize_and_report()

    def _finalize_and_report(self):
        """Finalizes the training and prints a summary report."""
        print("\n\n🎉 PIPELINE COMPLETED: All models have been trained. 🎉")
        print("="*60)
        print("\n📊 Final Training Summary:")

        if not self.results:
            print("No models were trained successfully.")
            return

        for model_name, model_results in self.results.items():
            if not model_results:
                print(f"\n--- {model_name} ---")
                print("  No successful cycles.")
                continue

            best_cycle = min(model_results, key=lambda x: x['performance']['mae'])
            print(f"\n--- {model_name} (Best Cycle: {best_cycle['cycle']}) ---")
            print(f"  - Best MAE: {best_cycle['performance']['mae']:.4f}")
            print(f"  - Best Accuracy: {best_cycle['performance']['accuracy']:.4f}")
            print(f"  - Model Path: {best_cycle['model_path']}")

        print("\n\n💾 All trained models and metadata are saved in:", PATHS['models_dir'])
        
        # Export final registry
        registry_path = self.model_registry.export_to_json()
        print(f"📋 Full model registry exported to: {registry_path}")

if __name__ == '__main__':
    # This allows running the trainer directly for testing
    selected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    master_trainer = MasterTrainer(models_to_train=['N-Beats'], device=selected_device) # Train only one model for a quick test
    master_trainer.run()