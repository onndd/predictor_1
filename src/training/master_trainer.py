"""
Master Trainer for the JetX Prediction System

This module orchestrates the entire training, validation, and reporting process.
It is designed to be called from an execution environment like a Colab notebook.
"""

import os
import sys
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
import mlflow
import torch
import optuna
import copy
import numpy as np

# Ensure src path is available for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config.settings import get_aggressive_training_profiles, PATHS, CONFIG
from src.data_processing.loader import load_data_from_sqlite
from src.training.rolling_trainer import RollingTrainer
from src.training.model_registry import ModelRegistry

class MasterTrainer:
    """
    Orchestrates the end-to-end model training process.
    """
    def __init__(self, models_to_train: Optional[List[str]] = None, device: str = 'cpu'):
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

    def _load_and_prepare_data(self, chunk_size: int = 1000) -> List[Any]:
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

    def _create_objective_function(self, model_name: str, profile: Dict[str, Any], chunks: List[Any]) -> Any:
        """Creates the objective function for Optuna study."""
        
        hpo_space = CONFIG.get('hpo_search_space', {}).get(model_name)
        if not hpo_space:
            return None

        def objective(trial: optuna.Trial) -> float:
            """The objective function to be minimized by Optuna."""
            # Deepcopy to avoid modifying the original profile during trials
            trial_profile = copy.deepcopy(profile)
            
            # Suggest hyperparameters based on the search space in config.yaml
            for param, space in hpo_space.items():
                if space['type'] == 'categorical':
                    trial_profile[param] = trial.suggest_categorical(param, space['choices'])
                elif space['type'] == 'float':
                    low_val = float(space['low'])
                    high_val = float(space['high'])
                    print(f"  [HPO DEBUG] Param: {param}, Low: {low_val}, High: {high_val}, Type Low: {type(low_val)}, Type High: {type(high_val)}")
                    trial_profile[param] = trial.suggest_float(
                        param,
                        low_val,
                        high_val,
                        log=space.get('log', False)
                    )
                elif space['type'] == 'int':
                    trial_profile[param] = trial.suggest_int(param, space['low'], space['high'])
            
            print(f"  [HPO Trial] Testing params: {trial.params}")

            # Use a smaller subset of data for faster HPO
            hpo_chunks = chunks[:min(3, len(chunks))] # Use first 3 chunks for speed

            trainer = RollingTrainer(
                model_registry=ModelRegistry(), # Use a temporary registry for HPO
                chunks=hpo_chunks,
                model_type=model_name,
                config=trial_profile,
                device=self.device
            )
            
            try:
                results = trainer.execute_rolling_training()
                if not results:
                    return float('inf') # Return a large value if training fails
                
                # We want to minimize MAE
                avg_mae = np.mean([r['performance']['mae'] for r in results])
                return float(avg_mae)
            except Exception as e:
                print(f"  [HPO Trial] Exception: {e}")
                return float('inf') # Penalize failed trials

        return objective

    def _run_hpo(self, model_name: str, profile: Dict[str, Any], chunks: List[Any]) -> Dict[str, Any]:
        """Runs Hyperparameter Optimization for a given model."""
        print(f"🔬 Starting HPO for {model_name}...")
        
        objective_func = self._create_objective_function(model_name, profile, chunks)
        if not objective_func:
            print(f"⚠️ No HPO search space defined for {model_name}. Skipping HPO.")
            return profile

        study = optuna.create_study(direction='minimize')
        n_trials = CONFIG.get('hpo_search_space', {}).get('n_trials', 20)
        
        study.optimize(objective_func, n_trials=n_trials)
        
        print(f"🏆 HPO finished for {model_name}. Best MAE: {study.best_value:.4f}")
        print(f"  - Best params: {study.best_params}")
        
        # Update the original profile with the best parameters found
        optimized_profile = copy.deepcopy(profile)
        optimized_profile.update(study.best_params)
        
        return optimized_profile

    def run(self):
        """
        Executes the entire automated training pipeline, including HPO.
        """
        print("🚀 MASTER TRAINER: Starting automated training pipeline.")
        print("="*60)

        rolling_chunks = self._load_and_prepare_data()
        if not rolling_chunks:
            print("❌ Pipeline stopped due to data loading failure.")
            return

        with mlflow.start_run(run_name="Master_Training_Run"):
            mlflow.set_tag("execution_time", datetime.now().isoformat())
            print(f"🚀 MLflow Run Started. Check UI at http://127.0.0.1:5000")

            for model_name in self.models_to_train:
                print(f"\n\n--- Processing Model: {model_name} ---")
                
                # 1. Get base profile
                base_profile = self.training_profiles[model_name]
                
                # 2. Run HPO to get optimized profile
                optimized_profile = self._run_hpo(model_name, base_profile, rolling_chunks)

                # 3. Run final training with optimized profile
                print(f"🎓 Starting final training for {model_name} with optimized parameters...")
                try:
                    with mlflow.start_run(run_name=f"Final_{model_name}", nested=True) as final_run:
                        mlflow.log_params(optimized_profile)
                        mlflow.set_tag("model_type", model_name)

                        trainer = RollingTrainer(
                            model_registry=self.model_registry,
                            chunks=rolling_chunks,
                            model_type=model_name,
                            config=optimized_profile,
                            device=self.device
                        )
                        
                        model_results = trainer.execute_rolling_training()
                        self.results[model_name] = model_results
                        
                        if model_results:
                            best_cycle = min(model_results, key=lambda x: x['performance']['mae'])
                            mlflow.log_metrics({
                                'best_mae': best_cycle['performance']['mae'],
                                'best_accuracy': best_cycle['performance']['accuracy'],
                                'best_rmse': best_cycle['performance']['rmse']
                            })
                            mlflow.log_artifact(best_cycle['model_path'])
                            print(f"✅ Successfully completed final training for {model_name}.")
                        else:
                            print(f"⚠️ Final training for {model_name} produced no results.")

                except Exception as e:
                    print(f"❌ An error occurred during final training for {model_name}: {e}")
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
    # Train only one model for a quick test of the full pipeline
    master_trainer = MasterTrainer(models_to_train=['N-Beats'], device=selected_device)
    master_trainer.run()