"""
End-to-End Training Pipeline Manager for the JetX Prediction System.

This module provides a high-level PipelineManager class that orchestrates
the entire ML workflow, including data loading, hyperparameter optimization (HPO),
rolling window training, model registration, and reporting.
"""

import os
import sys
import traceback
from datetime import datetime
import mlflow
import optuna
import copy
import numpy as np
import shutil
from typing import Dict, List, Any, Optional

# Ensure src path is available for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config.settings import get_model_default_params, get_hpo_space, CONFIG
from src.data_processing.manager import DataManager
from src.training.rolling_trainer import RollingTrainer
from src.training.model_registry import ModelRegistry
from src.evaluation.reporting import generate_text_report, generate_performance_plot

class PipelineManager:
    """
    Orchestrates the entire training pipeline from data loading to reporting.
    """
    def __init__(self, models_to_train: List[str], device: str = 'cpu'):
        self.models_to_train = models_to_train
        self.device = device
        self.data_manager = DataManager()
        self.model_registry = ModelRegistry()
        self.results: Dict[str, List[Dict[str, Any]]] = {}

    def run_pipeline(self) -> List[str]:
        """
        Executes the full training pipeline and returns paths of trained models.
        """
        print("🚀 PIPELINE MANAGER: Starting end-to-end training pipeline.")
        print("="*60)
        trained_model_paths = []

        # 1. Load Data
        all_data = self.data_manager.get_all_data()
        if not all_data or len(all_data) < CONFIG.get('training', {}).get('min_data_points', 500):
            print("❌ Pipeline stopped: Not enough data for training.")
            return trained_model_paths

        rolling_chunks = self.data_manager.get_rolling_chunks(all_data, chunk_size=1000)
        if not rolling_chunks:
            print("❌ Pipeline stopped: Not enough data to create rolling chunks.")
            return trained_model_paths

        # 2. Setup MLflow
        self._setup_mlflow()

        with mlflow.start_run(run_name="JetX_Training_Pipeline") as run:
            print(f"🚀 MLflow Run Started: {run.info.run_id}")
            
            for model_name in self.models_to_train:
                model_path = self._process_model(model_name, rolling_chunks)
                if model_path:
                    trained_model_paths.append(model_path)

        # 4. Final Report
        self._finalize_and_report()
        return trained_model_paths

    def _process_model(self, model_name: str, chunks: List[List[float]]) -> Optional[str]:
        """Processes a single model and returns the path of the best trained artifact."""
        print(f"\n\n--- Processing Model: {model_name} ---")
        
        base_profile = get_model_default_params(model_name)
        
        # HPO
        optimized_profile = self._run_hpo(model_name, base_profile, chunks)
        
        # Final Training
        return self._run_final_training(model_name, optimized_profile, chunks)

    def _run_hpo(self, model_name: str, profile: Dict, chunks: List) -> Dict:
        """Runs HPO for a given model."""
        print(f"🔬 Starting HPO for {model_name}...")
        hpo_space = get_hpo_space(model_name)
        if not hpo_space:
            print(f"⚠️ No HPO space for {model_name}. Using default parameters.")
            return profile

        study = optuna.create_study(direction='minimize')
        n_trials = CONFIG.get('training', {}).get('hpo_trials', 15)
        
        study.optimize(
            lambda trial: self._hpo_objective(trial, model_name, profile, chunks),
            n_trials=n_trials
        )
        
        if study.best_trial:
            print(f"🏆 HPO Best MAE for {model_name}: {study.best_value:.4f}")
            optimized_profile = copy.deepcopy(profile)
            optimized_profile.update(study.best_trial.params)
            return optimized_profile
        else:
            print("⚠️ HPO could not find a best trial. Using default parameters.")
            return profile

    def _hpo_objective(self, trial: optuna.Trial, model_name: str, profile: Dict, chunks: List) -> float:
        """Objective function for Optuna HPO."""
        trial_profile = copy.deepcopy(profile)
        hpo_space = get_hpo_space(model_name)
        
        for param, space in hpo_space.items():
            if space['type'] == 'categorical':
                trial_profile[param] = trial.suggest_categorical(param, space['choices'])
            elif space['type'] == 'float':
                trial_profile[param] = trial.suggest_float(param, space['low'], space['high'], log=space.get('log', False))
            elif space['type'] == 'int':
                trial_profile[param] = trial.suggest_int(param, space['low'], space['high'])

        # Use a smaller subset of data for faster HPO
        hpo_chunks = chunks[:min(3, len(chunks))]
        trainer = RollingTrainer(ModelRegistry(), hpo_chunks, model_name, trial_profile, self.device)
        
        try:
            results = trainer.execute_rolling_training()
            if not results:
                raise optuna.exceptions.TrialPruned("Training returned no results.")
            
            avg_mae = np.mean([r['performance']['mae'] for r in results])
            return float(avg_mae)
        except Exception as e:
            print(f"❌ HPO trial failed for {model_name}: {e}")
            return float('inf')

    def _run_final_training(self, model_name: str, config: Dict, chunks: List) -> Optional[str]:
        """Runs final training and returns the path of the best model artifact."""
        print(f"🎓 Starting final training for {model_name}...")
        best_model_path = None
        with mlflow.start_run(run_name=f"Final_Training_{model_name}", nested=True):
            mlflow.log_params(config)
            mlflow.set_tag("model_type", model_name)

            trainer = RollingTrainer(self.model_registry, chunks, model_name, config, self.device)
            model_results = trainer.execute_rolling_training()
            self.results[model_name] = model_results
            
            if model_results:
                best_cycle = min(model_results, key=lambda x: x['performance']['mae'])
                mlflow.log_metrics({
                    'best_mae': best_cycle['performance']['mae'],
                    'best_threshold_accuracy': best_cycle['performance'].get('threshold_accuracy', 0),
                })
                best_model_path = best_cycle['model_path']
                mlflow.log_artifact(best_model_path)
                print(f"✅ Final training for {model_name} completed.")
        return best_model_path

    def _setup_mlflow(self):
        """Sets up the local MLflow tracking URI and experiment."""
        mlruns_dir = os.path.join(os.getcwd(), "mlruns")
        os.makedirs(mlruns_dir, exist_ok=True)
        mlflow.set_tracking_uri(f"file://{os.path.abspath(mlruns_dir)}")
        mlflow.set_experiment("JetX_Main_Pipeline")

    def _finalize_and_report(self):
        """Generates and saves the final reports."""
        print("\n\n🎉 PIPELINE COMPLETED. Generating reports...")
        print("="*60)
        
        text_report = generate_text_report(self.results)
        print(text_report)

        reports_dir = os.path.join(os.getcwd(), 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(reports_dir, f"performance_report_{timestamp}.png")
        report_path = os.path.join(reports_dir, f"training_summary_{timestamp}.txt")

        if generate_performance_plot(self.results, plot_path):
            print(f"\n📈 Performance plot saved to: {plot_path}")
        
        with open(report_path, 'w') as f:
            f.write(text_report)
        print(f"📋 Text report saved to: {report_path}")

        registry_path = self.model_registry.export_to_json()
        print(f"📦 Model registry exported to: {registry_path}")

        # Copy the best model to a final_model directory
        self._copy_best_model_to_final_dir()

    def _copy_best_model_to_final_dir(self):
        """Identifies and copies the best performing model to a dedicated directory."""
        print("\n" + "-"*60)
        print("🔍 Identifying best model to copy to final directory...")
        
        best_model_info = self.model_registry.get_best_overall_model(metric='f1')
        
        if not best_model_info:
            print("⚠️ Could not determine the best model. Skipping copy.")
            return

        final_model_dir = os.path.join(os.getcwd(), 'final_model')
        os.makedirs(final_model_dir, exist_ok=True)

        try:
            model_path = best_model_info['model_path']
            metadata_path = best_model_info['metadata_path']

            if os.path.exists(model_path):
                shutil.copy(model_path, final_model_dir)
                print(f"✅ Copied best model to: {os.path.join(final_model_dir, os.path.basename(model_path))}")
            else:
                print(f"❌ Source model file not found: {model_path}")

            if os.path.exists(metadata_path):
                shutil.copy(metadata_path, final_model_dir)
                print(f"✅ Copied metadata to: {os.path.join(final_model_dir, os.path.basename(metadata_path))}")
            else:
                print(f"❌ Source metadata file not found: {metadata_path}")

        except Exception as e:
            print(f"❌ Failed to copy best model files: {e}")
