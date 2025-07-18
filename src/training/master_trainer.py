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
import gc
import psutil
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Ensure src path is available for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config.settings import get_aggressive_training_profiles, PATHS, CONFIG
from src.data_processing.loader import load_data_from_sqlite
from src.training.rolling_trainer import RollingTrainer
from src.training.model_registry import ModelRegistry
from src.evaluation.reporting import generate_text_report, generate_performance_plot

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

    def _check_memory_usage(self):
        """Check and report memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        print(f"💾 Memory usage: {memory_mb:.1f} MB")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_memory_max = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"🔥 GPU memory: {gpu_memory:.1f} MB (max: {gpu_memory_max:.1f} MB)")
            
            # Clear GPU cache if usage is high
            if gpu_memory > 1000:  # 1GB threshold
                torch.cuda.empty_cache()
                print("🧹 GPU cache cleared")

    def _load_and_prepare_data(self, chunk_size: int = 1000) -> List[Any]:
        """Loads data from SQLite and prepares it in rolling chunks."""
        print("📊 Loading and preparing JetX data...")
        try:
            # Check available memory
            self._check_memory_usage()
            
            data = load_data_from_sqlite(self.db_path)
            if not data or len(data) < chunk_size:
                print("❌ Not enough data to start training.")
                return []
            
            # Adjust chunk size based on memory
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                if gpu_memory < 8000:  # Less than 8GB GPU
                    chunk_size = min(chunk_size, 500)
                    print(f"⚠️ Reduced chunk size to {chunk_size} due to limited GPU memory")
            
            chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size) if len(data[i:i + chunk_size]) >= chunk_size]
            print(f"✅ Data loaded. Created {len(chunks)} rolling chunks.")
            
            # Force garbage collection
            gc.collect()
            
            return chunks
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            traceback.print_exc()
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
                    
                    # Defensively handle potential Optuna errors and prune the trial
                    try:
                        # Explicitly check to prevent Optuna's ValueError and prune trial if invalid
                        if low_val >= high_val:
                            print(f"  [HPO WARNING] Pruning trial for param '{param}' because low >= high ({low_val} >= {high_val}).")
                            raise optuna.exceptions.TrialPruned(f"Invalid range for {param}")

                        trial_profile[param] = trial.suggest_float(
                            param,
                            low_val,
                            high_val,
                            log=space.get('log', False)
                        )
                    except (ValueError, TypeError) as e:
                        # Catch other potential errors from Optuna and prune the trial gracefully
                        print(f"  [HPO WARNING] Pruning trial due to an error suggesting param '{param}': {e}")
                        print(f"  > Values that caused the error: low={low_val}, high={high_val}, log={space.get('log', False)}")
                        raise optuna.exceptions.TrialPruned()
                elif space['type'] == 'int':
                    trial_profile[param] = trial.suggest_int(param, space['low'], space['high'])
            
            print(f"  [HPO Trial #{trial.number}] Testing params: {trial.params}")

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
                    print(f"  [HPO Trial #{trial.number}] ⚠️  No results returned from training. Pruning.")
                    raise optuna.exceptions.TrialPruned("Training returned no results.")
                
                avg_mae = np.mean([r['performance']['mae'] for r in results])
                trial.set_user_attr("avg_mae", avg_mae) # Store result for logging
                return float(avg_mae)
            except optuna.exceptions.TrialPruned as e:
                # Re-raise to let Optuna handle it
                raise e
            except Exception as e:
                error_msg = f"  [HPO Trial #{trial.number}] ❌ FAILED. Params: {trial.params}. Error: {e}"
                print(error_msg)
                traceback.print_exc(limit=1)
                trial.set_user_attr("status", "FAILED")
                trial.set_user_attr("error", str(e))
                # Return a high value to penalize this trial
                return float('inf')

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
        
        # MLflow entegrasyonu ile HPO
        mlflow.set_tag("hpo_model", model_name)
        study.optimize(objective_func, n_trials=n_trials, callbacks=[self._mlflow_callback])
        
        pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
        failed_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.FAIL])

        print(f"🏆 HPO finished for {model_name}: {len(complete_trials)} complete, {len(pruned_trials)} pruned, {len(failed_trials)} failed.")
        
        if study.best_trial:
            print(f"  - Best MAE: {study.best_value:.4f}")
            print(f"  - Best params: {study.best_trial.params}")
            # Update the original profile with the best parameters found
            optimized_profile = copy.deepcopy(profile)
            optimized_profile.update(study.best_trial.params)
        else:
            print("  ⚠️ HPO could not find a best trial. Using the base profile.")
            optimized_profile = profile # Fallback to base profile
        
        return optimized_profile
    
    def _mlflow_callback(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        """Callback to log Optuna trials to MLflow."""
        with mlflow.start_run(run_name=f"hpo_{study.study_name}_trial_{trial.number}", nested=True) as run:
            mlflow.log_params(trial.params)
            
            # Log user attributes (custom metrics or errors)
            for key, value in trial.user_attrs.items():
                mlflow.set_tag(key, value)
            
            if trial.state == optuna.trial.TrialState.COMPLETE:
                mlflow.log_metric("value", trial.value)
                mlflow.set_tag("status", "COMPLETE")
            elif trial.state == optuna.trial.TrialState.PRUNED:
                mlflow.set_tag("status", "PRUNED")
            elif trial.state == optuna.trial.TrialState.FAIL:
                mlflow.set_tag("status", "FAILED")

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

        # --- MLflow Yerel URI Düzeltmesi ---
        # MLflow'un hem deneyleri hem de artifact'leri yerel dosya sistemine kaydetmesini sağlıyoruz.
        # Bu, 'mlflow-artifacts' URI'si ile ilgili hataları önler.
        mlruns_dir = os.path.join(os.getcwd(), "mlruns")
        os.makedirs(mlruns_dir, exist_ok=True)
        
        # Hem tracking URI'sini hem de deneyi ayarlayarak MLflow'un kafasının karışmasını önlüyoruz.
        mlflow.set_tracking_uri(f"file://{os.path.abspath(mlruns_dir)}")
        mlflow.set_experiment("JetX_Training_Experiment")
        # ------------------------------------

        with mlflow.start_run(run_name="Master_Training_Run") as run:
            print(f"🚀 MLflow Run Started. Run ID: {run.info.run_id}")
            print(f"   Artifacts will be logged to: {mlflow.get_artifact_uri()}")
            mlflow.set_tag("execution_time", datetime.now().isoformat())

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
                                'best_threshold_accuracy': best_cycle['performance'].get('threshold_accuracy', 0),
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
        """Finalizes the training and generates a comprehensive report."""
        print("\n\n🎉 PIPELINE COMPLETED: All models have been trained. 🎉")
        print("="*60)
        
        # 1. Generate and print the text report
        text_report = generate_text_report(self.results)
        print(text_report)

        # 2. Generate and save the performance plot
        reports_dir = os.path.join(PATHS['models_dir'], '..', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        report_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(reports_dir, f"performance_report_{report_timestamp}.png")
        
        plot_saved = generate_performance_plot(self.results, plot_path)
        if plot_saved:
            print(f"\n📈 Performance comparison plot saved to: {plot_path}")

        # 3. Save the text report to a file
        report_path = os.path.join(reports_dir, f"training_summary_{report_timestamp}.txt")
        with open(report_path, 'w') as f:
            f.write(text_report)
        print(f"📋 Text report saved to: {report_path}")

        # 4. Export the final model registry
        registry_path = self.model_registry.export_to_json()
        print(f"📦 Full model registry exported to: {registry_path}")

        # --- En İyi Modelleri İndirme Talimatları ---
        print("\n\n📥 --- Download Best Models ---")
        print("To download the best performing model for each type, use the following paths:")
        for model_name, results in self.results.items():
            if results:
                best_cycle = min(results, key=lambda x: x['performance']['mae'])
                model_path = best_cycle['model_path']
                print(f"\n  - Model: {model_name}")
                print(f"    Path: {os.path.abspath(model_path)}")
        print("\nExample Colab download command: from google.colab import files; files.download('path/to/your/model.pth')")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Master Trainer for JetX Prediction Models")
    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help="List of model names to train (e.g., N-Beats TFT). If not provided, all models will be trained."
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="Device to use for training ('cuda' or 'cpu'). If not provided, it will auto-detect."
    )
    
    args = parser.parse_args()

    # Cihazı belirle
    if args.device:
        selected_device = args.device
    else:
        selected_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # MasterTrainer'ı başlat
    master_trainer = MasterTrainer(models_to_train=args.models, device=selected_device)
    master_trainer.run()
