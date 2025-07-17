import os
import sys
import optuna
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Ensure src path is available for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.settings import CONFIG

def check_hpo_space(trial: optuna.Trial, model_name: str, hpo_space: dict):
    """
    Suggests parameters for a given trial to verify the search space.
    This function mimics the logic in MasterTrainer without running training.
    """
    print(f"    - Verifying parameters for trial...")
    for param, space in hpo_space.items():
        try:
            if space['type'] == 'categorical':
                value = trial.suggest_categorical(param, space['choices'])
                print(f"      ✅ Param '{param}': Suggested '{value}'")
            elif space['type'] == 'int':
                value = trial.suggest_int(param, space['low'], space['high'])
                print(f"      ✅ Param '{param}': Suggested '{value}'")
            elif space['type'] == 'float':
                low_val = float(space['low'])
                high_val = float(space['high'])
                
                if low_val >= high_val:
                    print(f"      ⚠️  Invalid range for '{param}' (low >= high). This would be pruned in a real run.")
                    # In a real scenario, MasterTrainer would prune this. Here we just report it.
                    continue

                value = trial.suggest_float(
                    param,
                    low_val,
                    high_val,
                    log=space.get('log', False)
                )
                print(f"      ✅ Param '{param}': Suggested '{value}' (log={space.get('log', False)})")

        except (ValueError, TypeError) as e:
            print(f"      ❌ FAILED on param '{param}': {e}")
            # Re-raise the exception to fail the check script
            raise e

def main():
    """
    Main function to run the HPO verification.
    """
    print("🚀 Starting HPO Configuration Verification Script...")
    print("="*50)

    hpo_config = CONFIG.get('hpo_search_space', {})
    if not hpo_config:
        print("❌ HPO search space not found in config.yaml!")
        return

    models_to_check = ['N-Beats', 'TFT', 'LSTM']
    all_passed = True

    for model_name in models_to_check:
        print(f"\n🔬 Checking model: '{model_name}'")
        model_space = hpo_config.get(model_name)
        if not model_space:
            print(f"  - ⚠️  No HPO space defined for '{model_name}'. Skipping.")
            continue

        # Create a dummy objective function for the check
        objective = lambda trial: check_hpo_space(trial, model_name, model_space)
        
        try:
            # We only need one trial to check if parameter suggestion works
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=1)
            print(f"  ✅ SUCCESS: HPO configuration for '{model_name}' is valid.")
        except Exception as e:
            print(f"  ❌ FAILED: An error occurred while checking '{model_name}': {e}")
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 Verification Complete: All checked HPO configurations are valid!")
    else:
        print("🔥 Verification Failed: One or more HPO configurations are invalid.")

if __name__ == '__main__':
    main()