import pytest
import os
import sys
import shutil

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.training.pipeline_manager import PipelineManager
from src.config.settings import get_all_models

@pytest.fixture(scope="module")
def setup_test_environment():
    """Create a temporary environment for testing the pipeline."""
    # Use a temporary directory for reports and models to avoid cluttering the project
    temp_dir = "temp_test_pipeline"
    os.makedirs(os.path.join(temp_dir, "reports"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "trained_models"), exist_ok=True)
    
    # Override default paths for testing
    from src.config import settings
    settings.PATHS['models_dir'] = os.path.join(temp_dir, "trained_models")
    
    yield
    
    # Teardown: remove the temporary directory
    shutil.rmtree(temp_dir)

def test_pipeline_manager_execution(setup_test_environment, monkeypatch):
    """
    Test the full execution of the PipelineManager for a single model.
    This is an integration test that mocks the actual training to run quickly.
    """
    print("\\n🧪 Testing PipelineManager full execution...")

    # Mock the time-consuming training function
    def mock_execute_training(self):
        print(f"   MOCK: Skipping actual training for {self.model_type}.")
        # Return a realistic-looking dummy result
        dummy_performance = {'mae': 0.5, 'rmse': 0.7, 'threshold_accuracy': 0.6, 'f1': 0.65, 'recall': 0.7}
        model_name = f"{self.model_type}_cycle_1_mock"
        model_path = os.path.join(self.models_dir, f"{model_name}.pth")
        # Create a dummy model file
        with open(model_path, 'w') as f:
            f.write("dummy model")
            
        self.model_registry.register_model(
            model_name=model_name,
            model_type=self.model_type,
            config=self.config,
            performance=dummy_performance,
            model_path=model_path,
            metadata_path=model_path.replace('.pth', '_metadata.json')
        )
        return [{'cycle': 1, 'performance': dummy_performance, 'model_path': model_path}]

    monkeypatch.setattr("src.training.rolling_trainer.RollingTrainer.execute_rolling_training", mock_execute_training)

    models_to_test = get_all_models()
    if not models_to_test:
        pytest.skip("No models found in config to test.")
        
    test_model = models_to_test[0]
    
    print(f"🔧 Testing with model: {test_model}")

    pipeline = PipelineManager(models_to_train=[test_model], device='cpu')
    
    # Mock HPO to run for only 1 trial
    from src.config import settings
    settings.CONFIG['training']['hpo_trials'] = 1
    
    pipeline.run_pipeline()
    
    # Assertions
    assert test_model in pipeline.results, f"Results should contain '{test_model}'."
    assert len(pipeline.results[test_model]) > 0, "Mock training should produce a result."
    
    registered_model = pipeline.model_registry.get_best_model(test_model)
    assert registered_model is not None, "A model should be registered."
    assert os.path.exists(registered_model['model_path']), "Dummy model file should exist."
    
    print("✅ PipelineManager execution test passed.")
