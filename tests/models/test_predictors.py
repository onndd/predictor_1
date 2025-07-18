import pytest
import torch
import numpy as np
import sys
import os

# Add src to path to allow for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from models.deep_learning.n_beats.n_beats_model import NBeatsPredictor
from models.deep_learning.tft.enhanced_tft_model import EnhancedTFTPredictor
from models.sequential.enhanced_lstm_pytorch import EnhancedLSTMPredictor
from config.settings import get_model_default_params

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def sample_data():
    """Generate a small, consistent sample of time series data for testing."""
    return np.random.uniform(1.0, 5.0, 400).tolist()

@pytest.fixture(scope="module")
def nbeats_config():
    """Get default config for N-Beats."""
    return get_model_default_params('N-Beats')

@pytest.fixture(scope="module")
def tft_config():
    """Get default config for TFT."""
    return get_model_default_params('TFT')

@pytest.fixture(scope="module")
def lstm_config():
    """Get default config for LSTM."""
    return get_model_default_params('LSTM')

# --- Predictor Test Functions ---

def run_predictor_test_suite(PredictorClass, config, sample_data):
    """A generic test suite to run for any predictor class."""
    
    # 1. Test Initialization
    print(f"\\n🧪 Testing Initialization for {PredictorClass.__name__}...")
    predictor = PredictorClass(device='cpu', **config)
    assert predictor.model is not None, "Model should be initialized."
    assert predictor.is_trained is False, "Model should not be trained initially."
    print("✅ Initialization successful.")

    # 2. Test Data Preparation
    print(f"🧪 Testing prepare_sequences for {PredictorClass.__name__}...")
    X, y = predictor.prepare_sequences(sample_data)
    assert isinstance(X, torch.Tensor), "X should be a torch.Tensor."
    assert isinstance(y, torch.Tensor), "y should be a torch.Tensor."
    assert X.shape[0] == y.shape[0], "X and y should have the same number of samples."
    assert X.shape[1] == config['sequence_length'], "Sequence length should match config."
    print("✅ prepare_sequences successful.")

    # 3. Test Training on a single batch
    print(f"🧪 Testing train method (1 epoch) for {PredictorClass.__name__}...")
    train_params = config.get('train_params', {})
    history = predictor.train(
        data=sample_data, 
        epochs=1, 
        batch_size=train_params.get('batch_size', 16), 
        validation_split=train_params.get('validation_split', 0.2),
        verbose=False
    )
    assert isinstance(history, dict), "Training should return a history dict."
    assert 'train_losses' in history and 'val_losses' in history, "History should contain losses."
    assert predictor.is_trained is True, "Model should be marked as trained."
    print("✅ train method successful.")

    # 4. Test Prediction
    print(f"🧪 Testing predict_with_confidence for {PredictorClass.__name__}...")
    test_sequence = sample_data[:config['sequence_length']]
    prediction = predictor.predict_with_confidence(test_sequence)
    assert isinstance(prediction, tuple) and len(prediction) == 3, "Prediction should be a tuple of 3 floats."
    assert all(isinstance(p, float) for p in prediction), "All prediction elements should be floats."
    assert not any(np.isnan(p) for p in prediction), "Prediction should not contain NaNs."
    print("✅ predict_with_confidence successful.")

# --- Pytest Test Cases ---

def test_nbeats_predictor(nbeats_config, sample_data):
    """Run the full test suite for NBeatsPredictor."""
    run_predictor_test_suite(NBeatsPredictor, nbeats_config, sample_data)

def test_tft_predictor(tft_config, sample_data):
    """Run the full test suite for EnhancedTFTPredictor."""
    run_predictor_test_suite(EnhancedTFTPredictor, tft_config, sample_data)

def test_lstm_predictor(lstm_config, sample_data):
    """Run the full test suite for EnhancedLSTMPredictor."""
    run_predictor_test_suite(EnhancedLSTMPredictor, lstm_config, sample_data)