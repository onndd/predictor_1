"""
Configuration settings for the JetX Prediction System
"""

import os
from typing import Dict, Any

# Database settings
DATABASE_PATH = "data/jetx_data.db"
MODELS_DIR = "trained_models"

# Model configurations
MODEL_CONFIGS = {
    # Deep Learning Models
    'n_beats': {
        'sequence_length': 200,
        'hidden_size': 256,
        'num_stacks': 3,
        'num_blocks': 3,
        'learning_rate': 0.001,
        'is_heavy': True,
        'description': 'Neural Basis Expansion Analysis for Time Series'
    },
    'tft': {
        'sequence_length': 200,
        'hidden_size': 256,
        'num_heads': 8,
        'num_layers': 2,
        'learning_rate': 0.001,
        'is_heavy': True,
        'description': 'Temporal Fusion Transformer'
    },
    'informer': {
        'sequence_length': 200,
        'd_model': 512,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'learning_rate': 0.001,
        'is_heavy': True,
        'description': 'Long Sequence Time Series Forecasting'
    },
    'autoformer': {
        'sequence_length': 200,
        'd_model': 512,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'learning_rate': 0.001,
        'is_heavy': True,
        'description': 'Auto-Correlation for Time Series'
    },
    'pathformer': {
        'sequence_length': 200,
        'd_model': 512,
        'n_heads': 8,
        'num_layers': 6,
        'path_length': 3,
        'learning_rate': 0.001,
        'is_heavy': True,
        'description': 'Path-based Attention for Time Series'
    },
    # Statistical Models (Light)
    'light_ensemble': {
        'is_heavy': False,
        'description': 'Ensemble of Light Statistical Models'
    },
    'hybrid_predictor': {
        'is_heavy': False,
        'description': 'Hybrid Statistical Predictor'
    },
    'crash_detector': {
        'is_heavy': False,
        'description': 'Crash Detection Model'
    }
}

# Training settings
TRAINING_CONFIG = {
    'default_epochs': 100,
    'batch_size': 32,
    'validation_split': 0.2,
    'min_data_points': 500,
    'sequence_length': 200,
    'threshold': 1.5
}

# Prediction settings
PREDICTION_CONFIG = {
    'confidence_thresholds': {
        'high': 0.7,
        'medium': 0.5,
        'low': 0.3
    },
    'ensemble_weights': {
        'deep_learning': 0.6,
        'statistical': 0.4
    }
}

# Application settings
APP_CONFIG = {
    'auto_train_heavy_models': False,  # Default: manual training
    'max_display_values': 50,
    'update_frequency': 15,  # Retrain every N new values
    'save_models_automatically': True
}

# File paths
PATHS = {
    'database': DATABASE_PATH,
    'models_dir': MODELS_DIR,
    'deep_learning_models': os.path.join(MODELS_DIR, "deep_learning"),
    'statistical_models': os.path.join(MODELS_DIR, "statistical"),
    'ensemble_models': os.path.join(MODELS_DIR, "ensemble"),
    'metadata': os.path.join(MODELS_DIR, "metadata"),
    'backup': os.path.join(MODELS_DIR, "backup")
}

# Logging settings
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'jetx_prediction.log'
}

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model"""
    return MODEL_CONFIGS.get(model_name, {})

def get_heavy_models() -> list:
    """Get list of heavy models"""
    return [name for name, config in MODEL_CONFIGS.items() if config.get('is_heavy', False)]

def get_light_models() -> list:
    """Get list of light models"""
    return [name for name, config in MODEL_CONFIGS.items() if not config.get('is_heavy', False)]

def get_all_models() -> list:
    """Get list of all models"""
    return list(MODEL_CONFIGS.keys())