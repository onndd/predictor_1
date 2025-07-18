"""
Configuration settings for the JetX Prediction System.
Loads settings from the central config.yaml file.
"""

import os
import yaml
from typing import Dict, Any, List, Optional

# --- Merkezi Konfigürasyon Yükleyici ---

def load_config() -> Dict[str, Any]:
    """
    Loads the main configuration from config.yaml.
    Searches for the file starting from the current script's directory and going up.
    """
    # Proje ana dizinini bulmak için daha sağlam bir yol
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # src/config -> src -> ana dizin
    project_root = os.path.join(current_dir, '..', '..')
    config_path = os.path.join(project_root, 'config.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- Global Konfigürasyon ve Yollar ---

# Tüm konfigürasyonu bir kez yükle
CONFIG = load_config()

# Temel yolları config'den al
_PATHS_CONFIG = CONFIG.get('paths', {})
MODELS_DIR = _PATHS_CONFIG.get('models_dir', 'trained_models')
DATABASE_PATH = _PATHS_CONFIG.get('database', 'data/jetx_data.db')
LOG_FILE = _PATHS_CONFIG.get('log_file', 'jetx_prediction.log')

# Dinamik olarak tam yolları oluştur
PATHS = {
    'database': DATABASE_PATH,
    'models_dir': MODELS_DIR,
    'log_file': LOG_FILE,
    'deep_learning_models': os.path.join(MODELS_DIR, "deep_learning"),
    'statistical_models': os.path.join(MODELS_DIR, "statistical"),
    'ensemble_models': os.path.join(MODELS_DIR, "ensemble"),
    'metadata': os.path.join(MODELS_DIR, "metadata"),
    'backup': os.path.join(MODELS_DIR, "backup")
}

# --- Konfigürasyon Erişim Fonksiyonları ---

def get_app_config() -> Dict[str, Any]:
    """Get application settings."""
    return CONFIG.get('application', {})

def get_training_config() -> Dict[str, Any]:
    """Get general training settings."""
    return CONFIG.get('training', {})

def get_prediction_config() -> Dict[str, Any]:
    """Get prediction settings."""
    return CONFIG.get('prediction', {})

def get_logging_config() -> Dict[str, Any]:
    """Get logging settings."""
    return CONFIG.get('logging', {})

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model."""
    return CONFIG.get('models', {}).get(model_name, {})

def get_model_default_params(model_name: str) -> Dict[str, Any]:
    """Gets the default parameters for a given model."""
    return CONFIG.get('models', {}).get(model_name, {}).get('default_params', {})

def get_hpo_space(model_name: str) -> Optional[Dict[str, Any]]:
    """Gets the HPO search space for a given model."""
    return CONFIG.get('models', {}).get(model_name, {}).get('hpo_space')

def get_all_models() -> List[str]:
    """Get list of all models."""
    return list(CONFIG.get('models', {}).keys())

def get_heavy_models() -> List[str]:
    """Get list of heavy models."""
    models_config = CONFIG.get('models', {})
    return [name for name, config in models_config.items() if config.get('is_heavy', False)]

def get_light_models() -> List[str]:
    """Get list of light models."""
    models_config = CONFIG.get('models', {})
    return [name for name, config in models_config.items() if not config.get('is_heavy', False)]
