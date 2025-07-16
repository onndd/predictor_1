"""
Path Management Utility for Cross-Platform Compatibility
Handles path resolution for different environments (Windows, Linux, macOS, Colab)
"""

import os
import sys
import platform
from pathlib import Path
from typing import Optional, Union
import warnings

class PathManager:
    """
    Cross-platform path management utility
    """
    
    def __init__(self):
        self.platform = platform.system()
        self.is_colab = self._detect_colab()
        self.project_root = self._find_project_root()
        
    def _detect_colab(self) -> bool:
        """Detect if running in Google Colab"""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def _find_project_root(self) -> Path:
        """Find the project root directory"""
        current_dir = Path(__file__).parent
        
        # Look for common project indicators
        indicators = [
            'requirements.txt',
            'requirements_enhanced.txt',
            'src',
            'colab',
            'trained_models',
            '.git'
        ]
        
        # Search upward for project root
        for parent in [current_dir] + list(current_dir.parents):
            if any((parent / indicator).exists() for indicator in indicators):
                return parent
        
        # Fallback to current directory
        return current_dir
    
    def get_project_root(self) -> Path:
        """Get project root directory"""
        return self.project_root
    
    def get_models_dir(self) -> Path:
        """Get models directory"""
        models_dir = self.project_root / "trained_models"
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir
    
    def get_data_dir(self) -> Path:
        """Get data directory"""
        data_dir = self.project_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    
    def get_colab_dir(self) -> Path:
        """Get colab directory"""
        colab_dir = self.project_root / "colab"
        colab_dir.mkdir(parents=True, exist_ok=True)
        return colab_dir
    
    def get_src_dir(self) -> Path:
        """Get source directory"""
        return self.project_root / "src"
    
    def get_db_path(self) -> Path:
        """Get database path"""
        return self.get_data_dir() / "jetx_data.db"
    
    def get_model_path(self, model_name: str, model_type: str = "deep_learning") -> Path:
        """Get model file path"""
        model_dir = self.get_models_dir() / model_type
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / f"{model_name}.pth"
    
    def get_metadata_path(self, model_name: str, model_type: str = "deep_learning") -> Path:
        """Get model metadata path"""
        model_dir = self.get_models_dir() / model_type
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / f"{model_name}_metadata.json"
    
    def get_ensemble_path(self) -> Path:
        """Get ensemble directory path"""
        ensemble_dir = self.get_models_dir() / "ensemble"
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        return ensemble_dir
    
    def get_backup_path(self) -> Path:
        """Get backup directory path"""
        backup_dir = self.get_models_dir() / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        return backup_dir
    
    def normalize_path(self, path: Union[str, Path]) -> Path:
        """Normalize path for current platform"""
        if isinstance(path, str):
            path = Path(path)
        
        # Convert to absolute path
        if not path.is_absolute():
            path = self.project_root / path
        
        # Resolve path
        return path.resolve()
    
    def ensure_directory(self, path: Union[str, Path]) -> Path:
        """Ensure directory exists"""
        path = self.normalize_path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_relative_path(self, path: Union[str, Path]) -> Path:
        """Get relative path from project root"""
        path = self.normalize_path(path)
        try:
            return path.relative_to(self.project_root)
        except ValueError:
            # Path is not relative to project root
            return path
    
    def get_environment_info(self) -> dict:
        """Get environment information"""
        return {
            'platform': self.platform,
            'is_colab': self.is_colab,
            'project_root': str(self.project_root),
            'python_version': sys.version,
            'python_executable': sys.executable,
            'current_directory': str(Path.cwd()),
            'home_directory': str(Path.home())
        }
    
    def add_to_path(self, directory: Union[str, Path]) -> None:
        """Add directory to Python path"""
        directory = self.normalize_path(directory)
        str_dir = str(directory)
        
        if str_dir not in sys.path:
            sys.path.insert(0, str_dir)
    
    def setup_project_paths(self) -> None:
        """Setup project paths in sys.path"""
        # Add src directory to path
        self.add_to_path(self.get_src_dir())
        
        # Add project root to path
        self.add_to_path(self.project_root)
        
        # Add colab directory to path if exists
        colab_dir = self.get_colab_dir()
        if colab_dir.exists():
            self.add_to_path(colab_dir)

# Global instance
path_manager = PathManager()

# Convenience functions
def get_project_root() -> Path:
    """Get project root directory"""
    return path_manager.get_project_root()

def get_models_dir() -> Path:
    """Get models directory"""
    return path_manager.get_models_dir()

def get_data_dir() -> Path:
    """Get data directory"""
    return path_manager.get_data_dir()

def get_db_path() -> Path:
    """Get database path"""
    return path_manager.get_db_path()

def get_model_path(model_name: str, model_type: str = "deep_learning") -> Path:
    """Get model file path"""
    return path_manager.get_model_path(model_name, model_type)

def normalize_path(path: Union[str, Path]) -> Path:
    """Normalize path for current platform"""
    return path_manager.normalize_path(path)

def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists"""
    return path_manager.ensure_directory(path)

def setup_project_paths() -> None:
    """Setup project paths in sys.path"""
    path_manager.setup_project_paths()

def get_environment_info() -> dict:
    """Get environment information"""
    return path_manager.get_environment_info()

# Auto-setup on import
try:
    setup_project_paths()
except Exception as e:
    warnings.warn(f"Could not setup project paths: {e}")
