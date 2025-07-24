"""
Data Management for the JetX Prediction System.

This module provides a centralized DataManager class to handle all data-related
operations, including loading from SQLite, caching, sequence preparation,
and splitting data for training and validation.
"""

import os
import pickle
import sqlite3
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any

from src.config.settings import PATHS, CONFIG
from src.feature_engineering.unified_extractor import UnifiedFeatureExtractor

class DataManager:
    """
    Manages all data loading, preprocessing, and preparation tasks.
    """
    def __init__(self, db_path: str = PATHS['database'], use_cache: bool = True):
        self.db_path = db_path
        self.use_cache = use_cache
        self.cache_path = self._get_cache_path()
        training_config = CONFIG.get('training', {})
        self.feature_extractor = UnifiedFeatureExtractor(
            feature_windows=training_config.get('feature_windows', [50, 75, 100, 200, 500]),
            lag_windows=training_config.get('lag_windows', [5, 10, 20, 50]),
            lags=training_config.get('lags', [1, 2, 3, 5, 10]),
            model_sequence_length=training_config.get('model_sequence_length', 300),
            threshold=training_config.get('threshold', 1.5)
        )
        print(f"DataManager initialized with DB: {self.db_path}")

    def _get_cache_path(self) -> str:
        """Generates a path for the cache file based on the db path."""
        cache_dir = PATHS.get('cache_dir', 'data/cache')
        os.makedirs(cache_dir, exist_ok=True)
        db_filename = os.path.basename(self.db_path)
        cache_filename = f"{os.path.splitext(db_filename)[0]}_full_data.pkl"
        return os.path.join(cache_dir, cache_filename)

    def _load_from_cache(self) -> Optional[List[float]]:
        """Loads data from the pickle cache file if it's valid."""
        if os.path.exists(self.cache_path):
            db_mod_time = os.path.getmtime(self.db_path)
            cache_mod_time = os.path.getmtime(self.cache_path)
            if cache_mod_time > db_mod_time:
                try:
                    with open(self.cache_path, 'rb') as f:
                        print(f"⚡️ Loading data from cache: {self.cache_path}")
                        return pickle.load(f)
                except (pickle.UnpicklingError, EOFError):
                    print("⚠️ Cache file is corrupted. Re-loading from DB.")
        return None

    def _save_to_cache(self, data: List[float]):
        """Saves data to a pickle cache file."""
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"💾 Data saved to cache: {self.cache_path}")
        except IOError as e:
            print(f"❌ Could not write to cache file: {e}")

    def get_all_data(self) -> List[float]:
        """
        Loads all data from the database, utilizing a cache for speed.
        """
        if self.use_cache:
            cached_data = self._load_from_cache()
            if cached_data:
                return cached_data

        print("💾 Loading data directly from database...")
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found at {self.db_path}")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM jetx_results ORDER BY id ASC")
            rows = cursor.fetchall()
            conn.close()
            
            data = [row[0] for row in rows]
            
            if self.use_cache:
                self._save_to_cache(data)
                
            return data
        except sqlite3.Error as e:
            print(f"❌ Database error: {e}")
            return []

    def prepare_sequences(self, data: List[float], sequence_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepares sequences with rich features for training.
        """
        if len(data) <= sequence_length:
            return torch.empty(0, sequence_length, 1), torch.empty(0)

        # Fit the feature extractor on the provided data chunk
        self.feature_extractor.fit(data)
        
        # Generate features for the entire chunk
        all_features = self.feature_extractor.transform(data)
        
        # Create sequences from the generated features
        sequences, targets = [], []
        for i in range(len(all_features) - sequence_length):
            # Sequence of features
            seq = all_features[i:i + sequence_length]
            # The target is the raw value from the original data
            target = data[i + sequence_length]
            sequences.append(seq)
            targets.append(target)
            
        if not sequences:
            # Determine the number of features to return an empty tensor with the correct shape
            num_features = all_features.shape[1] if all_features.shape[0] > 0 else 0
            return torch.empty(0, sequence_length, num_features), torch.empty(0)

        sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        
        return sequences_tensor, targets_tensor

    def get_rolling_chunks(self, data: List[float], chunk_size: int) -> List[List[float]]:
        """Splits data into rolling window chunks."""
        if len(data) < chunk_size:
            return []
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size) if len(data[i:i + chunk_size]) >= chunk_size]
