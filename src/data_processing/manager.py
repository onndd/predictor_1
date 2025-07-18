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

from src.config.settings import PATHS

class DataManager:
    """
    Manages all data loading, preprocessing, and preparation tasks.
    """
    def __init__(self, db_path: str = PATHS['database'], use_cache: bool = True):
        self.db_path = db_path
        self.use_cache = use_cache
        self.cache_path = self._get_cache_path()
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
        Prepares sequences for training.
        """
        if len(data) <= sequence_length:
            return torch.empty(0, sequence_length, 1), torch.empty(0)

        sequences, targets = [], []
        for i in range(len(data) - sequence_length):
            seq = data[i:i + sequence_length]
            target = data[i + sequence_length]
            sequences.append(seq)
            targets.append(target)
        
        sequences_tensor = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        
        return sequences_tensor, targets_tensor

    def get_rolling_chunks(self, data: List[float], chunk_size: int) -> List[List[float]]:
        """Splits data into rolling window chunks."""
        if len(data) < chunk_size:
            return []
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size) if len(data[i:i + chunk_size]) >= chunk_size]
