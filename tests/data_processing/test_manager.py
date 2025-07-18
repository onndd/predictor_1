import pytest
import os
import sqlite3
import torch
import sys

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_processing.manager import DataManager
from src.config.settings import PATHS

@pytest.fixture(scope="module")
def temp_db():
    """Create a temporary database for testing."""
    db_path = os.path.join(PATHS.get('cache_dir', 'data/cache'), "test_db.sqlite")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE jetx_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        value REAL NOT NULL
    )
    """)
    sample_data = [(i * 0.1 + 1.0,) for i in range(500)]
    cursor.executemany("INSERT INTO jetx_results (value) VALUES (?)", sample_data)
    conn.commit()
    conn.close()
    
    yield db_path
    
    os.remove(db_path)
    cache_file = os.path.join(PATHS.get('cache_dir', 'data/cache'), "test_db_full_data.pkl")
    if os.path.exists(cache_file):
        os.remove(cache_file)

def test_data_manager_initialization(temp_db):
    """Test if DataManager initializes correctly."""
    manager = DataManager(db_path=temp_db)
    assert manager.db_path == temp_db
    assert "test_db_full_data.pkl" in manager.cache_path

def test_get_all_data_from_db(temp_db):
    """Test loading all data directly from the database."""
    manager = DataManager(db_path=temp_db, use_cache=False)
    data = manager.get_all_data()
    assert isinstance(data, list)
    assert len(data) == 500
    assert isinstance(data[0], float)

def test_caching_logic(temp_db):
    """Test if caching works as expected."""
    manager = DataManager(db_path=temp_db, use_cache=True)
    
    # 1. First load (should be from DB and create cache)
    data_from_db = manager.get_all_data()
    assert os.path.exists(manager.cache_path), "Cache file should be created."
    
    # 2. Second load (should be from cache)
    # To ensure it's from cache, we can check the console output (manual) or trust the logic.
    # For automated test, we can check modification times.
    db_mod_time = os.path.getmtime(manager.db_path)
    cache_mod_time = os.path.getmtime(manager.cache_path)
    assert cache_mod_time > db_mod_time
    
    data_from_cache = manager.get_all_data()
    assert data_from_db == data_from_cache, "Data from cache should match data from DB."

def test_prepare_sequences(temp_db):
    """Test the sequence preparation method."""
    manager = DataManager(db_path=temp_db)
    data = manager.get_all_data()
    
    seq_len = 50
    X, y = manager.prepare_sequences(data, sequence_length=seq_len)
    
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] == len(data) - seq_len
    assert X.shape[1] == seq_len
    assert X.shape[2] == 1 # Feature dimension

def test_get_rolling_chunks(temp_db):
    """Test the creation of rolling chunks."""
    manager = DataManager(db_path=temp_db)
    data = manager.get_all_data()
    
    chunk_size = 100
    chunks = manager.get_rolling_chunks(data, chunk_size=chunk_size)
    
    assert isinstance(chunks, list)
    assert len(chunks) == len(data) // chunk_size
    assert len(chunks[0]) == chunk_size
