import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def create_sequences(data, seq_length):
    """
    Create sequential arrays from time series data
    
    Args:
        data: Numerical data array
        seq_length: Sequence length
        
    Returns:
        tuple: X (input sequences) and y (target values)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def time_series_split(data, n_splits=5, test_size=0.2):
    """
    Split time series data into training/test sets
    
    Args:
        data: Data array
        n_splits: Number of different splits to create
        test_size: Test set size ratio
        
    Returns:
        list: List of (train_indices, test_indices) pairs
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(data) * test_size))
    splits = []
    
    for train_index, test_index in tscv.split(data):
        splits.append((train_index, test_index))
        
    return splits

def create_above_threshold_target(data, threshold=1.5):
    """
    Create binary target values for above/below threshold
    
    Args:
        data: Data array
        threshold: Threshold value (default=1.5)
        
    Returns:
        numpy.ndarray: Binary target values (0=below, 1=above)
    """
    return (np.array(data) >= threshold).astype(int)

def create_category_target(data):
    """
    Create category target values
    
    Args:
        data: Data array
        
    Returns:
        list: List of category codes
    """
    from .transformer import transform_to_categories
    return transform_to_categories(data)

def split_recent(data, recent_size=1000):
    """
    Split data into recent N records and older records
    
    Args:
        data: Data array
        recent_size: Number of recent records to consider as "new"
        
    Returns:
        tuple: (older_data, recent_data)
    """
    if len(data) <= recent_size:
        return np.array([]), np.array(data)
    
    older_data = np.array(data[:-recent_size])
    recent_data = np.array(data[-recent_size:])
    
    return older_data, recent_data
