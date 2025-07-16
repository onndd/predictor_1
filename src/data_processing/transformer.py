"""
JetX Data Transformer - Advanced categorization system for JetX game values
Provides detailed value categorization, fuzzy membership, and n-gram analysis
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Detailed value categories for JetX game outcomes
VALUE_CATEGORIES = {
    # Crash Zone (1.00 - 1.09) - Very detailed
    'CRASH_100_101': (1.00, 1.01),  # Only 1.00x
    'CRASH_101_103': (1.01, 1.03),
    'CRASH_103_106': (1.03, 1.06),
    'CRASH_106_110': (1.06, 1.10),

    # Low Zone (1.10 - 1.49) - Detailed
    'LOW_110_115': (1.10, 1.15),
    'LOW_115_120': (1.15, 1.20),
    'LOW_120_125': (1.20, 1.25),
    'LOW_125_130': (1.25, 1.30),
    'LOW_130_135': (1.30, 1.35),
    'LOW_135_140': (1.35, 1.40),
    'LOW_140_145': (1.40, 1.45),
    'LOW_145_149': (1.45, 1.49),  # Closest to 1.5 threshold

    # Threshold Zone (1.50 - 1.99) - Medium detail
    'THRESH_150_160': (1.50, 1.60),  # Just above 1.5 threshold
    'THRESH_160_170': (1.60, 1.70),
    'THRESH_170_185': (1.70, 1.85),
    'THRESH_185_199': (1.85, 1.99),  # Close to 2x

    # Early Multipliers (2.00 - 4.99)
    'EARLY_2X': (2.00, 2.49),
    'EARLY_2_5X': (2.50, 2.99),
    'EARLY_3X': (3.00, 3.99),
    'EARLY_4X': (4.00, 4.99),

    # Mid Multipliers (5.00 - 19.99)
    'MID_5_7X': (5.00, 7.49),
    'MID_7_10X': (7.50, 9.99),
    'MID_10_15X': (10.00, 14.99),
    'MID_15_20X': (15.00, 19.99),

    # High Multipliers (20.00 - 99.99)
    'HIGH_20_30X': (20.00, 29.99),
    'HIGH_30_50X': (30.00, 49.99),
    'HIGH_50_70X': (50.00, 69.99),
    'HIGH_70_100X': (70.00, 99.99),

    # Very High Multipliers (100.00+)
    'XHIGH_100_200X': (100.00, 199.99),
    'XHIGH_200PLUS': (200.00, float('inf'))  # Highest category
}


STEP_CATEGORIES = {
    'T1': (1.00, 1.49),
    'T2': (1.50, 2.00),
    'T3': (2.00, 2.50),
    'T4': (2.50, 3.00),
    'T5': (3.00, 3.50),
    'T6': (3.50, 4.00),
    'T7': (4.00, 4.50),
    'T8': (4.50, 5.00),
    'T9': (5.00, float('inf'))
}

SEQUENCE_CATEGORIES = {
    'S1': (5, 10),
    'S2': (10, 20),
    'S3': (20, 50),
    'S4': (50, 100),
    'S5': (100, 200),
    'S6': (200, 500),
    'S7': (500, 1000),
    'S8': (1000, float('inf'))
}

def get_value_category(value: float) -> str:
    """
    Get detailed category for a JetX value.
    
    Args:
        value: JetX game result (multiplier)
        
    Returns:
        str: Category code
    """
    for category, (min_val, max_val) in VALUE_CATEGORIES.items():
        if min_val <= value < max_val:
            return category
    return 'XHIGH_200PLUS'  # Default to highest category if no match

def get_step_category(value: float) -> str:
    """
    Get step category for a value (T1-T9 in 0.5 increments).
    
    Args:
        value: JetX game result (multiplier)
        
    Returns:
        str: Category code
    """
    for category, (min_val, max_val) in STEP_CATEGORIES.items():
        if min_val <= value < max_val:
            return category
    return 'T9'  # Default to highest category if no match

def get_sequence_category(seq_length: int) -> str:
    """
    Get sequence length category (S1-S8).
    
    Args:
        seq_length: Sequence length
        
    Returns:
        str: Category code
    """
    for category, (min_val, max_val) in SEQUENCE_CATEGORIES.items():
        if min_val <= seq_length < max_val:
            return category
    return 'S8'  # Default to highest category if no match

def get_compound_category(value: float, seq_length: int) -> str:
    """
    Create compound category using VALUE_CATEGORIES, STEP_CATEGORIES, and SEQUENCE_CATEGORIES.
    
    Args:
        value: JetX game result (multiplier)
        seq_length: Sequence length
        
    Returns:
        str: Compound category code
    """
    val_cat = get_value_category(value)
    step_cat = get_step_category(value)
    seq_cat = get_sequence_category(seq_length)
    
    return f"{val_cat}__{step_cat}__{seq_cat}"

def transform_to_categories(values: List[float]) -> List[str]:
    """
    Transform values to detailed categories.
    
    Args:
        values: List of JetX values
        
    Returns:
        list: List of detailed category codes
    """
    return [get_value_category(val) for val in values]

def transform_to_step_categories(values: List[float]) -> List[str]:
    """
    Transform values to step categories (0.5 increment categories).
    
    Args:
        values: List of JetX values
        
    Returns:
        list: List of category codes (T1-T9)
    """
    return [get_step_category(val) for val in values]

def transform_to_compound_categories(values: List[float]) -> List[str]:
    """
    Transform values to compound categories (VALUE, STEP, SEQUENCE combined).
    
    Args:
        values: List of JetX values
        
    Returns:
        list: List of compound category codes
    """
    result = []
    for i, val in enumerate(values):
        seq_length = len(values) - i  # Remaining elements count
        result.append(get_compound_category(val, seq_length))
    return result

def fuzzy_membership(value: float, category_key: str) -> float:
    """
    Calculate fuzzy membership degree of a value to a specific VALUE_CATEGORIES category (0-1).
    
    Args:
        value: JetX game result (multiplier)
        category_key: Category key in VALUE_CATEGORIES (e.g., 'LOW_110_115')
        
    Returns:
        float: Membership degree (0-1)
    """
    if category_key not in VALUE_CATEGORIES:
        return 0.0

    min_val, max_val = VALUE_CATEGORIES[category_key]
    
    if max_val == float('inf'):
        return 1.0 if value >= min_val else 0.0

    range_size = max_val - min_val
    if range_size <= 0:
        range_size = 0.1

    if min_val <= value < max_val:
        mid_point = (min_val + max_val) / 2
        distance_to_mid = abs(value - mid_point)
        max_distance_to_mid = range_size / 2
        if max_distance_to_mid == 0:
            return 1.0
        return 1.0 - (distance_to_mid / max_distance_to_mid) * 0.5
    
    overlap_ratio = 0.1
    extended_min = min_val - range_size * overlap_ratio
    extended_max = max_val + range_size * overlap_ratio

    if extended_min <= value < min_val:
        distance_to_boundary = min_val - value
        return max(0, 0.5 - (distance_to_boundary / (range_size * overlap_ratio)) * 0.5)
    if max_val <= value < extended_max:
        distance_to_boundary = value - max_val
        return max(0, 0.5 - (distance_to_boundary / (range_size * overlap_ratio)) * 0.5)
            
    return 0.0

def get_value_step_crossed_category(value: float) -> str:
    """
    Cross detailed VALUE_CATEGORIES with STEP_CATEGORIES.
    
    Args:
        value: JetX game result (multiplier)
        
    Returns:
        str: Crossed category code (e.g., 'LOW_145_149__T1')
    """
    val_cat = get_value_category(value)
    step_cat = get_step_category(value)
    
    return f"{val_cat}__{step_cat}"

def transform_to_value_step_crossed_categories(values: List[float]) -> List[str]:
    """
    Transform value list to crossed (VALUE and STEP) categories.
    
    Args:
        values: List of JetX values
        
    Returns:
        list: List of crossed category codes
    """
    return [get_value_step_crossed_category(val) for val in values]

def transform_to_category_ngrams(categories: List[str], n: int = 2) -> List[Tuple[str, ...]]:
    """
    Create n-grams (sequential n-element groups) from a category list.
    
    Args:
        categories: List of category codes (e.g., ['LOW_110_115', 'HIGH_20_30X'])
        n: Number of elements in each group (2 = bigram, 3 = trigram)
        
    Returns:
        list: List of n-grams (as tuples)
              Example for n=2: [('LOW_110_115', 'HIGH_20_30X'), ...]
    """
    if len(categories) < n:
        return []
    
    # Use zip to create n-grams efficiently
    ngrams = zip(*[categories[i:] for i in range(n)])
    return list(ngrams)
