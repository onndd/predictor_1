import numpy as np
import pandas as pd
from collections import Counter, deque
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class UnifiedFeatureExtractor:
    """
    Unified Feature Extractor for JetX Prediction System
    
    This class standardizes feature extraction across all modules and ensures consistency.
    
    Features:
    - Standardized feature shapes and formats
    - Consistent windowing across all feature types
    - Efficient memory usage with optimized computations
    - Proper handling of edge cases and missing data
    - Cached computations for better performance
    """
    
    def __init__(self, 
                 sequence_length: int = 200,
                 window_sizes: List[int] = [5, 10, 20, 50, 100],
                 threshold: float = 1.5,
                 cache_size: int = 1000):
        """
        Initialize unified feature extractor
        
        Args:
            sequence_length: Standard sequence length for all features
            window_sizes: Standard window sizes for all windowed features
            threshold: Decision threshold for binary features
            cache_size: Maximum cache size for computed features
        """
        self.sequence_length = sequence_length
        self.window_sizes = sorted(window_sizes)
        self.threshold = threshold
        self.cache_size = cache_size
        
        # Feature configuration
        self.feature_config = {
            'statistical': True,
            'categorical': True,
            'pattern': True,
            'trend': True,
            'volatility': True
        }
        
        # Caching system
        self.feature_cache = {}
        self.cache_keys = deque(maxlen=cache_size)
        
        # Feature dimensions (will be calculated during fit)
        self.feature_dimensions = {}
        self.total_features = 0
        
        # Categorical mappings
        self.categorical_mappings = {}
        self.pattern_mappings = {}
        
        # Fitted status
        self.is_fitted = False
        
        print(f"UnifiedFeatureExtractor initialized with sequence_length={sequence_length}")

    def fit(self, values: List[float]) -> 'UnifiedFeatureExtractor':
        """
        Fit the feature extractor on training data
        
        Args:
            values: Training data values
            
        Returns:
            self: Fitted extractor
        """
        if len(values) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} values for fitting")
        
        print("Fitting UnifiedFeatureExtractor...")
        
        # Convert to numpy array
        values_array = np.array(values, dtype=np.float32)
        
        # Fit categorical mappings
        self._fit_categorical_mappings(values_array)
        
        # Fit pattern mappings
        self._fit_pattern_mappings(values_array)
        
        # Calculate feature dimensions
        self._calculate_feature_dimensions()
        
        self.is_fitted = True
        print(f"✅ UnifiedFeatureExtractor fitted with {self.total_features} features")
        
        return self

    def transform(self, values: List[float]) -> np.ndarray:
        """
        Transform values into feature matrix
        
        Args:
            values: Input values
            
        Returns:
            Feature matrix of shape (n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transform")
        
        # Convert to numpy array
        values_array = np.array(values, dtype=np.float32)
        n_samples = len(values_array)
        
        # Initialize feature matrix
        features = np.zeros((n_samples, self.total_features), dtype=np.float32)
        
        # Extract all feature types
        feature_idx = 0
        
        if self.feature_config['statistical']:
            stat_features = self._extract_statistical_features(values_array)
            features[:, feature_idx:feature_idx + stat_features.shape[1]] = stat_features
            feature_idx += stat_features.shape[1]
        
        if self.feature_config['categorical']:
            cat_features = self._extract_categorical_features(values_array)
            features[:, feature_idx:feature_idx + cat_features.shape[1]] = cat_features
            feature_idx += cat_features.shape[1]
        
        if self.feature_config['pattern']:
            pattern_features = self._extract_pattern_features(values_array)
            features[:, feature_idx:feature_idx + pattern_features.shape[1]] = pattern_features
            feature_idx += pattern_features.shape[1]
        
        if self.feature_config['trend']:
            trend_features = self._extract_trend_features(values_array)
            features[:, feature_idx:feature_idx + trend_features.shape[1]] = trend_features
            feature_idx += trend_features.shape[1]
        
        if self.feature_config['volatility']:
            vol_features = self._extract_volatility_features(values_array)
            features[:, feature_idx:feature_idx + vol_features.shape[1]] = vol_features
            feature_idx += vol_features.shape[1]
        
        # Handle NaN and inf values
        features = self._clean_features(features)
        
        return features

    def fit_transform(self, values: List[float]) -> np.ndarray:
        """
        Fit and transform in one step
        
        Args:
            values: Input values
            
        Returns:
            Feature matrix
        """
        return self.fit(values).transform(values)

    def _fit_categorical_mappings(self, values: np.ndarray):
        """Fit categorical mappings"""
        # Define category ranges
        self.categorical_mappings = {
            'very_low': (0.0, 1.2),
            'low': (1.2, 1.5),
            'medium': (1.5, 2.0),
            'high': (2.0, 5.0),
            'very_high': (5.0, float('inf'))
        }
        
        # Get category names
        self.category_names = list(self.categorical_mappings.keys())

    def _fit_pattern_mappings(self, values: np.ndarray):
        """Fit pattern mappings"""
        # Convert values to categories
        categories = self._values_to_categories(values)
        
        # Extract n-grams
        ngrams = []
        for i in range(len(categories) - 1):
            bigram = (categories[i], categories[i + 1])
            ngrams.append(bigram)
        
        # Get top n-grams
        ngram_counts = Counter(ngrams)
        top_ngrams = [item[0] for item in ngram_counts.most_common(20)]
        
        self.pattern_mappings = {ngram: i for i, ngram in enumerate(top_ngrams)}

    def _calculate_feature_dimensions(self):
        """Calculate feature dimensions"""
        self.feature_dimensions = {}
        
        # Statistical features: mean, std, min, max, skew, kurtosis for each window
        self.feature_dimensions['statistical'] = len(self.window_sizes) * 6
        
        # Categorical features: one-hot + counts for each window
        self.feature_dimensions['categorical'] = len(self.category_names) + len(self.window_sizes) * len(self.category_names)
        
        # Pattern features: n-gram frequencies for each window
        self.feature_dimensions['pattern'] = len(self.window_sizes) * len(self.pattern_mappings)
        
        # Trend features: slope, correlation, momentum for each window
        self.feature_dimensions['trend'] = len(self.window_sizes) * 3
        
        # Volatility features: volatility, range, percentiles for each window
        self.feature_dimensions['volatility'] = len(self.window_sizes) * 4
        
        # Calculate total
        self.total_features = sum(self.feature_dimensions.values())

    def _extract_statistical_features(self, values: np.ndarray) -> np.ndarray:
        """Extract statistical features"""
        n_samples = len(values)
        n_features = self.feature_dimensions['statistical']
        features = np.zeros((n_samples, n_features), dtype=np.float32)
        
        for i in range(n_samples):
            feature_idx = 0
            
            for window_size in self.window_sizes:
                start_idx = max(0, i - window_size)
                window_values = values[start_idx:i+1]
                
                if len(window_values) > 0:
                    # Basic statistics
                    features[i, feature_idx] = np.mean(window_values)
                    features[i, feature_idx + 1] = np.std(window_values)
                    features[i, feature_idx + 2] = np.min(window_values)
                    features[i, feature_idx + 3] = np.max(window_values)
                    
                    # Higher order moments
                    if len(window_values) >= 3:
                        features[i, feature_idx + 4] = self._safe_skew(window_values)
                        features[i, feature_idx + 5] = self._safe_kurtosis(window_values)
                
                feature_idx += 6
        
        return features

    def _extract_categorical_features(self, values: np.ndarray) -> np.ndarray:
        """Extract categorical features"""
        n_samples = len(values)
        n_features = self.feature_dimensions['categorical']
        features = np.zeros((n_samples, n_features), dtype=np.float32)
        
        # Convert to categories
        categories = self._values_to_categories(values)
        
        for i in range(n_samples):
            feature_idx = 0
            
            # One-hot encoding for current value
            current_category = categories[i]
            if current_category in self.category_names:
                cat_idx = self.category_names.index(current_category)
                features[i, feature_idx + cat_idx] = 1.0
            
            feature_idx += len(self.category_names)
            
            # Category counts for each window
            for window_size in self.window_sizes:
                start_idx = max(0, i - window_size)
                window_categories = categories[start_idx:i+1]
                
                if len(window_categories) > 0:
                    # Count categories in window
                    category_counts = Counter(window_categories)
                    total_count = len(window_categories)
                    
                    for cat_name in self.category_names:
                        count = category_counts.get(cat_name, 0)
                        features[i, feature_idx] = count / total_count
                        feature_idx += 1
                else:
                    feature_idx += len(self.category_names)
        
        return features

    def _extract_pattern_features(self, values: np.ndarray) -> np.ndarray:
        """Extract pattern features"""
        n_samples = len(values)
        n_features = self.feature_dimensions['pattern']
        features = np.zeros((n_samples, n_features), dtype=np.float32)
        
        if not self.pattern_mappings:
            return features
        
        # Convert to categories
        categories = self._values_to_categories(values)
        
        for i in range(1, n_samples):  # Start from 1 to have at least one bigram
            feature_idx = 0
            
            for window_size in self.window_sizes:
                start_idx = max(0, i - window_size)
                window_categories = categories[start_idx:i+1]
                
                if len(window_categories) >= 2:
                    # Extract n-grams from window
                    window_ngrams = []
                    for j in range(len(window_categories) - 1):
                        bigram = (window_categories[j], window_categories[j + 1])
                        window_ngrams.append(bigram)
                    
                    # Count n-grams
                    ngram_counts = Counter(window_ngrams)
                    total_ngrams = len(window_ngrams)
                    
                    # Fill features
                    for ngram, pattern_idx in self.pattern_mappings.items():
                        count = ngram_counts.get(ngram, 0)
                        features[i, feature_idx + pattern_idx] = count / total_ngrams
                
                feature_idx += len(self.pattern_mappings)
        
        return features

    def _extract_trend_features(self, values: np.ndarray) -> np.ndarray:
        """Extract trend features"""
        n_samples = len(values)
        n_features = self.feature_dimensions['trend']
        features = np.zeros((n_samples, n_features), dtype=np.float32)
        
        for i in range(n_samples):
            feature_idx = 0
            
            for window_size in self.window_sizes:
                start_idx = max(0, i - window_size)
                window_values = values[start_idx:i+1]
                
                if len(window_values) >= 3:
                    # Linear trend (slope)
                    x = np.arange(len(window_values))
                    slope = self._safe_slope(x, window_values)
                    
                    # Autocorrelation
                    autocorr = self._safe_autocorr(window_values)
                    
                    # Momentum
                    momentum = (window_values[-1] - window_values[0]) / (window_values[0] + 1e-8)
                    
                    features[i, feature_idx] = slope
                    features[i, feature_idx + 1] = autocorr
                    features[i, feature_idx + 2] = momentum
                
                feature_idx += 3
        
        return features

    def _extract_volatility_features(self, values: np.ndarray) -> np.ndarray:
        """Extract volatility features"""
        n_samples = len(values)
        n_features = self.feature_dimensions['volatility']
        features = np.zeros((n_samples, n_features), dtype=np.float32)
        
        for i in range(n_samples):
            feature_idx = 0
            
            for window_size in self.window_sizes:
                start_idx = max(0, i - window_size)
                window_values = values[start_idx:i+1]
                
                if len(window_values) >= 2:
                    # Volatility (standard deviation)
                    volatility = np.std(window_values)
                    
                    # Range
                    value_range = np.max(window_values) - np.min(window_values)
                    
                    # Percentiles
                    q25 = np.percentile(window_values, 25)
                    q75 = np.percentile(window_values, 75)
                    
                    features[i, feature_idx] = volatility
                    features[i, feature_idx + 1] = value_range
                    features[i, feature_idx + 2] = q25
                    features[i, feature_idx + 3] = q75
                
                feature_idx += 4
        
        return features

    def _values_to_categories(self, values: np.ndarray) -> List[str]:
        """Convert values to categories"""
        categories = []
        for value in values:
            for cat_name, (min_val, max_val) in self.categorical_mappings.items():
                if min_val <= value < max_val:
                    categories.append(cat_name)
                    break
            else:
                categories.append('very_high')  # Default for extreme values
        return categories

    def _safe_skew(self, values: np.ndarray) -> float:
        """Safely calculate skewness"""
        try:
            from scipy.stats import skew
            return skew(values) if len(values) > 2 else 0.0
        except:
            return 0.0

    def _safe_kurtosis(self, values: np.ndarray) -> float:
        """Safely calculate kurtosis"""
        try:
            from scipy.stats import kurtosis
            return kurtosis(values) if len(values) > 2 else 0.0
        except:
            return 0.0

    def _safe_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        """Safely calculate slope"""
        try:
            if len(x) != len(y) or len(x) < 2:
                return 0.0
            slope = np.polyfit(x, y, 1)[0]
            return slope if np.isfinite(slope) else 0.0
        except:
            return 0.0

    def _safe_autocorr(self, values: np.ndarray) -> float:
        """Safely calculate autocorrelation"""
        try:
            if len(values) < 2:
                return 0.0
            return np.corrcoef(values[:-1], values[1:])[0, 1]
        except:
            return 0.0

    def _clean_features(self, features: np.ndarray) -> np.ndarray:
        """Clean features by replacing NaN and inf values"""
        # Replace NaN with 0
        features[np.isnan(features)] = 0.0
        
        # Replace inf with large finite values
        features[np.isinf(features)] = 0.0
        
        return features

    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted first")
        
        feature_names = []
        
        # Statistical features
        if self.feature_config['statistical']:
            for window in self.window_sizes:
                for stat in ['mean', 'std', 'min', 'max', 'skew', 'kurtosis']:
                    feature_names.append(f'stat_{stat}_w{window}')
        
        # Categorical features
        if self.feature_config['categorical']:
            # One-hot features
            for cat in self.category_names:
                feature_names.append(f'cat_onehot_{cat}')
            
            # Count features
            for window in self.window_sizes:
                for cat in self.category_names:
                    feature_names.append(f'cat_count_{cat}_w{window}')
        
        # Pattern features
        if self.feature_config['pattern']:
            for window in self.window_sizes:
                for i, ngram in enumerate(self.pattern_mappings.keys()):
                    feature_names.append(f'pattern_{i}_w{window}')
        
        # Trend features
        if self.feature_config['trend']:
            for window in self.window_sizes:
                for trend in ['slope', 'autocorr', 'momentum']:
                    feature_names.append(f'trend_{trend}_w{window}')
        
        # Volatility features
        if self.feature_config['volatility']:
            for window in self.window_sizes:
                for vol in ['volatility', 'range', 'q25', 'q75']:
                    feature_names.append(f'vol_{vol}_w{window}')
        
        return feature_names

    def get_feature_importance_groups(self) -> Dict[str, List[int]]:
        """Get feature groups for importance analysis"""
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted first")
        
        groups = {}
        feature_idx = 0
        
        # Statistical features
        if self.feature_config['statistical']:
            size = self.feature_dimensions['statistical']
            groups['statistical'] = list(range(feature_idx, feature_idx + size))
            feature_idx += size
        
        # Categorical features
        if self.feature_config['categorical']:
            size = self.feature_dimensions['categorical']
            groups['categorical'] = list(range(feature_idx, feature_idx + size))
            feature_idx += size
        
        # Pattern features
        if self.feature_config['pattern']:
            size = self.feature_dimensions['pattern']
            groups['pattern'] = list(range(feature_idx, feature_idx + size))
            feature_idx += size
        
        # Trend features
        if self.feature_config['trend']:
            size = self.feature_dimensions['trend']
            groups['trend'] = list(range(feature_idx, feature_idx + size))
            feature_idx += size
        
        # Volatility features
        if self.feature_config['volatility']:
            size = self.feature_dimensions['volatility']
            groups['volatility'] = list(range(feature_idx, feature_idx + size))
            feature_idx += size
        
        return groups

    def configure_features(self, **kwargs):
        """Configure which feature types to extract"""
        for feature_type, enabled in kwargs.items():
            if feature_type in self.feature_config:
                self.feature_config[feature_type] = enabled
        
        # Recalculate dimensions if fitted
        if self.is_fitted:
            self._calculate_feature_dimensions()

    def get_info(self) -> Dict[str, Any]:
        """Get extractor information"""
        info = {
            'is_fitted': self.is_fitted,
            'sequence_length': self.sequence_length,
            'window_sizes': self.window_sizes,
            'threshold': self.threshold,
            'feature_config': self.feature_config,
            'total_features': self.total_features,
            'feature_dimensions': self.feature_dimensions,
            'categorical_mappings': self.categorical_mappings,
            'pattern_mappings_count': len(self.pattern_mappings)
        }
        return info

    def save_extractor(self, filepath: str):
        """Save extractor state"""
        import pickle
        
        state = {
            'sequence_length': self.sequence_length,
            'window_sizes': self.window_sizes,
            'threshold': self.threshold,
            'feature_config': self.feature_config,
            'categorical_mappings': self.categorical_mappings,
            'pattern_mappings': self.pattern_mappings,
            'category_names': self.category_names,
            'feature_dimensions': self.feature_dimensions,
            'total_features': self.total_features,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_extractor(self, filepath: str):
        """Load extractor state"""
        import pickle
        import os
        
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.sequence_length = state['sequence_length']
            self.window_sizes = state['window_sizes']
            self.threshold = state['threshold']
            self.feature_config = state['feature_config']
            self.categorical_mappings = state['categorical_mappings']
            self.pattern_mappings = state['pattern_mappings']
            self.category_names = state['category_names']
            self.feature_dimensions = state['feature_dimensions']
            self.total_features = state['total_features']
            self.is_fitted = state['is_fitted']
            
            return True
            
        except Exception as e:
            print(f"Error loading extractor state: {e}")
            return False
