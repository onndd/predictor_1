import numpy as np
from typing import Dict, List, Optional, Any

# Import individual feature extractors
from .statistical_features import extract_statistical_features
from .categorical_features import CategoricalFeatureEncoder
from .pattern_features import NgramFeatureEncoder

class UnifiedFeatureExtractor:
    """
    Acts as a facade to orchestrate various feature extraction modules.
    This class combines features from statistical, categorical, and pattern-based extractors.
    """
    
    def __init__(self, 
                 sequence_length: int = 200,
                 threshold: float = 1.5,
                 top_n_ngrams: int = 20):
        """
        Initialize the unified feature extractor.
        
        Args:
            sequence_length: Standard sequence length for all features.
            threshold: Decision threshold for binary features.
            top_n_ngrams: Number of top n-grams to consider for pattern features.
        """
        self.sequence_length = sequence_length
        self.threshold = threshold
        
        # Initialize individual feature encoders
        self.categorical_encoder = CategoricalFeatureEncoder()
        self.ngram_encoder = NgramFeatureEncoder(top_n_ngrams=top_n_ngrams)
        
        self.is_fitted = False
        self.feature_names: List[str] = []

    def fit(self, values: List[float]) -> 'UnifiedFeatureExtractor':
        """
        Fit all underlying feature extractors on the training data.
        
        Args:
            values: Training data values.
            
        Returns:
            self: Fitted extractor.
        """
        if len(values) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} values for fitting.")
        
        print("Fitting UnifiedFeatureExtractor...")
        
        # Fit categorical and pattern encoders
        self.categorical_encoder.fit(values)
        self.ngram_encoder.fit(values)
        
        self.is_fitted = True
        
        # Generate feature names after fitting
        self._generate_feature_names(values)
        
        print(f"✅ UnifiedFeatureExtractor fitted with {len(self.feature_names)} features.")
        return self

    def transform(self, values: List[float]) -> np.ndarray:
        """
        Transform values into a combined feature matrix.
        
        Args:
            values: Input values.
            
        Returns:
            Feature matrix of shape (n_samples, n_features).
        """
        if not self.is_fitted:
            raise RuntimeError("Feature extractor must be fitted before transform.")
        
        # 1. Statistical Features
        statistical_f = extract_statistical_features(values)
        
        # 2. Categorical Features
        categorical_f = self.categorical_encoder.transform(values)
        
        # 3. Pattern (N-gram) Features
        ngram_f = self.ngram_encoder.transform(values)
        
        # Ensure all feature sets have the same number of samples
        min_len = min(len(statistical_f), len(categorical_f), len(ngram_f))
        
        # Combine all features
        combined_features = np.hstack([
            statistical_f[:min_len],
            categorical_f[:min_len],
            ngram_f[:min_len]
        ])
        
        # Clean final feature matrix
        return self._clean_features(combined_features)

    def fit_transform(self, values: List[float]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(values).transform(values)

    def _generate_feature_names(self, sample_values: List[float]):
        """Generates a list of all feature names."""
        # This is a simplified way to get feature names.
        # A more robust implementation would have each encoder return its feature names.
        
        stat_features = extract_statistical_features(sample_values)
        cat_features = self.categorical_encoder.transform(sample_values)
        ngram_features = self.ngram_encoder.transform(sample_values)
        
        names = []
        names.extend([f"stat_{i}" for i in range(stat_features.shape[1])])
        names.extend([f"cat_{i}" for i in range(cat_features.shape[1])])
        names.extend([f"ngram_{i}" for i in range(ngram_features.shape[1])])
        
        self.feature_names = names

    def get_feature_names(self) -> List[str]:
        """Get the list of generated feature names."""
        if not self.is_fitted:
            raise RuntimeError("Must fit the extractor before getting feature names.")
        return self.feature_names

    def _clean_features(self, features: np.ndarray) -> np.ndarray:
        """Clean features by replacing NaN and inf values."""
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features
