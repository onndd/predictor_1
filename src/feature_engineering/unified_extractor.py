import numpy as np
from typing import Dict, List, Optional, Any

# Import individual feature extractors
from .statistical_features import extract_statistical_features
from .categorical_features import CategoricalFeatureEncoder
from .pattern_features import NgramFeatureEncoder
from .similarity_features import SimilarityFeatureEncoder

class UnifiedFeatureExtractor:
    """
    Acts as a facade to orchestrate various feature extraction modules.
    This class combines features from statistical, categorical, and pattern-based extractors.
    """
    
    def __init__(self,
                 feature_windows: List[int],
                 lag_windows: List[int],
                 lags: List[int],
                 model_sequence_length: int,
                 threshold: float = 1.5,
                 top_n_ngrams: int = 20):
        """
        Initialize the unified feature extractor.
        
        Args:
            feature_windows: List of window sizes for statistical features.
            lag_windows: List of window sizes for rolling features.
            lags: List of lag values for lag features.
            model_sequence_length: The sequence length the model will receive.
            threshold: Decision threshold for binary features.
            top_n_ngrams: Number of top n-grams to consider for pattern features.
        """
        self.feature_windows = feature_windows
        self.lag_windows = lag_windows
        self.lags = lags
        self.model_sequence_length = model_sequence_length
        self.threshold = threshold
        
        # Initialize individual feature encoders
        self.categorical_encoder = CategoricalFeatureEncoder()
        self.ngram_encoder = NgramFeatureEncoder(top_n_ngrams=top_n_ngrams)
        self.similarity_encoder = SimilarityFeatureEncoder()
        
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
        if len(values) < self.model_sequence_length:
            raise ValueError(f"Need at least {self.model_sequence_length} values for fitting.")
        
        print("Fitting UnifiedFeatureExtractor...")
        
        # Fit categorical and pattern encoders
        self.categorical_encoder.fit(values)
        self.ngram_encoder.fit(values)
        self.similarity_encoder.fit(values, self.model_sequence_length)
        
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
        statistical_f = extract_statistical_features(
            values,
            feature_windows=self.feature_windows,
            lag_windows=self.lag_windows,
            lags=self.lags
        )
        
        # 2. Categorical Features
        categorical_f = self.categorical_encoder.transform(values)
        
        # 3. Pattern (N-gram) Features
        ngram_f = self.ngram_encoder.transform(values)
        
        # 4. Similarity Features
        similarity_f = self.similarity_encoder.transform(values, self.model_sequence_length)

        # Ensure all feature sets have the same number of samples
        min_len = min(len(statistical_f), len(categorical_f), len(ngram_f), len(similarity_f))
        
        # Combine all features
        combined_features = np.hstack([
            statistical_f[:min_len],
            categorical_f[:min_len],
            ngram_f[:min_len],
            similarity_f[:min_len]
        ])
        
        # Clean final feature matrix
        return self._clean_features(combined_features)

    def fit_transform(self, values: List[float]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(values).transform(values)

    def _generate_feature_names(self, sample_values: List[float]):
        """Generates a list of all feature names based on the fitted encoders."""
        if not self.is_fitted:
            raise RuntimeError("Must fit the extractor before generating feature names.")

        names = []
        
        # 1. Statistical Features
        # This is tricky as the functions don't expose names. We'll create generic ones.
        # A better refactor would be to have each stat function return names.
        basic_stat_names = ['median', 'min', 'max', 'skew', 'kurt']
        names.extend([f"stat_{name}_{w}" for w in self.feature_windows for name in basic_stat_names])
        
        threshold_run_names = ['current_above_run', 'current_below_run', 'max_above_run', 'max_below_run']
        names.extend([f"stat_{name}" for name in threshold_run_names])

        trend_names = ['slope', 'autocorr', 'strength']
        names.extend([f"stat_trend_{name}_{w}" for w in self.feature_windows for name in trend_names])

        adv_stat_names = ['volatility', 'momentum', 'q25', 'q75']
        names.extend([f"stat_adv_{name}_{w}" for w in self.feature_windows for name in adv_stat_names])

        names.extend([f"stat_lag_{l}" for l in self.lags])
        names.extend([f"stat_roll_mean_{w}" for w in self.lag_windows])
        names.extend([f"stat_roll_std_{w}" for w in self.lag_windows])

        # 2. Categorical Features
        if self.categorical_encoder.one_hot_categories:
            names.extend([f"cat_onehot_{c}" for c in self.categorical_encoder.one_hot_categories])
        if self.categorical_encoder.one_hot_categories: # Fuzzy has same categories
            names.extend([f"cat_fuzzy_{c}" for c in self.categorical_encoder.one_hot_categories])
        
        cat_count_windows = [5, 10, 20, 50, 100]
        if self.categorical_encoder.count_categories:
            names.extend([f"cat_count_{c}_w{w}" for w in cat_count_windows for c in self.categorical_encoder.count_categories])

        threshold_stat_names = ['above_ratio', 'below_ratio', 'mean_above', 'mean_below']
        names.extend([f"cat_thresh_{name}_w{w}" for w in cat_count_windows for name in threshold_stat_names])

        # 3. N-gram Features
        ngram_windows = [10, 50, 100]
        if self.ngram_encoder.most_common_ngrams:
            ngram_names = [f"ngram_{'_'.join(gram)}" for gram in self.ngram_encoder.most_common_ngrams]
            names.extend([f"{name}_w{w}" for w in ngram_windows for name in ngram_names])

        # 4. Similarity Features
        sim_names = ['sim_avg_next', 'sim_std_next', 'sim_avg_score', 'sim_neighbors', 'sim_prob_above']
        names.extend(sim_names)
        
        self.feature_names = names

    def get_feature_names(self) -> List[str]:
        """Get the list of generated feature names."""
        if not self.is_fitted or not self.feature_names:
            raise RuntimeError("Must fit the extractor and generate feature names before getting them.")
        return self.feature_names

    def _clean_features(self, features: np.ndarray) -> np.ndarray:
        """Clean features by replacing NaN and inf values."""
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features
