# categorical_features.py DOSYASININ TAM VE GÜNCEL HALİ

import numpy as np
import pandas as pd
from src.data_processing.transformer import (
    get_value_category, get_step_category,
    transform_to_categories, transform_to_step_categories,
    fuzzy_membership, VALUE_CATEGORIES
)

class CategoricalFeatureEncoder:
    def __init__(self):
        self.one_hot_categories = None
        self.count_categories = None

    def fit(self, values):
        """
        Tüm veri setindeki benzersiz kategorileri öğrenir.
        """
        print("CategoricalFeatureEncoder 'fit' ediliyor: Benzersiz kategoriler öğreniliyor...")
        all_categories = transform_to_categories(values)
        self.one_hot_categories = sorted(list(VALUE_CATEGORIES.keys()))
        self.count_categories = sorted(list(set(all_categories)))
        print(f"  -> One-hot için {len(self.one_hot_categories)}, sayım için {len(self.count_categories)} kategori bulundu.")

    def transform(self, values, window_sizes=[5, 10, 20, 50, 100]):
        """
        Öğrenilmiş kategorilere göre özellikleri dönüştürür.
        """
        if self.one_hot_categories is None or self.count_categories is None:
            raise RuntimeError("Encoder önce 'fit' edilmelidir.")

        # Kategorilere dönüştür
        categories = transform_to_categories(values)
        
        # Özellikler
        one_hot = self._encode_one_hot(categories)
        fuzzy = self._encode_fuzzy_membership(values)
        cat_counts = self._encode_category_counts(categories, window_sizes)
        threshold_stats = self._encode_threshold_statistics(values, 1.5, window_sizes)
        
        # Özellikleri birleştir
        return np.hstack([one_hot, fuzzy, cat_counts, threshold_stats])

    def _encode_one_hot(self, categories):
        n_samples = len(categories)
        n_features = len(self.one_hot_categories)
        encoded = np.zeros((n_samples, n_features))
        
        category_to_index = {cat: i for i, cat in enumerate(self.one_hot_categories)}
        
        for i, category in enumerate(categories):
            j = category_to_index.get(category)
            if j is not None:
                encoded[i, j] = 1
        return encoded

    def _encode_fuzzy_membership(self, values):
        categories_to_check = list(VALUE_CATEGORIES.keys())
        n_samples = len(values)
        n_features = len(categories_to_check)
        fuzzy_encoded = np.zeros((n_samples, n_features))
        
        for i, value in enumerate(values):
            for j, category in enumerate(categories_to_check):
                fuzzy_encoded[i, j] = fuzzy_membership(value, category)
        return fuzzy_encoded

    def _encode_category_counts(self, categories, window_sizes):
        n_samples = len(categories)
        n_features = len(window_sizes) * len(self.count_categories)
        counts = np.zeros((n_samples, n_features))
        
        category_to_index = {cat: i for i, cat in enumerate(self.count_categories)}

        for i in range(n_samples):
            feature_idx = 0
            for window in window_sizes:
                start = max(0, i - window)
                window_cats = categories[start:i]
                
                if len(window_cats) > 0:
                    # Pencere içindeki kategorileri say
                    window_counts = pd.Series(window_cats).value_counts(normalize=True)
                    for cat, freq in window_counts.items():
                        j = category_to_index.get(cat)
                        if j is not None:
                            counts[i, feature_idx + j] = freq
                
                feature_idx += len(self.count_categories)
        return counts

    def _encode_threshold_statistics(self, values, threshold, window_sizes):
        n_samples = len(values)
        n_features = len(window_sizes) * 4
        stats = np.zeros((n_samples, n_features))
        
        values_arr = np.array(values)

        for i in range(n_samples):
            feature_idx = 0
            for window in window_sizes:
                start = max(0, i - window)
                window_vals = values_arr[start:i]
                
                if len(window_vals) > 0:
                    above = window_vals[window_vals >= threshold]
                    below = window_vals[window_vals < threshold]
                    
                    stats[i, feature_idx] = len(above) / len(window_vals)
                    stats[i, feature_idx + 1] = len(below) / len(window_vals)
                    stats[i, feature_idx + 2] = np.mean(above) if len(above) > 0 else 0
                    stats[i, feature_idx + 3] = np.mean(below) if len(below) > 0 else 0
                
                feature_idx += 4
        return stats