# pattern_features.py DOSYASININ TAM VE GÜNCEL HALİ

import numpy as np
from collections import Counter
import pandas as pd
from src.data_processing.transformer import transform_to_categories, transform_to_category_ngrams

class NgramFeatureEncoder:
    def __init__(self, top_n_ngrams=20):
        self.top_n_ngrams = top_n_ngrams
        self.most_common_ngrams = None

    def fit(self, values):
        """
        Tüm veri setindeki en popüler n-gram'ları öğrenir.
        """
        print("NgramFeatureEncoder 'fit' ediliyor: En popüler n-gram'lar öğreniliyor...")
        categories = transform_to_categories(values)
        if len(categories) < 2:
            self.most_common_ngrams = []
            return
            
        all_ngrams = transform_to_category_ngrams(categories, n=2)
        if not all_ngrams:
            self.most_common_ngrams = []
            return
            
        ngram_counts = Counter(all_ngrams)
        self.most_common_ngrams = [item[0] for item in ngram_counts.most_common(self.top_n_ngrams)]
        print(f"  -> {len(self.most_common_ngrams)} adet en popüler n-gram bulundu.")

    def transform(self, values, window_sizes=[10, 50, 100]):
        """
        Öğrenilmiş n-gram'lara göre özellikleri dönüştürür.
        """
        if self.most_common_ngrams is None:
            raise RuntimeError("Encoder önce 'fit' edilmelidir.")

        categories = transform_to_categories(values)
        n_samples = len(categories)
        n_features = len(window_sizes) * len(self.most_common_ngrams)
        
        if not self.most_common_ngrams: # Eğer hiç popüler n-gram öğrenilemediyse
            return np.zeros((n_samples, n_features))

        features = np.zeros((n_samples, n_features))
        
        ngram_to_index = {ngram: i for i, ngram in enumerate(self.most_common_ngrams)}
        
        for i in range(1, n_samples):
            feature_col_idx = 0
            for window in window_sizes:
                start = max(0, i - window)
                window_cats = categories[start:i]
                
                if len(window_cats) >= 2:
                    window_ngrams = transform_to_category_ngrams(window_cats, n=2)
                    if window_ngrams:
                        window_ngram_counts = Counter(window_ngrams)
                        for ngram, count in window_ngram_counts.items():
                            j = ngram_to_index.get(ngram)
                            if j is not None:
                                features[i, feature_col_idx + j] = count / len(window_ngrams)
                
                feature_col_idx += len(self.most_common_ngrams)
                
        return features

# Diğer eski fonksiyonlarınız (extract_pattern_sequences vb.) burada kalabilir, sorun teşkil etmezler.