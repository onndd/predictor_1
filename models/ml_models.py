# ml_models.py DOSYASININ TAM VE GÜNCEL HALİ

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

from feature_engineering.statistical_features import extract_statistical_features
from feature_engineering.categorical_features import CategoricalFeatureEncoder
from feature_engineering.pattern_features import NgramFeatureEncoder

class RandomForestJetXPredictor:
    def __init__(self, n_estimators=150, random_state=42, threshold=1.5):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.threshold = threshold
        self.min_history_for_features = 201
        
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1,
            min_samples_leaf=5
        )
        # Her RF modeli kendi özellik encoder'larına sahip olacak
        self.categorical_encoder = CategoricalFeatureEncoder()
        self.ngram_encoder = NgramFeatureEncoder(top_n_ngrams=20)
        
        self.is_fitted = False
        print(f"Tümleşik Özellikli RandomForestJetXPredictor başlatıldı: min_history={self.min_history_for_features}, n_estimators={self.n_estimators}")

    def _combine_all_features(self, values):
        """
        Tüm özellik setlerini (istatistiksel, kategorik, n-gram) birleştirir.
        """
        statistical_f = extract_statistical_features(values)
        categorical_f = self.categorical_encoder.transform(values)
        ngram_f = self.ngram_encoder.transform(values)
        
        min_len = min(len(statistical_f), len(categorical_f), len(ngram_f))
        
        combined_features = np.hstack([
            statistical_f[:min_len],
            categorical_f[:min_len],
            ngram_f[:min_len]
        ])
        return combined_features

    def _create_features(self, values):
        """
        Verilen değerler dizisinden tüm özellikleri çıkarır ve model için 
        X (özellikler) ve y (etiketler) setlerini oluşturur.
        """
        if len(values) <= self.min_history_for_features:
            return np.array([]), np.array([])

        all_features = self._combine_all_features(values)
        
        X = all_features[self.min_history_for_features-1:-1]
        y_values = values[self.min_history_for_features:]
        y = (np.array(y_values) >= self.threshold).astype(int)
        
        if X.shape[0] != y.shape[0]:
            min_len = min(X.shape[0], y.shape[0])
            X = X[:min_len]
            y = y[:min_len]
        return X, y

    def fit(self, values):
        """
        Modeli ve özellik encoder'larını verilen JetX değerleriyle eğitir.
        """
        print(f"RandomForestJetXPredictor {len(values)} değer ile eğitime başlıyor...")
        
        # Önce encoder'ları tüm veriyle 'fit' et
        self.categorical_encoder.fit(values)
        self.ngram_encoder.fit(values)
        
        # Sonra özellikleri oluştur ve modeli eğit
        X_train, y_train = self._create_features(values)

        if X_train.shape[0] == 0:
            print("RandomForest için özellik çıkarılamadı. Eğitim atlanıyor.")
            self.is_fitted = False
            return
        try:
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            print("RandomForestJetXPredictor (Tümleşik Özelliklerle) başarıyla eğitildi.")
        except Exception as e:
            print(f"RandomForest eğitimi sırasında hata: {e}")
            self.is_fitted = False

    def predict_next_value(self, sequence):
        """
        Verilen son değerler dizisine göre bir sonraki adım için tahmin yapar.
        """
        if not self.is_fitted:
            return None, 0.5, 0.3
        if len(sequence) < self.min_history_for_features:
            return None, 0.5, 0.3
        
        features_for_prediction_full = self._combine_all_features(sequence)
        current_features = features_for_prediction_full[-1].reshape(1, -1)
        
        try:
            probabilities = self.model.predict_proba(current_features)[0]
            above_threshold_probability = probabilities[1]
            confidence = abs(above_threshold_probability - 0.5) * 2
            return None, above_threshold_probability, confidence
        except Exception as e:
            print(f"RandomForest tahmini sırasında hata: {e}")
            return None, 0.5, 0.3