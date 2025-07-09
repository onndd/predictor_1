# crash_detector.py DOSYASININ TAM İÇERİĞİ

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.exceptions import NotFittedError

class CrashDetector:
    def __init__(self, crash_threshold=1.5, lookback_period=5,
                 very_low_value_threshold=1.05, low_avg_threshold=1.20,
                 sustained_high_streak_threshold=1.80, streak_length_for_sustained_high=4):
        self.crash_threshold = crash_threshold
        self.lookback_period = lookback_period
        self.very_low_value_threshold = very_low_value_threshold
        self.low_avg_threshold = low_avg_threshold
        self.sustained_high_streak_threshold = sustained_high_streak_threshold
        self.streak_length_for_sustained_high = streak_length_for_sustained_high
        self.feature_names = ['avg_last_n', 'count_very_low', 'all_above_sustained_high', 'is_decreasing_sharply']
        self.X_train = []
        self.y_train = []
        self.ml_model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
        self.is_ml_model_fitted = False
        print(f"CrashDetector başlatıldı: Geriye bakış={self.lookback_period}, Özellikler: {self.feature_names}")

    def _extract_features(self, sequence):
        default_feature_values = {
            'avg_last_n': 1.5,
            'count_very_low': 0,
            'all_above_sustained_high': 0,
            'is_decreasing_sharply': 0
        }
        if len(sequence) < self.lookback_period:
            features_array = np.array([default_feature_values[name] for name in self.feature_names])
            return features_array
        relevant_sequence = np.array(sequence[-self.lookback_period:])
        avg_last_n = np.mean(relevant_sequence)
        count_very_low = np.sum(relevant_sequence < self.very_low_value_threshold)
        all_above_sustained_high = 1 if np.all(relevant_sequence > self.sustained_high_streak_threshold) else 0
        is_decreasing_sharply = 0
        if len(relevant_sequence) >= 3:
            if relevant_sequence[-1] < relevant_sequence[-2] < relevant_sequence[-3] and \
               (relevant_sequence[-3] - relevant_sequence[-1]) > 0.3:
                is_decreasing_sharply = 1
        calculated_features = {
            'avg_last_n': avg_last_n if not np.isnan(avg_last_n) else default_feature_values['avg_last_n'],
            'count_very_low': count_very_low,
            'all_above_sustained_high': all_above_sustained_high,
            'is_decreasing_sharply': is_decreasing_sharply
        }
        features_array = np.array([calculated_features[name] for name in self.feature_names])
        return features_array

    def fit(self, historical_values):
        print(f"CrashDetector, {len(historical_values)} adet geçmiş değer ile eğitime hazırlanıyor...")
        self.X_train = []
        self.y_train = []
        self.is_ml_model_fitted = False
        if len(historical_values) <= self.lookback_period:
            print("CrashDetector 'fit' metodu için yeterli geçmiş veri yok (en az lookback_period + 1). Model eğitilmeyecek.")
            return
        for i in range(len(historical_values) - self.lookback_period):
            current_sequence_for_features = historical_values[i : i + self.lookback_period]
            features = self._extract_features(current_sequence_for_features)
            actual_next_value = historical_values[i + self.lookback_period]
            label = 1 if actual_next_value < self.crash_threshold else 0
            self.X_train.append(features)
            self.y_train.append(label)
        if not self.X_train:
            print("CrashDetector için hiç özellik örneği oluşturulamadı. Model eğitilmeyecek.")
            return
        X_np = np.array(self.X_train)
        y_np = np.array(self.y_train)
        X_shuffled, y_shuffled = shuffle(X_np, y_np, random_state=42)
        min_samples_for_training = 20 
        if len(X_shuffled) < min_samples_for_training:
            print(f"CrashDetector eğitimi için yeterli örnek yok ({len(X_shuffled)}/{min_samples_for_training}). Model eğitilmeyecek.")
            return
        try:
            print(f"CrashDetector ML modeli {len(X_shuffled)} örnek ile eğitiliyor...")
            self.ml_model.fit(X_shuffled, y_shuffled)
            self.is_ml_model_fitted = True
            print("CrashDetector ML modeli başarıyla eğitildi.")
        except Exception as e:
            print(f"CrashDetector ML modeli eğitimi sırasında hata: {e}")
            self.is_ml_model_fitted = False

    def predict_crash_risk(self, current_sequence):
        if not self.is_ml_model_fitted:
            return 0.1
        if len(current_sequence) < self.lookback_period:
            return 0.1
        features_for_prediction = self._extract_features(current_sequence)
        if features_for_prediction.ndim == 1:
            features_for_prediction = features_for_prediction.reshape(1, -1)
        try:
            probabilities = self.ml_model.predict_proba(features_for_prediction)
            crash_probability = probabilities[0, 1]
            return crash_probability
        except NotFittedError:
            return 0.1
        except Exception as e:
            print(f"CrashDetector ML modeli tahmini sırasında hata: {e}. Varsayılan düşük risk (0.1).")
            return 0.1