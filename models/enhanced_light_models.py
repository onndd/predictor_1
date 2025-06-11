import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

class EnhancedRandomForest:
    """
    Gelişmiş Random Forest - JetX için optimize edilmiş
    """
    
    def __init__(self, threshold=1.5, n_estimators=300, max_depth=20):
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _extract_features(self, sequence):
        """Sequence'den gelişmiş özellikler çıkar"""
        seq = np.array(sequence)
        
        # Temel istatistikler
        features = [
            np.mean(seq),                    # Ortalama
            np.std(seq),                     # Standart sapma
            np.max(seq),                     # Maksimum
            np.min(seq),                     # Minimum
            np.median(seq),                  # Medyan
            np.percentile(seq, 75),          # 75. percentile
            np.percentile(seq, 25),          # 25. percentile
        ]
        
        # Threshold-based features
        features.extend([
            np.mean(seq >= 1.5),             # 1.5 üstü oranı
            np.mean(seq >= 2.0),             # 2.0 üstü oranı
            np.mean(seq >= 3.0),             # 3.0 üstü oranı
            np.mean(seq <= 1.3),             # 1.3 altı oranı
        ])
        
        # Trend features
        if len(seq) > 1:
            diff = np.diff(seq)
            features.extend([
                np.mean(diff),               # Ortalama değişim
                np.std(diff),                # Değişim standart sapması
                np.mean(diff > 0),           # Artış oranı
                np.mean(diff < 0),           # Azalış oranı
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Rolling statistics (son N değerler)
        for window in [5, 10, 20]:
            if len(seq) >= window:
                rolling_seq = seq[-window:]
                features.extend([
                    np.mean(rolling_seq),
                    np.std(rolling_seq),
                    np.mean(rolling_seq >= self.threshold)
                ])
            else:
                features.extend([np.mean(seq), np.std(seq), np.mean(seq >= self.threshold)])
        
        # Pattern features
        # Consecutive patterns
        consecutive_above = 0
        consecutive_below = 0
        current_above = 0
        current_below = 0
        
        for val in seq[-20:]:  # Son 20 değer
            if val >= self.threshold:
                current_above += 1
                current_below = 0
                consecutive_above = max(consecutive_above, current_above)
            else:
                current_below += 1
                current_above = 0
                consecutive_below = max(consecutive_below, current_below)
        
        features.extend([consecutive_above, consecutive_below])
        
        # Volatility features
        if len(seq) > 2:
            returns = np.diff(seq) / seq[:-1]
            features.extend([
                np.std(returns),             # Volatilite
                np.mean(np.abs(returns)),    # Ortalama mutlak getiri
            ])
        else:
            features.extend([0, 0])
        
        return features
    
    def fit(self, sequences, labels):
        """Model eğitimi"""
        print(f"Enhanced Random Forest eğitimi başlıyor... {len(sequences)} sequence")
        
        # Feature extraction
        X = []
        for seq in sequences:
            features = self._extract_features(seq)
            X.append(features)
        
        X = np.array(X)
        y = np.array(labels)
        
        # Feature scaling
        X_scaled = self.scaler.fit_transform(X)
        
        # Model oluştur
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Eğitim
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        print("Enhanced Random Forest eğitimi tamamlandı!")
        return self
    
    def predict_next_value(self, sequence):
        """Tahmin yap"""
        if not self.is_fitted:
            return None, 0.5, 0.0
        
        try:
            features = self._extract_features(sequence)
            X = np.array(features).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Probability prediction
            prob = self.model.predict_proba(X_scaled)[0][1]
            
            # Confidence (feature importance based)
            confidence = min(1.0, max(0.0, abs(prob - 0.5) * 2))
            
            # Predicted value estimate
            predicted_value = np.mean(sequence[-10:]) * (1.2 if prob > 0.5 else 0.8)
            
            return predicted_value, prob, confidence
            
        except Exception as e:
            print(f"Enhanced RF prediction error: {e}")
            return None, 0.5, 0.0


class EnhancedGradientBoosting:
    """
    Gelişmiş Gradient Boosting - Güçlü pattern recognition
    """
    
    def __init__(self, threshold=1.5, n_estimators=200, learning_rate=0.1):
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _extract_advanced_features(self, sequence):
        """Gelişmiş özellik çıkarımı"""
        seq = np.array(sequence)
        features = []
        
        # Multi-window statistics
        for window in [3, 5, 10, 15, 20, 30]:
            if len(seq) >= window:
                window_seq = seq[-window:]
                features.extend([
                    np.mean(window_seq),
                    np.std(window_seq),
                    np.max(window_seq),
                    np.min(window_seq),
                    np.mean(window_seq >= self.threshold),
                    # Momentum
                    (window_seq[-1] - window_seq[0]) / window if window > 1 else 0
                ])
            else:
                features.extend([0] * 6)
        
        # Pattern counting
        patterns = {
            'high_crash': 0,     # >3.0 sonrası <1.5
            'low_stable': 0,     # <1.5 ardışık
            'high_stable': 0,    # >1.5 ardışık
            'oscillation': 0     # Up-down pattern
        }
        
        for i in range(1, min(len(seq), 20)):
            if seq[-i-1] > 3.0 and seq[-i] < 1.5:
                patterns['high_crash'] += 1
            
            if i > 1:
                if seq[-i-1] < 1.5 and seq[-i] < 1.5:
                    patterns['low_stable'] += 1
                elif seq[-i-1] > 1.5 and seq[-i] > 1.5:
                    patterns['high_stable'] += 1
                
                if i > 2:
                    # Oscillation detection
                    if ((seq[-i-2] > seq[-i-1] and seq[-i-1] < seq[-i]) or
                        (seq[-i-2] < seq[-i-1] and seq[-i-1] > seq[-i])):
                        patterns['oscillation'] += 1
        
        features.extend(list(patterns.values()))
        
        # Technical indicators
        if len(seq) >= 10:
            # Simple Moving Average
            sma_5 = np.mean(seq[-5:])
            sma_10 = np.mean(seq[-10:])
            features.extend([
                sma_5,
                sma_10,
                sma_5 - sma_10,  # SMA difference
                seq[-1] - sma_5,  # Distance from SMA
            ])
            
            # Bollinger Bands
            std_10 = np.std(seq[-10:])
            upper_band = sma_10 + 2 * std_10
            lower_band = sma_10 - 2 * std_10
            features.extend([
                (seq[-1] - lower_band) / (upper_band - lower_band) if upper_band > lower_band else 0.5
            ])
        else:
            features.extend([0] * 5)
        
        return features
    
    def fit(self, sequences, labels):
        """Model eğitimi"""
        print(f"Enhanced Gradient Boosting eğitimi başlıyor... {len(sequences)} sequence")
        
        # Feature extraction
        X = []
        for seq in sequences:
            features = self._extract_advanced_features(seq)
            X.append(features)
        
        X = np.array(X)
        y = np.array(labels)
        
        # Feature scaling
        X_scaled = self.scaler.fit_transform(X)
        
        # Model
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        )
        
        # Eğitim
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        print("Enhanced Gradient Boosting eğitimi tamamlandı!")
        return self
    
    def predict_next_value(self, sequence):
        """Tahmin yap"""
        if not self.is_fitted:
            return None, 0.5, 0.0
        
        try:
            features = self._extract_advanced_features(sequence)
            X = np.array(features).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            prob = self.model.predict_proba(X_scaled)[0][1]
            confidence = min(1.0, max(0.0, abs(prob - 0.5) * 2.5))  # Boosting confidence
            
            # Value prediction
            predicted_value = np.mean(sequence[-5:]) * (1.3 if prob > 0.6 else 0.7)
            
            return predicted_value, prob, confidence
            
        except Exception as e:
            print(f"Enhanced GB prediction error: {e}")
            return None, 0.5, 0.0


class LightModelEnsemble:
    """
    Hafif modellerin ensemble'ı
    """
    
    def __init__(self, threshold=1.5):
        self.threshold = threshold
        self.models = {}
        self.model_weights = {}
        self.is_fitted = False
        
    def add_model(self, name, model, weight=1.0):
        """Model ekle"""
        self.models[name] = model
        self.model_weights[name] = weight
        
    def fit(self, sequences, labels):
        """Tüm modelleri eğit"""
        print(f"Light Model Ensemble eğitimi başlıyor... {len(self.models)} model")
        
        # Her modeli eğit
        for name, model in self.models.items():
            print(f"  -> {name} eğitiliyor...")
            model.fit(sequences, labels)
        
        self.is_fitted = True
        print("Light Model Ensemble eğitimi tamamlandı!")
        
    def predict_next_value(self, sequence):
        """Ensemble tahmin"""
        if not self.is_fitted:
            return None, 0.5, 0.0
        
        predictions = []
        confidences = []
        values = []
        total_weight = 0
        
        for name, model in self.models.items():
            try:
                val, prob, conf = model.predict_next_value(sequence)
                if prob is not None:
                    weight = self.model_weights[name]
                    predictions.append(prob * weight)
                    confidences.append(conf * weight)
                    if val is not None:
                        values.append(val * weight)
                    total_weight += weight
            except:
                continue
        
        if total_weight == 0:
            return None, 0.5, 0.0
        
        # Weighted average
        avg_prob = sum(predictions) / total_weight
        avg_confidence = sum(confidences) / total_weight if confidences else 0.0
        avg_value = sum(values) / len(values) if values else np.mean(sequence[-5:])
        
        return avg_value, avg_prob, avg_confidence


def create_enhanced_light_models(threshold=1.5):
    """
    Gelişmiş hafif modelleri oluştur
    """
    models = {}
    
    # Enhanced Random Forest
    models['enhanced_rf'] = EnhancedRandomForest(
        threshold=threshold,
        n_estimators=300,
        max_depth=20
    )
    
    # Enhanced Gradient Boosting
    models['enhanced_gb'] = EnhancedGradientBoosting(
        threshold=threshold,
        n_estimators=200,
        learning_rate=0.1
    )
    
    # Extra Trees (Random Forest variant)
    class ExtraTreesWrapper:
        def __init__(self, threshold=1.5):
            self.threshold = threshold
            self.model = ExtraTreesClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            self.scaler = StandardScaler()
            self.is_fitted = False
            
        def fit(self, sequences, labels):
            rf_model = EnhancedRandomForest(self.threshold)
            X = [rf_model._extract_features(seq) for seq in sequences]
            X = self.scaler.fit_transform(X)
            self.model.fit(X, labels)
            self.is_fitted = True
            
        def predict_next_value(self, sequence):
            if not self.is_fitted:
                return None, 0.5, 0.0
            rf_model = EnhancedRandomForest(self.threshold)
            features = rf_model._extract_features(sequence)
            X = self.scaler.transform([features])
            prob = self.model.predict_proba(X)[0][1]
            return None, prob, abs(prob - 0.5) * 2
    
    models['extra_trees'] = ExtraTreesWrapper(threshold)
    
    # SVM (Support Vector Machine)
    class SVMWrapper:
        def __init__(self, threshold=1.5):
            self.threshold = threshold
            self.model = SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
            self.scaler = StandardScaler()
            self.is_fitted = False
            
        def fit(self, sequences, labels):
            rf_model = EnhancedRandomForest(self.threshold)
            X = [rf_model._extract_features(seq) for seq in sequences]
            X = self.scaler.fit_transform(X)
            self.model.fit(X, labels)
            self.is_fitted = True
            
        def predict_next_value(self, sequence):
            if not self.is_fitted:
                return None, 0.5, 0.0
            rf_model = EnhancedRandomForest(self.threshold)
            features = rf_model._extract_features(sequence)
            X = self.scaler.transform([features])
            prob = self.model.predict_proba(X)[0][1]
            return None, prob, abs(prob - 0.5) * 2
    
    models['svm'] = SVMWrapper(threshold)
    
    return models