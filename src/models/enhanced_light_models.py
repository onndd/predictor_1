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
import pickle
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

warnings.filterwarnings('ignore')

class HeavyModelKnowledge:
    """
    Heavy modellerden çıkarılan bilgi sınıfı
    """
    
    def __init__(self):
        self.pattern_weights = {}
        self.feature_importance = {}
        self.decision_boundaries = {}
        self.performance_metrics = {}
        self.extracted_patterns = {}
        self.threshold_adjustments = {}
        self.confidence_calibration = {}
        self.creation_time = datetime.now()
        
    def add_pattern_weight(self, pattern_name: str, weight: float):
        """Pattern ağırlığı ekle"""
        self.pattern_weights[pattern_name] = weight
        
    def add_feature_importance(self, feature_name: str, importance: float):
        """Feature importance ekle"""
        self.feature_importance[feature_name] = importance
        
    def add_decision_boundary(self, feature_range: Tuple[float, float], decision: str):
        """Karar sınırı ekle"""
        self.decision_boundaries[feature_range] = decision
        
    def add_extracted_pattern(self, pattern_type: str, pattern_data: Dict):
        """Çıkarılan pattern ekle"""
        self.extracted_patterns[pattern_type] = pattern_data
        
    def add_threshold_adjustment(self, condition: str, adjustment: float):
        """Threshold ayarlama ekle"""
        self.threshold_adjustments[condition] = adjustment
        
    def get_pattern_weight(self, pattern_name: str) -> float:
        """Pattern ağırlığını al"""
        return self.pattern_weights.get(pattern_name, 1.0)
        
    def get_feature_importance(self, feature_name: str) -> float:
        """Feature importance al"""
        return self.feature_importance.get(feature_name, 1.0)
        
    def get_threshold_adjustment(self, sequence: List[float]) -> float:
        """Sequence için threshold ayarlama öner"""
        # Basit örnek - gerçek implementasyon daha karmaşık olacak
        recent_avg = np.mean(sequence[-10:])
        
        for condition, adjustment in self.threshold_adjustments.items():
            if condition == "high_volatility" and np.std(sequence[-10:]) > 0.5:
                return adjustment
            elif condition == "low_values" and recent_avg < 1.3:
                return adjustment
            elif condition == "high_values" and recent_avg > 2.0:
                return adjustment
                
        return 0.0
        
    def get_confidence_boost(self, prediction_prob: float) -> float:
        """Prediction confidence boost hesapla"""
        # Heavy model bilgisine dayalı confidence artışı
        if prediction_prob > 0.7:
            return 0.15  # Yüksek confidence'a daha fazla boost
        elif prediction_prob > 0.6:
            return 0.10
        elif prediction_prob > 0.4:
            return 0.05
        else:
            return 0.0
            
    def save_knowledge(self, filepath: str):
        """Bilgiyi dosyaya kaydet"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load_knowledge(cls, filepath: str) -> Optional['HeavyModelKnowledge']:
        """Dosyadan bilgi yükle"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading knowledge: {e}")
        return None
        
    def is_valid(self) -> bool:
        """Bilgi geçerli mi kontrol et"""
        return len(self.pattern_weights) > 0 or len(self.feature_importance) > 0
        
    def get_summary(self) -> Dict[str, Any]:
        """Bilgi özeti"""
        return {
            'pattern_weights_count': len(self.pattern_weights),
            'feature_importance_count': len(self.feature_importance),
            'decision_boundaries_count': len(self.decision_boundaries),
            'extracted_patterns_count': len(self.extracted_patterns),
            'threshold_adjustments_count': len(self.threshold_adjustments),
            'creation_time': self.creation_time,
            'age_hours': (datetime.now() - self.creation_time).total_seconds() / 3600
        }

class EnhancedRandomForest:
    """
    Gelişmiş Random Forest - JetX için optimize edilmiş
    Heavy model knowledge ile güçlendirilmiş
    """
    
    def __init__(self, threshold=1.5, n_estimators=300, max_depth=20):
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.heavy_knowledge = None
        self.knowledge_boost_enabled = False
        
    def update_with_heavy_knowledge(self, knowledge: HeavyModelKnowledge):
        """Heavy model bilgisini güncelle"""
        if knowledge and knowledge.is_valid():
            self.heavy_knowledge = knowledge
            self.knowledge_boost_enabled = True
            print("✅ Random Forest heavy model bilgisi ile güçlendirildi")
        else:
            print("⚠️ Geçersiz heavy model bilgisi")
        
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
        """Heavy model bilgisi ile güçlendirilmiş tahmin"""
        if not self.is_fitted:
            return None, 0.5, 0.0
        
        try:
            features = self._extract_features(sequence)
            X = np.array(features).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Base probability prediction
            prob = self.model.predict_proba(X_scaled)[0][1]
            
            # Heavy model knowledge ile güçlendirme
            if self.knowledge_boost_enabled and self.heavy_knowledge:
                # Threshold adjustment
                threshold_adjustment = self.heavy_knowledge.get_threshold_adjustment(sequence)
                adjusted_threshold = self.threshold + threshold_adjustment
                
                # Pattern-based probability adjustment
                pattern_boost = 0.0
                recent_avg = np.mean(sequence[-10:])
                recent_std = np.std(sequence[-10:])
                
                # High volatility pattern
                if recent_std > 0.5:
                    pattern_boost += self.heavy_knowledge.get_pattern_weight("high_volatility") * 0.05
                
                # Low values pattern
                if recent_avg < 1.3:
                    pattern_boost += self.heavy_knowledge.get_pattern_weight("low_values") * 0.08
                
                # High values pattern
                if recent_avg > 2.0:
                    pattern_boost += self.heavy_knowledge.get_pattern_weight("high_values") * 0.06
                
                # Consecutive pattern detection
                consecutive_count = 0
                for i in range(min(5, len(sequence))):
                    if sequence[-(i+1)] >= self.threshold:
                        consecutive_count += 1
                    else:
                        break
                
                if consecutive_count >= 3:
                    pattern_boost -= 0.1  # Crash likelihood after consecutive highs
                
                # Apply pattern boost
                prob = max(0.0, min(1.0, prob + pattern_boost))
                
                # Confidence boost from heavy model
                confidence_boost = self.heavy_knowledge.get_confidence_boost(prob)
                base_confidence = abs(prob - 0.5) * 2
                confidence = min(1.0, base_confidence + confidence_boost)
                
                # Value prediction with heavy model insights
                base_value = np.mean(sequence[-10:])
                if prob > 0.6:
                    # High confidence above threshold
                    value_multiplier = 1.3 + (prob - 0.6) * 0.5
                else:
                    # Lower confidence or below threshold
                    value_multiplier = 0.7 + prob * 0.6
                
                predicted_value = base_value * value_multiplier
                
            else:
                # Standard prediction without heavy model knowledge
                confidence = min(1.0, max(0.0, abs(prob - 0.5) * 2))
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
        """Heavy model bilgisi ile güçlendirilmiş ensemble tahmin"""
        if not self.is_fitted:
            return None, 0.5, 0.0
        
        predictions = []
        confidences = []
        values = []
        total_weight = 0
        knowledge_boost_count = 0
        
        for name, model in self.models.items():
            try:
                val, prob, conf = model.predict_next_value(sequence)
                if prob is not None:
                    weight = self.model_weights[name]
                    
                    # Heavy model bilgisi olan modellere extra weight
                    if hasattr(model, 'knowledge_boost_enabled') and model.knowledge_boost_enabled:
                        weight *= 1.2  # %20 extra weight
                        knowledge_boost_count += 1
                    
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
        
        # Ensemble-level heavy model knowledge boost
        if self.knowledge_boost_enabled and self.heavy_knowledge:
            # Global confidence boost
            ensemble_confidence_boost = self.heavy_knowledge.get_confidence_boost(avg_prob)
            avg_confidence = min(1.0, avg_confidence + ensemble_confidence_boost * 0.5)
            
            # Ensemble-level pattern adjustment
            recent_avg = np.mean(sequence[-10:])
            if recent_avg > 2.0 and avg_prob > 0.7:
                # High values with high confidence - slight reduction for safety
                avg_prob = max(0.5, avg_prob - 0.05)
            elif recent_avg < 1.3 and avg_prob < 0.3:
                # Low values with low confidence - slight increase
                avg_prob = min(0.5, avg_prob + 0.05)
        
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
