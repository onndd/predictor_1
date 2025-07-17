import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
import warnings

warnings.filterwarnings('ignore')


class HighValueFeatureExtractor:
    """
    10x ve üzeri değerler için özelleştirilmiş özellik çıkarımı
    """
    
    def __init__(self, threshold=10.0):
        self.threshold = threshold
        
    def extract_specialized_features(self, sequence):
        """
        Yüksek değer tahminler için özel özellik çıkarımı
        """
        seq = np.array(sequence)
        features = []
        
        # === YÜKSEK DEĞER ÖZEL ÖZELLİKLER ===
        
        # 1. Yüksek değer yoğunluğu analizi - Çoklu threshold'lar
        high_thresholds = [2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 50.0]
        for th in high_thresholds:
            features.append(np.mean(seq >= th))  # Bu threshold üstü oranı
            
        # 2. Yüksek değer kategorileri
        features.extend([
            np.mean(seq >= 10.0),                # 10x+ oranı
            np.mean((seq >= 10.0) & (seq < 20.0)),   # 10-20x arası
            np.mean((seq >= 20.0) & (seq < 50.0)),   # 20-50x arası
            np.mean((seq >= 50.0) & (seq < 100.0)),  # 50-100x arası
            np.mean(seq >= 100.0),               # 100x+ oranı
        ])
        
        # 3. Momentum ve trend analizi (yüksek değerler için)
        if len(seq) > 10:
            # Son 10 değerde yüksek değer momentum'u
            recent_high_trend = np.mean(seq[-10:] > np.mean(seq[-20:-10]))
            features.append(recent_high_trend)
            
            # Yükseliş trendi (exponential growth detection)
            if len(seq) >= 20:
                recent_slope = np.polyfit(range(10), seq[-10:], 1)[0]
                older_slope = np.polyfit(range(10), seq[-20:-10], 1)[0]
                momentum_change = recent_slope - older_slope
                features.append(momentum_change / max(1.0, np.mean(seq[-20:])))
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0])
            
        # 4. Build-up pattern detection (yüksek değer öncesi pattern'ler)
        buildup_patterns = 0
        if len(seq) >= 5:
            for i in range(1, min(len(seq), 15)):
                # Gradual increase pattern
                if (seq[-i-1] < seq[-i] and seq[-i] > 2.0 and 
                    seq[-i-1] > 1.5):  # Building up pattern
                    buildup_patterns += 1
        features.append(buildup_patterns / 15)
        
        # 5. Volatilite bazlı özellikler (yüksek değerler için)
        if len(seq) > 5:
            high_values = seq[seq >= 3.0]  # 3x üzeri değerleri analiz et
            if len(high_values) > 2:
                features.extend([
                    np.std(high_values),           # Yüksek değerlerin volatilitesi
                    np.mean(high_values),          # Yüksek değerlerin ortalaması
                    np.max(high_values),           # En yüksek değer
                    len(high_values) / len(seq),   # Yüksek değer yoğunluğu
                ])
            else:
                features.extend([0.0, 2.0, 2.0, 0.0])
        else:
            features.extend([0.0, 2.0, 2.0, 0.0])
            
        # 6. Sequence içi yüksek değer pozisyonları ve pattern'ler
        high_positions = []
        for i, val in enumerate(seq[-30:]):  # Son 30 değer
            if val >= 5.0:  # 5x üzeri değerleri takip et
                high_positions.append(i)
                
        if high_positions:
            features.extend([
                len(high_positions) / 30,      # Yüksek değer yoğunluğu
                np.mean(high_positions),       # Ortalama pozisyon
                max(high_positions),           # En son yüksek değer pozisyonu
                max(high_positions) / 30,      # Son yüksek değerin recency'si
            ])
        else:
            features.extend([0.0, 15.0, 0.0, 0.0])
            
        # 7. Ardışık yüksek değer ve growth pattern'ler
        consecutive_high = 0
        max_consecutive_high = 0
        growth_streaks = 0
        
        for i in range(len(seq[-20:])):  # Son 20 değer
            val = seq[-20:][i]
            if val >= 2.0:  # 2x üzeri değerler
                consecutive_high += 1
                max_consecutive_high = max(max_consecutive_high, consecutive_high)
                
                # Growth streak detection
                if i > 0 and val > seq[-20:][i-1] * 1.2:  # %20+ artış
                    growth_streaks += 1
            else:
                consecutive_high = 0
                
        features.extend([consecutive_high, max_consecutive_high, growth_streaks])
        
        # 8. Teknik indikatorlar (yüksek değer odaklı)
        if len(seq) >= 20:
            # Exponential Moving Averages
            ema_5 = self._calculate_ema(seq[-5:], 5)
            ema_20 = self._calculate_ema(seq[-20:], 20)
            
            features.extend([
                ema_5,
                ema_20,
                ema_5 / ema_20 if ema_20 > 0 else 1.0,  # EMA ratio
                (seq[-1] - ema_5) / ema_5 if ema_5 > 0 else 0.0,  # Distance from EMA
            ])
            
            # Acceleration indicator (2nd derivative)
            if len(seq) >= 25:
                recent_change = np.mean(np.diff(seq[-5:]))
                older_change = np.mean(np.diff(seq[-15:-10]))
                acceleration = recent_change - older_change
                features.append(acceleration / max(1.0, np.mean(seq[-15:])))
            else:
                features.append(0.0)
                
            # Relative Strength Index (RSI) - Modified for high values
            gains = []
            losses = []
            for i in range(1, min(len(seq), 15)):
                change = seq[-i] - seq[-i-1]
                if change > 0:
                    gains.append(change)
                else:
                    losses.append(abs(change))
                    
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0
            
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50
                
            features.append(rsi / 100)  # 0-1 normalize
        else:
            features.extend([np.mean(seq), np.mean(seq), 1.0, 0.0, 0.0, 0.5])
            
        # 9. Extremeness detection (outlier pattern'ler)
        if len(seq) >= 10:
            # Z-score based extremeness
            mean_val = np.mean(seq)
            std_val = np.std(seq)
            if std_val > 0:
                recent_z_scores = [(val - mean_val) / std_val for val in seq[-5:]]
                max_z_score = max(recent_z_scores)
                features.append(max(0, max_z_score))  # Positive z-scores only
            else:
                features.append(0.0)
                
            # Outlier ratio
            outlier_threshold = np.percentile(seq, 90)  # Top 10% threshold
            recent_outliers = np.sum(seq[-10:] > outlier_threshold)
            features.append(recent_outliers / 10)
        else:
            features.extend([0.0, 0.0])
            
        # 10. Statistical features adapted for high values
        features.extend([
            np.mean(seq),
            np.std(seq),
            np.max(seq),
            np.percentile(seq, 90),      # 90th percentile
            np.percentile(seq, 95),      # 95th percentile
            np.mean(seq[-5:]) / np.mean(seq[-15:-5]) if np.mean(seq[-15:-5]) > 0 else 1.0,  # Recent vs older ratio
        ])
        
        return np.array(features)
    
    def _calculate_ema(self, values, period):
        """Exponential Moving Average hesapla"""
        if len(values) == 0:
            return 0.0
        
        alpha = 2.0 / (period + 1)
        ema = values[0]
        
        for value in values[1:]:
            ema = alpha * value + (1 - alpha) * ema
            
        return ema


class HighValueClassifier:
    """
    10x ve üzeri değerleri tahmin etmek için özelleştirilmiş sınıflandırıcı
    """
    
    def __init__(self, threshold=10.0, model_type='ensemble'):
        self.threshold = threshold
        self.model_type = model_type
        self.feature_extractor = HighValueFeatureExtractor(threshold)
        self.scaler = RobustScaler()  # Outlier'lara dayanıklı
        self.model = None
        self.is_fitted = False
        
    def _create_model(self):
        """Model oluştur"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=600,  # Daha fazla tree (rare event için)
                max_depth=30,
                min_samples_split=2,  # Daha aggressive splitting
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'  # Dengesiz veri için critical
            )
        
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=500,
                learning_rate=0.03,  # Daha düşük learning rate (overfitting önleme)
                max_depth=15,
                min_samples_split=3,
                min_samples_leaf=1,
                subsample=0.8,
                random_state=42
            )
            
        elif self.model_type == 'neural_network':
            return MLPClassifier(
                hidden_layer_sizes=(256, 128, 64, 32),  # Daha derin network
                activation='relu',
                solver='adam',
                alpha=0.0001,  # L2 regularization
                learning_rate='adaptive',
                max_iter=800,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
            
        elif self.model_type == 'svm_high':
            return SVC(
                kernel='rbf',
                C=50.0,  # Daha yüksek C (complex patterns için)
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42
            )
            
        elif self.model_type == 'ensemble':
            # Specialized voting ensemble for high values
            from sklearn.ensemble import VotingClassifier
            
            rf = RandomForestClassifier(
                n_estimators=400, max_depth=25, random_state=42, 
                class_weight='balanced', n_jobs=-1,
                min_samples_split=2, min_samples_leaf=1
            )
            
            gb = GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=12,
                random_state=42, subsample=0.8
            )
            
            et = ExtraTreesClassifier(
                n_estimators=300, max_depth=20, random_state=42,
                class_weight='balanced', n_jobs=-1,
                min_samples_split=2
            )
            
            svm = SVC(
                kernel='rbf', C=20.0, probability=True,
                class_weight='balanced', random_state=42
            )
            
            return VotingClassifier(
                estimators=[('rf', rf), ('gb', gb), ('et', et), ('svm', svm)],
                voting='soft'
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, sequences, labels):
        """Model eğitimi"""
        print(f"High Value Classifier ({self.model_type}) eğitiliyor...")
        
        # Feature extraction
        X = []
        for seq in sequences:
            features = self.feature_extractor.extract_specialized_features(seq)
            X.append(features)
            
        X = np.array(X)
        y = np.array(labels)
        
        print(f"Features shape: {X.shape}")
        print(f"Positive samples: {np.sum(y)} / {len(y)} ({np.sum(y)/len(y)*100:.1f}%)")
        
        # Özellik ölçekleme
        X_scaled = self.scaler.fit_transform(X)
        
        # Model oluştur ve eğit
        self.model = self._create_model()
        
        # Probability calibration ile daha iyi tahminler (özellikle rare events için)
        self.model = CalibratedClassifierCV(
            self.model, 
            method='isotonic',  # Sigmoid yerine isotonic (rare events için daha iyi)
            cv=5  # Daha fazla fold (daha stabil calibration)
        )
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        print(f"High Value Classifier eğitimi tamamlandı!")
        return self
    
    def predict_high_value_probability(self, sequence):
        """10x üzeri olma olasılığını tahmin et"""
        if not self.is_fitted or self.model is None:
            return 0.1, 0.0  # Low default probability for rare events
            
        try:
            features = self.feature_extractor.extract_specialized_features(sequence)
            X = features.reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Olasılık tahmini
            prob_high = self.model.predict_proba(X_scaled)[0][1]
            
            # Güven skoru
            confidence = abs(float(prob_high) - 0.1) * 2 # Distance from low baseline
            confidence = max(0.0, min(1.0, confidence))
            
            return float(prob_high), confidence
            
        except Exception as e:
            print(f"High value prediction error: {e}")
            return 0.1, 0.0


class HighValueSpecialist:
    """
    10x ve üzeri değerler için özelleştirilmiş tahmin sistemi
    """
    
    def __init__(self, threshold=10.0):
        self.threshold = threshold
        self.classifiers = {}
        self.ensemble_weights = {}
        self.is_fitted = False
        
    def fit(self, sequences, labels):
        """Tüm özelleştirilmiş modelleri eğit"""
        print("High Value Specialist eğitim sistemi başlıyor...")
        
        # Farklı model türlerini eğit
        model_types = ['random_forest', 'gradient_boosting', 'neural_network', 'svm_high', 'ensemble']
        
        performances = {}
        
        # Time series cross validation - rare events için dikkatli
        tscv = TimeSeriesSplit(n_splits=5)  # Daha fazla split
        X_dummy = np.arange(len(sequences)).reshape(-1, 1)
        
        for model_type in model_types:
            print(f"  -> {model_type} eğitiliyor...")
            
            classifier = HighValueClassifier(self.threshold, model_type)
            
            # Cross validation performance
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_dummy):
                train_sequences = [sequences[i] for i in train_idx]
                train_labels = [labels[i] for i in train_idx]
                val_sequences = [sequences[i] for i in val_idx]
                val_labels = [labels[i] for i in val_idx]
                
                # Check if we have positive samples in training
                if sum(train_labels) == 0:
                    print(f"     Fold skipped - no positive samples in training")
                    continue
                
                # Temporary model training
                temp_classifier = HighValueClassifier(self.threshold, model_type)
                temp_classifier.fit(train_sequences, train_labels)
                
                # Validation predictions
                val_predictions = []
                for seq in val_sequences:
                    prob, _ = temp_classifier.predict_high_value_probability(seq)
                    val_predictions.append(1 if prob > 0.5 else 0)
                
                # Performance metrics - özellikle recall'ı önemse (rare events kaçırmama)
                if len(set(val_labels)) > 1:  # Both classes present
                    accuracy = accuracy_score(val_labels, val_predictions)
                    precision = precision_score(val_labels, val_predictions, zero_division='warn')
                    recall = recall_score(val_labels, val_predictions, zero_division='warn')
                    f1 = f1_score(val_labels, val_predictions, zero_division='warn')
                    
                    # Yüksek değerler için recall çok önemli (kaçırmamak)
                    combined_score = (accuracy + 3*recall + precision + f1) / 6
                    cv_scores.append(combined_score)
                else:
                    print(f"     Fold skipped - only one class in validation")
            
            if cv_scores:
                avg_performance = np.mean(cv_scores)
                performances[model_type] = avg_performance
                
                # Final model training
                classifier.fit(sequences, labels)
                self.classifiers[model_type] = classifier
                
                print(f"     CV Performance: {avg_performance:.3f}")
            else:
                print(f"     Model {model_type} could not be evaluated - skipping")
        
        if not performances:
            print("❌ No models could be trained successfully!")
            return {}
        
        # Ensemble weights (performance-based)
        total_performance = sum(performances.values())
        for model_type, perf in performances.items():
            self.ensemble_weights[model_type] = perf / total_performance
            
        self.is_fitted = True
        
        print("High Value Specialist eğitimi tamamlandı!")
        print("Ensemble weights:")
        for model_type, weight in self.ensemble_weights.items():
            print(f"  {model_type}: {weight:.3f}")
            
        return performances
    
    def predict_high_value(self, sequence):
        """
        10x üzeri tahmin - ensemble yaklaşımı
        
        Returns:
            tuple: (is_high_value, probability, confidence, detailed_predictions)
        """
        if not self.is_fitted:
            return False, 0.1, 0.0, {}
            
        predictions = {}
        weighted_probs = []
        confidences = []
        
        # Her modelden tahmin al
        for model_type, classifier in self.classifiers.items():
            prob, conf = classifier.predict_high_value_probability(sequence)
            weight = self.ensemble_weights[model_type]
            
            predictions[model_type] = {
                'probability': prob,
                'confidence': conf,
                'weight': weight,
                'prediction': prob > 0.5
            }
            
            weighted_probs.append(prob * weight)
            confidences.append(conf * weight)
        
        # Ensemble sonucu
        ensemble_prob = sum(weighted_probs)
        ensemble_confidence = sum(confidences)
        
        # Final decision - rare events için conservative
        # Yüksek değerler nadir olduğu için daha yüksek threshold
        high_value_threshold = 0.4  # Düşük threshold (rare events için)
        
        is_high_value = ensemble_prob > high_value_threshold
        
        # Confidence-based adjustment
        if ensemble_confidence < 0.3:
            is_high_value = False  # Düşük güvende hayır de
        elif ensemble_prob > 0.7 and ensemble_confidence > 0.6:
            is_high_value = True   # Yüksek güven ve prob'da evet
        
        return is_high_value, ensemble_prob, ensemble_confidence, predictions
    
    def get_high_value_insights(self, sequence):
        """
        10x üzeri değerler hakkında detaylı analiz
        """
        if not self.is_fitted:
            return {}
            
        feature_extractor = HighValueFeatureExtractor(self.threshold)
        features = feature_extractor.extract_specialized_features(sequence)
        
        seq = np.array(sequence)
        
        insights = {
            'recent_high_ratio': np.mean(seq[-10:] >= 5.0),  # Son 10'da 5x+ oranı
            'very_high_ratio': np.mean(seq >= self.threshold),  # 10x+ oranı
            'momentum_trend': np.mean(seq[-5:]) / np.mean(seq[-15:-5]) if np.mean(seq[-15:-5]) > 0 else 1.0,
            'consecutive_growth': self._count_consecutive_growth(seq),
            'buildup_patterns': self._detect_buildup_patterns(seq),
            'high_volatility': np.std(seq[seq >= 3.0]) if len(seq[seq >= 3.0]) > 1 else 0,
            'max_recent_value': np.max(seq[-20:]) if len(seq) >= 20 else np.max(seq),
            'prediction_summary': self.predict_high_value(sequence)
        }
        
        return insights
    
    def _count_consecutive_growth(self, sequence):
        """Ardışık büyüme sayısını say"""
        consecutive = 0
        for i in range(len(sequence)-1, 0, -1):
            if sequence[i] > sequence[i-1] * 1.1:  # %10+ artış
                consecutive += 1
            else:
                break
        return consecutive
    
    def _detect_buildup_patterns(self, sequence):
        """Buildup pattern'lerini tespit et"""
        patterns = 0
        for i in range(1, min(len(sequence), 15)):
            if (sequence[-i-1] < sequence[-i] and 
                sequence[-i] > 2.0 and 
                sequence[-i-1] > 1.5):
                patterns += 1
        return patterns


def create_enhanced_high_value_system(threshold=10.0):
    """
    10x üzeri değerler için gelişmiş tahmin sistemi oluştur
    """
    return HighValueSpecialist(threshold)