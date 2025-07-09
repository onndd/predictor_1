import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
import warnings

warnings.filterwarnings('ignore')


class LowValueFeatureExtractor:
    """
    1.5 altı değerler için özelleştirilmiş özellik çıkarımı
    """
    
    def __init__(self, threshold=1.5):
        self.threshold = threshold
        
    def extract_specialized_features(self, sequence):
        """
        1.5 altı tahminler için özel özellik çıkarımı
        """
        seq = np.array(sequence)
        features = []
        
        # === 1.5 ALTI ÖZEL ÖZELLİKLER ===
        
        # 1. Düşük değer yoğunluğu analizi
        low_thresholds = [1.1, 1.2, 1.3, 1.35, 1.4, 1.45]
        for th in low_thresholds:
            features.append(np.mean(seq <= th))  # Bu threshold altı oranı
            
        # 2. Düşük değer grupları
        features.extend([
            np.mean(seq <= 1.15),        # Çok düşük değerler
            np.mean((seq > 1.15) & (seq <= 1.35)),  # Orta-düşük değerler
            np.mean((seq > 1.35) & (seq < 1.5)),    # 1.5'e yakın düşük değerler
        ])
        
        # 3. Düşük değer trendleri
        if len(seq) > 10:
            # Son 10 değerde düşük değer trendi
            recent_low_trend = np.mean(seq[-10:] < self.threshold)
            features.append(recent_low_trend)
            
            # Düşük değerlerin momentum'u
            low_momentum = np.sum((seq[-10:] < seq[-20:-10]) & (seq[-10:] < self.threshold))
            features.append(low_momentum / 10)
        else:
            features.extend([0.0, 0.0])
            
        # 4. Crash pattern detection (yüksek değer sonrası düşük)
        crash_patterns = 0
        for i in range(1, min(len(seq), 20)):
            if seq[-i-1] > 2.0 and seq[-i] < 1.3:  # Crash pattern
                crash_patterns += 1
        features.append(crash_patterns / 20)
        
        # 5. Volatilite bazlı özellikler (düşük değerler için)
        if len(seq) > 5:
            low_values = seq[seq < self.threshold]
            if len(low_values) > 2:
                features.extend([
                    np.std(low_values),           # Düşük değerlerin volatilitesi
                    np.mean(low_values),          # Düşük değerlerin ortalaması
                    np.max(low_values),           # En yüksek düşük değer
                    np.min(low_values),           # En düşük değer
                ])
            else:
                features.extend([0.0, 1.2, 1.0, 1.0])
        else:
            features.extend([0.0, 1.2, 1.0, 1.0])
            
        # 6. Sequence içi düşük değer pozisyonları
        low_positions = []
        for i, val in enumerate(seq[-20:]):  # Son 20 değer
            if val < self.threshold:
                low_positions.append(i)
                
        if low_positions:
            features.extend([
                len(low_positions) / 20,      # Düşük değer yoğunluğu
                np.mean(low_positions),       # Ortalama pozisyon
                max(low_positions),           # En son düşük değer pozisyonu
            ])
        else:
            features.extend([0.0, 10.0, 0.0])
            
        # 7. Ardışık düşük değer sayıları
        consecutive_low = 0
        max_consecutive_low = 0
        for val in seq[-30:]:  # Son 30 değer
            if val < self.threshold:
                consecutive_low += 1
                max_consecutive_low = max(max_consecutive_low, consecutive_low)
            else:
                consecutive_low = 0
        features.extend([consecutive_low, max_consecutive_low])
        
        # 8. Teknik indikatorlar (düşük değer odaklı)
        if len(seq) >= 20:
            # Düşük değer moving average
            low_ma_5 = np.mean(seq[-5:])
            low_ma_20 = np.mean(seq[-20:])
            
            features.extend([
                low_ma_5,
                low_ma_20,
                low_ma_5 - low_ma_20,         # MA farkı
                (seq[-1] - low_ma_5) / low_ma_5 if low_ma_5 > 0 else 0,  # Son değer/MA oranı
            ])
            
            # RSI benzeri indikator (düşük değerler için)
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
            features.extend([np.mean(seq), np.mean(seq), 0.0, 0.0, 0.5])
            
        # 9. Genel istatistiksel özellikler
        features.extend([
            np.mean(seq),
            np.std(seq),
            np.percentile(seq, 10),      # 10. percentile
            np.percentile(seq, 25),      # 25. percentile
            np.percentile(seq, 50),      # Medyan
        ])
        
        return np.array(features)


class LowValueClassifier:
    """
    1.5 altı değerleri tahmin etmek için özelleştirilmiş sınıflandırıcı
    """
    
    def __init__(self, threshold=1.5, model_type='ensemble'):
        self.threshold = threshold
        self.model_type = model_type
        self.feature_extractor = LowValueFeatureExtractor(threshold)
        self.scaler = RobustScaler()  # Outlier'lara dayanıklı
        self.model = None
        self.is_fitted = False
        
    def _create_model(self):
        """Model oluştur"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=500,
                max_depth=25,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'  # Dengesiz veri için
            )
        
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            )
            
        elif self.model_type == 'neural_network':
            return MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
            
        elif self.model_type == 'ensemble':
            # Voting ensemble
            from sklearn.ensemble import VotingClassifier
            
            rf = RandomForestClassifier(
                n_estimators=300, max_depth=20, random_state=42, 
                class_weight='balanced', n_jobs=-1
            )
            
            gb = GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.08, max_depth=10,
                random_state=42
            )
            
            et = ExtraTreesClassifier(
                n_estimators=200, max_depth=15, random_state=42,
                class_weight='balanced', n_jobs=-1
            )
            
            return VotingClassifier(
                estimators=[('rf', rf), ('gb', gb), ('et', et)],
                voting='soft'
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, sequences, labels):
        """Model eğitimi"""
        print(f"Low Value Classifier ({self.model_type}) eğitiliyor...")
        
        # Feature extraction
        X = []
        for seq in sequences:
            features = self.feature_extractor.extract_specialized_features(seq)
            X.append(features)
            
        X = np.array(X)
        y = np.array(labels)
        
        # Özellik ölçekleme
        X_scaled = self.scaler.fit_transform(X)
        
        # Model oluştur ve eğit
        self.model = self._create_model()
        
        # Probability calibration ile daha iyi tahminler
        self.model = CalibratedClassifierCV(
            self.model, 
            method='isotonic',  # Sigmoid yerine isotonic
            cv=3
        )
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        print(f"Low Value Classifier eğitimi tamamlandı!")
        return self
    
    def predict_low_value_probability(self, sequence):
        """1.5 altı olma olasılığını tahmin et"""
        if not self.is_fitted:
            return 0.5, 0.0
            
        try:
            features = self.feature_extractor.extract_specialized_features(sequence)
            X = features.reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Olasılık tahmini
            prob_low = self.model.predict_proba(X_scaled)[0][1]  # 1.5 altı olma olasılığı
            
            # Güven skoru
            if hasattr(self.model, 'estimators_'):
                # Ensemble model - individual predictions
                individual_probs = []
                for estimator in self.model.estimators_:
                    try:
                        if hasattr(estimator, 'predict_proba'):
                            ind_prob = estimator.predict_proba(X_scaled)[0][1]
                            individual_probs.append(ind_prob)
                    except:
                        continue
                        
                if individual_probs:
                    confidence = 1.0 - np.std(individual_probs)  # Model agreement
                else:
                    confidence = abs(prob_low - 0.5) * 2  # Distance from uncertainty
            else:
                confidence = abs(prob_low - 0.5) * 2
                
            confidence = max(0.0, min(1.0, confidence))
            
            return prob_low, confidence
            
        except Exception as e:
            print(f"Low value prediction error: {e}")
            return 0.5, 0.0


class LowValueSpecialist:
    """
    1.5 altı değerler için özelleştirilmiş tahmin sistemi
    """
    
    def __init__(self, threshold=1.5):
        self.threshold = threshold
        self.classifiers = {}
        self.ensemble_weights = {}
        self.is_fitted = False
        
    def fit(self, sequences, labels):
        """Tüm özelleştirilmiş modelleri eğit"""
        print("Low Value Specialist eğitim sistemi başlıyor...")
        
        # Farklı model türlerini eğit
        model_types = ['random_forest', 'gradient_boosting', 'neural_network', 'ensemble']
        
        performances = {}
        
        # Time series cross validation
        tscv = TimeSeriesSplit(n_splits=3)
        X_dummy = np.arange(len(sequences)).reshape(-1, 1)  # Dummy X for tscv
        
        for model_type in model_types:
            print(f"  -> {model_type} eğitiliyor...")
            
            classifier = LowValueClassifier(self.threshold, model_type)
            
            # Cross validation performance
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_dummy):
                train_sequences = [sequences[i] for i in train_idx]
                train_labels = [labels[i] for i in train_idx]
                val_sequences = [sequences[i] for i in val_idx]
                val_labels = [labels[i] for i in val_idx]
                
                # Temporary model training
                temp_classifier = LowValueClassifier(self.threshold, model_type)
                temp_classifier.fit(train_sequences, train_labels)
                
                # Validation predictions
                val_predictions = []
                for seq in val_sequences:
                    prob, _ = temp_classifier.predict_low_value_probability(seq)
                    val_predictions.append(1 if prob > 0.5 else 0)
                
                # Performance metrics
                accuracy = accuracy_score(val_labels, val_predictions)
                precision = precision_score(val_labels, val_predictions, zero_division=0)
                recall = recall_score(val_labels, val_predictions, zero_division=0)
                f1 = f1_score(val_labels, val_predictions, zero_division=0)
                
                # Özellikle recall'ı önemse (1.5 altı değerleri kaçırmama)
                combined_score = (accuracy + 2*recall + precision + f1) / 5
                cv_scores.append(combined_score)
            
            avg_performance = np.mean(cv_scores)
            performances[model_type] = avg_performance
            
            # Final model training
            classifier.fit(sequences, labels)
            self.classifiers[model_type] = classifier
            
            print(f"     CV Performance: {avg_performance:.3f}")
        
        # Ensemble weights (performance-based)
        total_performance = sum(performances.values())
        for model_type, perf in performances.items():
            self.ensemble_weights[model_type] = perf / total_performance
            
        self.is_fitted = True
        
        print("Low Value Specialist eğitimi tamamlandı!")
        print("Ensemble weights:")
        for model_type, weight in self.ensemble_weights.items():
            print(f"  {model_type}: {weight:.3f}")
            
        return performances
    
    def predict_low_value(self, sequence):
        """
        1.5 altı tahmin - ensemble yaklaşımı
        
        Returns:
            tuple: (is_low_value, probability, confidence, detailed_predictions)
        """
        if not self.is_fitted:
            return False, 0.5, 0.0, {}
            
        predictions = {}
        weighted_probs = []
        confidences = []
        
        # Her modelden tahmin al
        for model_type, classifier in self.classifiers.items():
            prob, conf = classifier.predict_low_value_probability(sequence)
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
        
        # Final decision
        is_low_value = ensemble_prob > 0.5
        
        # Conservative threshold için - 1.5 altı tahminlerinde daha dikkatli ol
        conservative_threshold = 0.6
        if ensemble_prob > conservative_threshold:
            is_low_value = True
        elif ensemble_prob < (1 - conservative_threshold):
            is_low_value = False
        else:
            # Belirsiz durumda en güvenilir modelin kararını al
            best_model = max(predictions.keys(), 
                           key=lambda x: predictions[x]['confidence'])
            is_low_value = predictions[best_model]['prediction']
        
        return is_low_value, ensemble_prob, ensemble_confidence, predictions
    
    def get_low_value_insights(self, sequence):
        """
        1.5 altı değerler hakkında detaylı analiz
        """
        if not self.is_fitted:
            return {}
            
        feature_extractor = LowValueFeatureExtractor(self.threshold)
        features = feature_extractor.extract_specialized_features(sequence)
        
        seq = np.array(sequence)
        
        insights = {
            'recent_low_ratio': np.mean(seq[-10:] < self.threshold),
            'very_low_ratio': np.mean(seq < 1.2),
            'low_trend': np.mean(seq[-5:] < self.threshold) - np.mean(seq[-15:-5] < self.threshold),
            'consecutive_low': self._count_consecutive_low(seq),
            'crash_patterns': self._detect_crash_patterns(seq),
            'low_volatility': np.std(seq[seq < self.threshold]) if len(seq[seq < self.threshold]) > 1 else 0,
            'prediction_summary': self.predict_low_value(sequence)
        }
        
        return insights
    
    def _count_consecutive_low(self, sequence):
        """Ardışık düşük değerleri say"""
        consecutive = 0
        for val in reversed(sequence):
            if val < self.threshold:
                consecutive += 1
            else:
                break
        return consecutive
    
    def _detect_crash_patterns(self, sequence):
        """Crash pattern'lerini tespit et"""
        patterns = 0
        for i in range(1, min(len(sequence), 10)):
            if sequence[-i-1] > 2.5 and sequence[-i] < 1.3:
                patterns += 1
        return patterns


def create_enhanced_low_value_system(threshold=1.5):
    """
    1.5 altı değerler için gelişmiş tahmin sistemi oluştur
    """
    return LowValueSpecialist(threshold)