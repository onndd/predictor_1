import numpy as np
import pandas as pd
import os
import joblib
import warnings
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf

warnings.filterwarnings('ignore')

class FeatureExtractor:
    """
    Ağır modelleri (LSTM, Transformer, GRU) kullanarak özellik çıkarımı yapar
    """
    def __init__(self, models_path="saved_models/heavy_models"):
        self.models_path = models_path
        self.heavy_models = {}
        self.feature_cache = {}
        self.seq_length = 200
        
    def add_heavy_model(self, name, model):
        """Ağır model ekle"""
        self.heavy_models[name] = model
        
    def extract_features(self, sequence):
        """
        Tüm ağır modellerden özellik çıkar
        
        Args:
            sequence: Input sequence
            
        Returns:
            np.array: Combined features from all heavy models
        """
        features = []
        
        # Sequence'i normalize et
        normalized_seq = self._normalize_sequence(sequence)
        
        for model_name, model in self.heavy_models.items():
            try:
                if hasattr(model, 'extract_features'):
                    # Model'den özellik çıkar
                    model_features = model.extract_features(normalized_seq)
                elif hasattr(model, 'model') and model.model is not None:
                    # LSTM/GRU/Transformer'dan hidden states al
                    model_features = self._extract_hidden_features(model, normalized_seq)
                else:
                    # Fallback: model prediction'ı feature olarak kullan
                    _, prob = model.predict_next(sequence)
                    model_features = [prob if prob is not None else 0.5]
                    
                features.extend(model_features)
                
            except Exception as e:
                print(f"Feature extraction error for {model_name}: {e}")
                # Error durumunda default feature
                features.extend([0.5, 0.0, 0.0])
                
        # İstatistiksel özellikler ekle
        stats_features = self._extract_statistical_features(sequence)
        features.extend(stats_features)
        
        return np.array(features)
    
    def _extract_hidden_features(self, model, sequence):
        """Neural network'ten hidden features çıkar - Robust implementation"""
        try:
            # Sequence'i normalize et
            sequence = self._normalize_sequence(sequence)
            
            # Sequence'i model input formatına çevir
            if len(sequence) < self.seq_length:
                padding = [sequence[0]] * (self.seq_length - len(sequence))
                sequence = padding + list(sequence)
            
            sequence = sequence[-self.seq_length:]
            
            # Model type'a göre feature extraction
            if hasattr(model, 'model') and model.model is not None:
                # TensorFlow models
                if hasattr(model.model, 'layers'):
                    return self._extract_from_tf_model(model, sequence)
                # PyTorch models
                elif hasattr(model, 'predict'):
                    return self._extract_from_torch_model(model, sequence)
            
            # Fallback: Use model's predict method
            if hasattr(model, 'predict_next_value'):
                result = model.predict_next_value(sequence)
                if isinstance(result, tuple) and len(result) >= 2:
                    _, prob, confidence = result if len(result) == 3 else (*result, 0.5)
                    return [prob, confidence, np.mean(sequence[-5:]), np.std(sequence[-5:])]
                else:
                    return [0.5, 0.0, np.mean(sequence[-5:]), np.std(sequence[-5:])]
            
            # Final fallback
            return [0.5, 0.0, np.mean(sequence[-5:]), np.std(sequence[-5:])]
            
        except Exception as e:
            print(f"Hidden feature extraction error: {e}")
            return [0.5, 0.0, np.mean(sequence[-5:]) if sequence else 1.5, 0.0]
    
    def _extract_from_tf_model(self, model, sequence):
        """TensorFlow modelinden feature çıkar"""
        try:
            X = np.array(sequence).reshape(1, self.seq_length, 1)
            
            # Model prediction
            if hasattr(model, 'predict_next'):
                _, prob = model.predict_next(sequence)
                prob = prob if prob is not None else 0.5
            else:
                prob = 0.5
            
            # Hidden layer features
            hidden_features = []
            
            # Try to get intermediate outputs
            try:
                for i, layer in enumerate(model.model.layers):
                    if any(layer_type in layer.name.lower() for layer_type in ['lstm', 'gru', 'dense']):
                        if i < len(model.model.layers) - 1:  # Not output layer
                            intermediate_model = tf.keras.Model(
                                inputs=model.model.input,
                                outputs=layer.output
                            )
                            hidden_output = intermediate_model.predict(X, verbose=0)
                            
                            # Extract features from hidden output
                            if len(hidden_output.shape) == 3:  # (batch, seq, features)
                                features = hidden_output[0, -1, :]  # Last timestep
                            else:  # (batch, features)
                                features = hidden_output[0, :]
                            
                            # Take first few features
                            hidden_features.extend(features[:3].tolist())
                            
                            if len(hidden_features) >= 6:  # Limit features
                                break
            except Exception as e:
                print(f"Intermediate layer extraction failed: {e}")
            
            # Combine features
            features = [prob] + hidden_features[:6]
            
            # Pad to consistent length
            while len(features) < 8:
                features.append(0.0)
            
            return features[:8]
            
        except Exception as e:
            print(f"TF model feature extraction error: {e}")
            return [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def _extract_from_torch_model(self, model, sequence):
        """PyTorch modelinden feature çıkar"""
        try:
            # Model prediction
            if hasattr(model, 'predict'):
                pred = model.predict(sequence)
                prob = pred if pred is not None else 0.5
            else:
                prob = 0.5
            
            # Statistical features from sequence
            seq_arr = np.array(sequence)
            features = [
                prob,
                np.mean(seq_arr[-10:]),
                np.std(seq_arr[-10:]),
                np.max(seq_arr[-10:]),
                np.min(seq_arr[-10:]),
                np.mean(seq_arr[-5:]),
                len(seq_arr[seq_arr >= 1.5]) / len(seq_arr),
                len(seq_arr[seq_arr >= 2.0]) / len(seq_arr)
            ]
            
            return features
            
        except Exception as e:
            print(f"PyTorch model feature extraction error: {e}")
            return [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def _extract_statistical_features(self, sequence):
        """İstatistiksel özellikler çıkar"""
        seq = np.array(sequence[-50:])  # Son 50 değer
        
        features = [
            np.mean(seq),                    # Ortalama
            np.std(seq),                     # Standart sapma
            np.max(seq),                     # Maksimum
            np.min(seq),                     # Minimum
            np.median(seq),                  # Medyan
            len(seq[seq >= 1.5]) / len(seq), # 1.5 üstü oranı
            len(seq[seq >= 2.0]) / len(seq), # 2.0 üstü oranı
            len(seq[seq >= 3.0]) / len(seq), # 3.0 üstü oranı
            np.sum(np.diff(seq) > 0) / (len(seq)-1),  # Yükseliş trendi
            np.sum(np.diff(seq) < 0) / (len(seq)-1),  # Düşüş trendi
        ]
        
        return features
    
    def _normalize_sequence(self, sequence):
        """Sequence'i normalize et"""
        seq = np.array(sequence)
        # Min-max normalization
        if len(seq) > 1:
            seq_min, seq_max = seq.min(), seq.max()
            if seq_max > seq_min:
                seq = (seq - seq_min) / (seq_max - seq_min)
        return seq.tolist()


class HybridPredictor:
    """
    Hibrit tahmin sistemi:
    - Ağır modeller feature extraction için
    - Hafif modeller hızlı tahmin için
    """
    
    def __init__(self, models_path="saved_models"):
        self.models_path = models_path
        self.heavy_models_path = os.path.join(models_path, "heavy_models")
        self.light_models_path = os.path.join(models_path, "light_models")
        
        # Klasörleri oluştur
        os.makedirs(self.heavy_models_path, exist_ok=True)
        os.makedirs(self.light_models_path, exist_ok=True)
        
        self.feature_extractor = FeatureExtractor(self.heavy_models_path)
        self.light_models = {}
        self.ensemble_model = None
        self.threshold = 1.5
        self.is_fitted = False
        
        # Training cache
        self.training_features = []
        self.training_labels = []
        
    def add_heavy_model(self, name, model):
        """Ağır model ekle"""
        self.feature_extractor.add_heavy_model(name, model)
        
    def add_light_model(self, name, model):
        """Hafif model ekle"""
        self.light_models[name] = model
        
    def fit(self, sequences, labels, test_size=0.2):
        """
        Hibrit sistemi eğit
        
        Args:
            sequences: List of sequences
            labels: Corresponding labels (1 if >= threshold, 0 otherwise)
        """
        print("Hibrit sistem eğitiliyor...")
        
        # 1. Ağır modellerden özellik çıkar
        print("  -> Ağır modellerden özellikler çıkarılıyor...")
        features_list = []
        
        for i, seq in enumerate(sequences):
            if i % 100 == 0:
                print(f"     İşlenen: {i}/{len(sequences)}")
                
            features = self.feature_extractor.extract_features(seq)
            features_list.append(features)
            
        X = np.array(features_list)
        y = np.array(labels)
        
        # 2. Train/test split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"  -> Eğitim seti: {len(X_train)}, Test seti: {len(X_test)}")
        
        # 3. Hafif modelleri eğit
        print("  -> Hafif modeller eğitiliyor...")
        
        # Random Forest (Ana model)
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.light_models['random_forest'] = rf
        
        # Gradient Boosting (Güçlü model)
        gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
        gb.fit(X_train, y_train)
        self.light_models['gradient_boosting'] = gb
        
        # Logistic Regression (Hızlı model)
        lr = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        lr.fit(X_train, y_train)
        self.light_models['logistic_regression'] = lr
        
        # 4. Ensemble model (Voting)
        from sklearn.ensemble import VotingClassifier
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('lr', lr)
            ],
            voting='soft'  # Probability-based voting
        )
        ensemble.fit(X_train, y_train)
        self.ensemble_model = ensemble
        
        # 5. Performansı değerlendir
        print("  -> Performans değerlendiriliyor...")
        for name, model in self.light_models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"     {name}: {accuracy:.3f}")
            
        # Ensemble performansı
        y_pred_ensemble = ensemble.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        print(f"     Ensemble: {ensemble_accuracy:.3f}")
        
        self.is_fitted = True
        print("Hibrit sistem eğitimi tamamlandı!")
        
        return ensemble_accuracy
    
    def predict_next_value(self, sequence):
        """
        Hızlı tahmin yap
        
        Args:
            sequence: Input sequence
            
        Returns:
            tuple: (predicted_value, above_threshold_probability, confidence)
        """
        if not self.is_fitted:
            print("Model henüz eğitilmemiş!")
            return None, 0.5, 0.0
            
        try:
            # Özellik çıkar
            features = self.feature_extractor.extract_features(sequence)
            X = features.reshape(1, -1)
            
            # Ensemble ile tahmin
            prob = self.ensemble_model.predict_proba(X)[0][1]  # Positive class probability
            
            # Individual model predictions (confidence için)
            predictions = []
            for model in self.light_models.values():
                pred_prob = model.predict_proba(X)[0][1]
                predictions.append(pred_prob)
                
            # Confidence: Model agreement
            confidence = 1.0 - np.std(predictions)  # Düşük std = yüksek agreement
            confidence = max(0.0, min(1.0, confidence))  # 0-1 arası sınırla
            
            # Predicted value (sequence'den tahmin)
            recent_mean = np.mean(sequence[-10:])
            predicted_value = recent_mean * (1.2 if prob > 0.5 else 0.8)
            
            return predicted_value, prob, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.5, 0.0
    
    def save_models(self):
        """Tüm modelleri kaydet"""
        print(f"Modeller kaydediliyor: {self.models_path}")
        
        try:
            # Light models
            light_models_file = os.path.join(self.light_models_path, "light_models.joblib")
            joblib.dump(self.light_models, light_models_file)
            
            ensemble_file = os.path.join(self.light_models_path, "ensemble.joblib")
            joblib.dump(self.ensemble_model, ensemble_file)
            
            # Heavy models (eğer kaydedilebilirse)
            for name, model in self.feature_extractor.heavy_models.items():
                try:
                    if hasattr(model, 'save') and hasattr(model, 'model') and model.model is not None:
                        model_path = os.path.join(self.heavy_models_path, f"{name}.h5")
                        model.save(model_path)
                        print(f"  -> {name} kaydedildi")
                except Exception as e:
                    print(f"  -> {name} kaydedilemedi: {e}")
                    
            # Metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'threshold': self.threshold,
                'is_fitted': self.is_fitted,
                'feature_count': len(self.feature_extractor.extract_features([1.5] * 50))
            }
            
            metadata_file = os.path.join(self.models_path, "metadata.joblib")
            joblib.dump(metadata, metadata_file)
            
            print("Tüm modeller başarıyla kaydedildi!")
            return True
            
        except Exception as e:
            print(f"Model kaydetme hatası: {e}")
            return False
    
    def load_models(self):
        """Kaydedilmiş modelleri yükle"""
        print(f"Modeller yükleniyor: {self.models_path}")
        
        try:
            # Light models
            light_models_file = os.path.join(self.light_models_path, "light_models.joblib")
            if os.path.exists(light_models_file):
                self.light_models = joblib.load(light_models_file)
                print("  -> Light modeller yüklendi")
            
            ensemble_file = os.path.join(self.light_models_path, "ensemble.joblib")
            if os.path.exists(ensemble_file):
                self.ensemble_model = joblib.load(ensemble_file)
                print("  -> Ensemble model yüklendi")
            
            # Metadata
            metadata_file = os.path.join(self.models_path, "metadata.joblib")
            if os.path.exists(metadata_file):
                metadata = joblib.load(metadata_file)
                self.is_fitted = metadata.get('is_fitted', False)
                print(f"  -> Metadata yüklendi: {metadata['timestamp']}")
            
            # Heavy models loading burada yapılabilir
            # (Neural network modelleri ayrı yüklenecek)
            
            return True
            
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            return False
    
    def get_model_info(self):
        """Model bilgilerini döndür"""
        info = {}
        
        if self.is_fitted and self.light_models:
            for name in self.light_models.keys():
                info[name] = {
                    'type': 'light_model',
                    'status': 'fitted'
                }
                
        if hasattr(self.ensemble_model, 'estimators_'):
            info['ensemble'] = {
                'type': 'ensemble',
                'status': 'fitted',
                'models_count': len(self.ensemble_model.estimators_)
            }
            
        return info
