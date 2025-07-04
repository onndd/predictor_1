import os
import pickle
import statistics
import math
import sqlite3
from datetime import datetime
import json
import pickle
# Conditional sklearn imports
try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    print("UYARI: Sklearn yüklü değil, temel işlevsellik sınırlı olacak")
    
    # Dummy implementations
    class TimeSeriesSplit:
        def __init__(self, n_splits=5):
            self.n_splits = n_splits
        def split(self, X):
            n = len(X)
            fold_size = n // self.n_splits
            for i in range(self.n_splits):
                start = i * fold_size
                end = start + fold_size
                train_idx = list(range(0, start)) + list(range(end, n))
                test_idx = list(range(start, end))
                yield train_idx, test_idx
    
    def accuracy_score(y_true, y_pred):
        return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)
    
    def classification_report(y_true, y_pred, output_dict=False):
        return {} if output_dict else ""
    
    def roc_auc_score(y_true, y_score):
        return 0.5
    
    HAS_SKLEARN = False
import warnings

warnings.filterwarnings('ignore')

class ModelManager:
    """
    Gelişmiş Model Yönetim Sistemi
    - Model eğitimi ve kaydetme
    - Hızlı model yükleme
    - Performance tracking
    - Automated retraining
    """
    
    def __init__(self, models_dir="trained_models", db_path="jetx_data.db"):
        self.models_dir = models_dir
        self.db_path = db_path
        self.create_directories()
        
        # Model registry
        self.trained_models = {}
        self.model_performances = {}
        self.model_metadata = {}
        
    def create_directories(self):
        """Gerekli klasörleri oluştur"""
        directories = [
            self.models_dir,
            os.path.join(self.models_dir, "ensemble"),
            os.path.join(self.models_dir, "individual"),
            os.path.join(self.models_dir, "metadata"),
            os.path.join(self.models_dir, "backup")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def prepare_training_data(self, window_size=5000, sequence_length=200, test_ratio=0.2):
        """
        Eğitim verilerini optimize edilmiş şekilde hazırla
        """
        print(f"Eğitim verisi hazırlanıyor... (pencere: {window_size}, sequence: {sequence_length})")
        
        # Veritabanından veri yükle
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"SELECT value FROM jetx_results ORDER BY id DESC LIMIT {window_size}"
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()
            
            if len(rows) < sequence_length * 3:
                print(f"HATA: Yeterli veri yok. Mevcut: {len(rows)}, Gerekli: {sequence_length * 3}")
                return None
                
            values = [row[0] for row in rows][::-1]  # Doğru sıralama
            
        except Exception as e:
            print(f"Veri yükleme hatası: {e}")
            return None
        
        # Sequences ve labels oluştur
        sequences = []
        labels = []
        
        for i in range(len(values) - sequence_length):
            seq = values[i:i + sequence_length]
            next_val = values[i + sequence_length]
            
            sequences.append(seq.tolist())
            labels.append(1 if next_val >= 1.5 else 0)
        
        # Train/test split (temporal)
        split_idx = int(len(sequences) * (1 - test_ratio))
        
        train_data = {
            'sequences': sequences[:split_idx],
            'labels': labels[:split_idx]
        }
        
        test_data = {
            'sequences': sequences[split_idx:],
            'labels': labels[split_idx:]
        }
        
        print(f"Veri hazırlandı - Train: {len(train_data['sequences'])}, Test: {len(test_data['sequences'])}")
        
        return {
            'train': train_data,
            'test': test_data,
            'metadata': {
                'total_samples': len(sequences),
                'sequence_length': sequence_length,
                'window_size': window_size,
                'created_at': datetime.now().isoformat()
            }
        }
    
    def train_optimized_ensemble(self, data_package, model_types=['rf', 'gb', 'svm', 'lstm']):
        """
        Optimize edilmiş ensemble model eğitimi
        """
        print("Optimize edilmiş ensemble eğitimi başlıyor...")
        
        train_data = data_package['train']
        test_data = data_package['test']
        
        trained_models = {}
        performances = {}
        
        # Feature extraction için enhanced feature engineering
        feature_extractor = EnhancedFeatureExtractor()
        
        print("Features çıkarılıyor...")
        X_train = feature_extractor.extract_batch_features(train_data['sequences'])
        X_test = feature_extractor.extract_batch_features(test_data['sequences'])
        y_train = train_data['labels']
        y_test = test_data['labels']
        
        # Model training
        if 'rf' in model_types:
            print("Random Forest eğitiliyor...")
            rf_model = self._train_random_forest(X_train, y_train)
            rf_score = self._evaluate_model(rf_model, X_test, y_test)
            trained_models['random_forest'] = rf_model
            performances['random_forest'] = rf_score
        
        if 'gb' in model_types:
            print("Gradient Boosting eğitiliyor...")
            gb_model = self._train_gradient_boosting(X_train, y_train)
            gb_score = self._evaluate_model(gb_model, X_test, y_test)
            trained_models['gradient_boosting'] = gb_model
            performances['gradient_boosting'] = gb_score
        
        if 'svm' in model_types:
            print("SVM eğitiliyor...")
            svm_model = self._train_svm(X_train, y_train)
            svm_score = self._evaluate_model(svm_model, X_test, y_test)
            trained_models['svm'] = svm_model
            performances['svm'] = svm_score
        
        # Ensemble creation
        ensemble_model = self._create_weighted_ensemble(trained_models, performances)
        ensemble_score = self._evaluate_ensemble(ensemble_model, X_test, y_test)
        
        # Save models
        model_package = {
            'models': trained_models,
            'ensemble': ensemble_model,
            'feature_extractor': feature_extractor,
            'performances': performances,
            'ensemble_performance': ensemble_score,
            'metadata': {
                'trained_at': datetime.now().isoformat(),
                'data_info': data_package['metadata'],
                'model_types': model_types
            }
        }
        
        self.save_model_package(model_package)
        
        print(f"Ensemble eğitimi tamamlandı! Accuracy: {ensemble_score['accuracy']:.3f}")
        return model_package
    
    def _train_random_forest(self, X, y):
        """Optimize Random Forest"""
        if not HAS_SKLEARN:
            print("Sklearn gerekli, dummy model döndürülüyor")
            return {'model': None, 'scaler': None, 'type': 'dummy'}
            
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        rf.fit(X_scaled, y)
        
        return {
            'model': rf,
            'scaler': scaler,
            'type': 'random_forest'
        }
    
    def _train_gradient_boosting(self, X, y):
        """Optimize Gradient Boosting"""
        if not HAS_SKLEARN:
            print("Sklearn gerekli, dummy model döndürülüyor")
            return {'model': None, 'scaler': None, 'type': 'dummy'}
            
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        gb = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        )
        
        gb.fit(X_scaled, y)
        
        return {
            'model': gb,
            'scaler': scaler,
            'type': 'gradient_boosting'
        }
    
    def _train_svm(self, X, y):
        """Optimize SVM"""
        if not HAS_SKLEARN:
            print("Sklearn gerekli, dummy model döndürülüyor")
            return {'model': None, 'scaler': None, 'type': 'dummy'}
            
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        svm = SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        svm.fit(X_scaled, y)
        
        return {
            'model': svm,
            'scaler': scaler,
            'type': 'svm'
        }
    
    def _evaluate_model(self, model_package, X_test, y_test):
        """Model performansını değerlendir"""
        X_scaled = model_package['scaler'].transform(X_test)
        
        y_pred = model_package['model'].predict(X_scaled)
        y_proba = model_package['model'].predict_proba(X_scaled)[:, 1]
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def _create_weighted_ensemble(self, models, performances):
        """Performansa dayalı ağırlıklı ensemble"""
        weights = {}
        total_score = sum(perf['accuracy'] for perf in performances.values())
        
        for model_name, perf in performances.items():
            weights[model_name] = perf['accuracy'] / total_score
        
        return {
            'models': models,
            'weights': weights,
            'type': 'weighted_ensemble'
        }
    
    def _evaluate_ensemble(self, ensemble, X_test, y_test):
        """Ensemble performansını değerlendir"""
        predictions = []
        
        for model_name, model_package in ensemble['models'].items():
            X_scaled = model_package['scaler'].transform(X_test)
            proba = model_package['model'].predict_proba(X_scaled)[:, 1]
            weight = ensemble['weights'][model_name]
            predictions.append(proba * weight)
        
        ensemble_proba = [sum(pred_list) for pred_list in zip(*predictions)]
        ensemble_pred = [1 if prob > 0.5 else 0 for prob in ensemble_proba]
        
        return {
            'accuracy': accuracy_score(y_test, ensemble_pred),
            'auc': roc_auc_score(y_test, ensemble_proba),
            'classification_report': classification_report(y_test, ensemble_pred, output_dict=True)
        }
    
    def save_model_package(self, model_package, version=None):
        """Model package'ını kaydet"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        package_dir = os.path.join(self.models_dir, f"package_v{version}")
        os.makedirs(package_dir, exist_ok=True)
        
        # Individual models
        models_file = os.path.join(package_dir, "trained_models.pkl")
        with open(models_file, 'wb') as f:
            pickle.dump(model_package['models'], f)
        
        # Ensemble
        ensemble_file = os.path.join(package_dir, "ensemble.pkl")
        with open(ensemble_file, 'wb') as f:
            pickle.dump(model_package['ensemble'], f)
        
        # Feature extractor
        feature_file = os.path.join(package_dir, "feature_extractor.pkl")
        with open(feature_file, 'wb') as f:
            pickle.dump(model_package['feature_extractor'], f)
        
        # Metadata
        metadata_file = os.path.join(package_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump({
                'performances': model_package['performances'],
                'ensemble_performance': model_package['ensemble_performance'],
                'metadata': model_package['metadata']
            }, f, indent=2)
        
        # Latest symlink
        latest_link = os.path.join(self.models_dir, "latest")
        if os.path.exists(latest_link):
            os.unlink(latest_link)
        os.symlink(package_dir, latest_link)
        
        print(f"Model package kaydedildi: {package_dir}")
        return package_dir
    
    def load_latest_models(self):
        """En son modelleri yükle"""
        latest_path = os.path.join(self.models_dir, "latest")
        
        if not os.path.exists(latest_path):
            print("Henüz eğitilmiş model bulunamadı!")
            return None
        
        try:
            # Models
            models_file = os.path.join(latest_path, "trained_models.pkl")
            with open(models_file, 'rb') as f:
                models = pickle.load(f)
            
            # Ensemble
            ensemble_file = os.path.join(latest_path, "ensemble.pkl")
            with open(ensemble_file, 'rb') as f:
                ensemble = pickle.load(f)
            
            # Feature extractor
            feature_file = os.path.join(latest_path, "feature_extractor.pkl")
            with open(feature_file, 'rb') as f:
                feature_extractor = pickle.load(f)
            
            # Metadata
            metadata_file = os.path.join(latest_path, "metadata.json")
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            print(f"Modeller başarıyla yüklendi! Ensemble accuracy: {metadata['ensemble_performance']['accuracy']:.3f}")
            
            return {
                'models': models,
                'ensemble': ensemble,
                'feature_extractor': feature_extractor,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            return None
    
    def predict_with_ensemble(self, model_package, sequence):
        """Ensemble ile hızlı tahmin"""
        if model_package is None:
            return None, 0.5, 0.0
        
        try:
            # Feature extraction
            features = model_package['feature_extractor'].extract_features(sequence)
            X = [features]  # Single sample as list
            
            # Ensemble prediction
            ensemble = model_package['ensemble']
            predictions = []
            
            for model_name, model_info in ensemble['models'].items():
                X_scaled = model_info['scaler'].transform(X)
                proba = model_info['model'].predict_proba(X_scaled)[0][1]
                weight = ensemble['weights'][model_name]
                predictions.append(proba * weight)
            
            ensemble_proba = sum(predictions)
            
            # Confidence calculation
            individual_probas = []
            for model_name, model_info in ensemble['models'].items():
                X_scaled = model_info['scaler'].transform(X)
                proba = model_info['model'].predict_proba(X_scaled)[0][1]
                individual_probas.append(proba)
            
            confidence = 1.0 - statistics.stdev(individual_probas) if len(individual_probas) > 1 else 0.5  # Model agreement
            confidence = max(0.0, min(1.0, confidence))
            
            # Predicted value
            predicted_value = statistics.mean(sequence[-10:]) * (1.3 if ensemble_proba > 0.6 else 0.75)
            
            return predicted_value, ensemble_proba, confidence
            
        except Exception as e:
            print(f"Ensemble prediction error: {e}")
            return None, 0.5, 0.0


class EnhancedFeatureExtractor:
    """
    Gelişmiş özellik çıkarım sistemi
    """
    
    def __init__(self):
        self.feature_names = []
        
    def extract_features(self, sequence):
        """Tek sequence için özellik çıkar"""
        seq = list(sequence)
        features = []
        
        # Basic statistics
        seq_sorted = sorted(seq)
        n = len(seq_sorted)
        features.extend([
            statistics.mean(seq),
            statistics.stdev(seq) if len(seq) > 1 else 0.0,
            max(seq),
            min(seq),
            statistics.median(seq),
            seq_sorted[n//4] if n >= 4 else seq_sorted[0],  # Q1
            seq_sorted[3*n//4] if n >= 4 else seq_sorted[-1],  # Q3
        ])
        
        # Threshold-based features
        for threshold in [1.3, 1.5, 2.0, 3.0, 5.0]:
            above_threshold = sum(1 for val in seq if val >= threshold)
            features.append(above_threshold / len(seq))
        
        # Rolling statistics
        for window in [5, 10, 20]:
            if len(seq) >= window:
                rolling_seq = seq[-window:]
                features.extend([
                    statistics.mean(rolling_seq),
                    statistics.stdev(rolling_seq) if len(rolling_seq) > 1 else 0.0,
                    max(rolling_seq),
                    min(rolling_seq)
                ])
            else:
                features.extend([
                    statistics.mean(seq), 
                    statistics.stdev(seq) if len(seq) > 1 else 0.0, 
                    max(seq), 
                    min(seq)
                ])
        
        # Trend features
        if len(seq) > 1:
            diff = [seq[i] - seq[i-1] for i in range(1, len(seq))]
            positive_diff = sum(1 for d in diff if d > 0)
            negative_diff = sum(1 for d in diff if d < 0)
            features.extend([
                statistics.mean(diff),
                statistics.stdev(diff) if len(diff) > 1 else 0.0,
                positive_diff / len(diff),
                negative_diff / len(diff)
            ])
        else:
            features.extend([0, 0, 0.5, 0.5])
        
        # Pattern features
        consecutive_high = 0
        consecutive_low = 0
        current_high = 0
        current_low = 0
        
        for val in seq[-20:]:
            if val >= 1.5:
                current_high += 1
                current_low = 0
                consecutive_high = max(consecutive_high, current_high)
            else:
                current_low += 1
                current_high = 0
                consecutive_low = max(consecutive_low, current_low)
        
        features.extend([consecutive_high, consecutive_low])
        
        # Volatility features
        if len(seq) > 2:
            returns = [(seq[i] - seq[i-1]) / seq[i-1] for i in range(1, len(seq)) if seq[i-1] != 0]
            if returns:
                abs_returns = [abs(r) for r in returns]
                features.extend([
                    statistics.stdev(returns) if len(returns) > 1 else 0.0,
                    statistics.mean(abs_returns)
                ])
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0])
        
        return features
    
    def extract_batch_features(self, sequences):
        """Batch sequences için özellik çıkar"""
        print(f"Batch feature extraction: {len(sequences)} sequences")
        
        features_list = []
        for i, seq in enumerate(sequences):
            if i % 1000 == 0:
                print(f"  Processed: {i}/{len(sequences)}")
            features = self.extract_features(seq)
            features_list.append(features)
        
        return features_list