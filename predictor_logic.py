# predictor_logic.py DOSYASININ TAM İÇERİĞİ

import numpy as np
import pandas as pd
import sqlite3
import copy
import os
import joblib

# Gerekli tüm modülleri import edelim
from data_processing.loader import load_data_from_sqlite, save_result_to_sqlite, save_prediction_to_sqlite, update_prediction_result
from data_processing.transformer import transform_to_categories
from data_processing.splitter import time_series_split
from sklearn.model_selection import TimeSeriesSplit

#from feature_engineering.categorical_features import extract_categorical_features
#from feature_engineering.pattern_features import encode_ngram_features
from feature_engineering.statistical_features import extract_statistical_features

from models.ml_models import RandomForestJetXPredictor
from models.transition.markov_model import MarkovModel
from models.similarity.pattern_matcher import PatternMatcher
from models.statistical.density_model import DensityModel
from models.crash_detector import CrashDetector

from ensemble.stacking import StackingEnsemble
from ensemble.confidence_estimator import ConfidenceEstimator


class JetXPredictor:
    # predictor_logic.py dosyasındaki JetXPredictor sınıfının İÇİNE

    def __init__(self, db_path="jetx_data.db", models_path="saved_models"):
        """
        JetX Tahmin Sistemi - Kaydetme/Yükleme Özellikli
        
        Args:
            db_path: SQLite veritabanı dosya yolu.
            models_path: Eğitilmiş modellerin kaydedileceği klasör yolu.
        """
        self.db_path = db_path
        self.models_path = models_path  # Kayıtlı modeller için klasör yolu
        
        self.models = {}
        self.ensemble = None
        self.confidence_estimator = None # Yüklenene kadar None
        self.crash_detector = None # Yüklenene kadar None

        self.threshold = 1.5
        self.training_window_size = 2500 
        self.min_confidence_threshold = 0.60

        # Model parametreleri (eğitim sırasında kullanılacak)
        self.markov_order = 4
        self.rf_n_estimators = 150
        self.pattern_lengths = [5, 8, 12, 15, 20, 25, 30, 40, 75, 100, 175, 250]
        self.pm_min_similarity_threshold = 0.75
        self.pm_max_similar_patterns = 13
        self.cd_lookback_period = 7
        
        print("JetX Tahmin Sistemi başlatılıyor...")
        
        # Başlangıçta hemen modelleri yüklemeye çalış
        if not self.load_all_models():
            # Eğer yükleme başarısız olursa (ilk çalıştırma gibi), 
            # nesneleri boş olarak başlat.
            print("Boş model nesneleri oluşturuluyor...")
            self.confidence_estimator = ConfidenceEstimator()
            self.crash_detector = CrashDetector(
                crash_threshold=self.threshold,
                lookback_period=self.cd_lookback_period
            )

    # predictor_logic.py dosyasındaki JetXPredictor sınıfının İÇİNE eklenecek

    def save_all_models(self):
        """
        Eğitilmiş olan tüm ana modelleri belirtilen yola kaydeder.
        """
        print(f"Modeller '{self.models_path}' klasörüne kaydediliyor...")
        os.makedirs(self.models_path, exist_ok=True) # Klasör yoksa oluştur

        try:
            # Ana ensemble modelini kaydet
            joblib.dump(self.ensemble, os.path.join(self.models_path, "ensemble.joblib"))
            
            # Crash detector modelini kaydet
            joblib.dump(self.crash_detector, os.path.join(self.models_path, "crash_detector.joblib"))
            
            # Confidence estimator'ı kaydet (geçmişiyle birlikte)
            joblib.dump(self.confidence_estimator, os.path.join(self.models_path, "confidence_estimator.joblib"))
            
            print("Tüm modeller başarıyla kaydedildi.")
            return True
        except Exception as e:
            print(f"Modelleri kaydederken hata oluştu: {e}")
            return False

    def load_all_models(self):
        """
        Kaydedilmiş modelleri diskten yükler.
        """
        print(f"Modeller '{self.models_path}' klasöründen yükleniyor...")
        
        # Tüm dosyaların var olup olmadığını kontrol et
        ensemble_path = os.path.join(self.models_path, "ensemble.joblib")
        crash_path = os.path.join(self.models_path, "crash_detector.joblib")
        conf_path = os.path.join(self.models_path, "confidence_estimator.joblib")

        if not all(os.path.exists(p) for p in [ensemble_path, crash_path, conf_path]):
            print("Kaydedilmiş model dosyaları bulunamadı. Modellerin eğitilmesi gerekiyor.")
            return False

        try:
            self.ensemble = joblib.load(ensemble_path)
            self.crash_detector = joblib.load(crash_path)
            self.confidence_estimator = joblib.load(conf_path)
            
            # Yüklenen modelleri ana model listesine de ata
            if self.ensemble and hasattr(self.ensemble, 'models'):
                self.models = self.ensemble.models

            print("Tüm modeller başarıyla yüklendi.")
            return True
        except Exception as e:
            print(f"Modelleri yüklerken hata oluştu: {e}")
            return False

    def _prepare_stacking_data(self, data, base_models):
        print("Stacking için meta-veri seti hazırlanıyor (Bu işlem biraz sürebilir)...")
        values = data['value'].values
        
        meta_features = []
        meta_labels = []
        
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)

        for i, (train_index, test_index) in enumerate(tscv.split(values)):
            print(f"  -> Stacking Fold {i+1}/{n_splits} işleniyor...")
            train_data, test_data = values[train_index], values[test_index]

            if len(train_data) < 250:
                print(f"     Fold {i+1} atlanıyor, eğitim için yeterli veri yok.")
                continue

            current_fold_models = {}
            for name, model in base_models.items():
                temp_model = copy.deepcopy(model)
                try:
                    temp_model.fit(train_data)
                    current_fold_models[name] = temp_model
                except Exception as e:
                    print(f"     HATA: Model '{name}' Fold {i+1} için eğitilemedi: {e}")

            min_history = 201 
            if len(test_data) < min_history:
                continue

            fold_meta_features = []
            
            for model_name, model_instance in sorted(current_fold_models.items()):
                model_predictions = []
                for j in range(min_history, len(test_data)):
                    sequence = test_data[j-min_history:j]
                    try:
                        _, prob, _ = model_instance.predict_next_value(sequence)
                        model_predictions.append(prob if prob is not None else 0.5)
                    except:
                        model_predictions.append(0.5)
                
                fold_meta_features.append(model_predictions)
            
            if fold_meta_features:
                stacked_features = np.array(fold_meta_features).T
                meta_features.extend(stacked_features)
                actual_labels = (test_data[min_history:] >= self.threshold).astype(int)
                meta_labels.extend(actual_labels)

        if not meta_features:
            print("UYARI: Stacking için hiç meta-özellik oluşturulamadı.")
            return np.array([]), np.array([])

        print(f"Meta-veri seti başarıyla oluşturuldu. Toplam {len(meta_features)} örnek.")
        return np.array(meta_features), np.array(meta_labels)

    def load_data(self, limit=None):
        if limit:
            print(f"Veritabanından son {limit} veri yükleniyor: {self.db_path}")
        else:
            print(f"Veritabanından veriler yükleniyor: {self.db_path}")
        
        try:
            df = load_data_from_sqlite(self.db_path, limit=limit)
            if df is not None:
                print(f"Toplam {len(df)} veri noktası yüklendi.")
            return df
        except Exception as e:
            print(f"Veri yükleme hatası: {e}")
            return None

    def initialize_models(self, data):
        print(f"Toplam {len(data)} adet veri mevcut. Eğitim için son {self.training_window_size} veri kullanılacak (Kayan Pencere).")
        data = data.tail(self.training_window_size).copy()

        if data is None or len(data) < 250:
            print(f"Eğitim için yeterli veri bulunamadı! (Mevcut: {len(data)}, Gerekli: ~250)")
            return
            
        values = data['value'].values
        print(f"Toplam {len(values)} veri noktası ile modeller eğitiliyor...")

        print("Alt modeller (base models) eğitiliyor...")
        base_models = {
            'markov': MarkovModel(order=self.markov_order, threshold=self.threshold),
            'pattern_matcher': PatternMatcher(
                threshold=self.threshold,
                pattern_lengths=self.pattern_lengths,
                min_similarity_threshold=self.pm_min_similarity_threshold,
                max_similar_patterns=self.pm_max_similar_patterns
            ),
            'density': DensityModel(threshold=self.threshold),
            'random_forest': RandomForestJetXPredictor(
                n_estimators=self.rf_n_estimators,
                random_state=42,
                threshold=self.threshold
            )
        }
        
        for name, model in base_models.items():
            try:
                print(f"  -> Model '{name}' tüm veriyle eğitiliyor...")
                model.fit(values)
            except Exception as e:
                print(f"     HATA: Model '{name}' eğitilirken hata oluştu: {e}")

        self.models = base_models
        
        if hasattr(self, 'crash_detector') and self.crash_detector is not None:
            print("CrashDetector hazırlanıyor/fit ediliyor...")
            self.crash_detector.fit(values)
            print("CrashDetector hazır.")
        
        X_meta, y_meta = self._prepare_stacking_data(data, copy.deepcopy(base_models))
        
        print("Stacking Ensemble (üst model) kuruluyor ve eğitiliyor...")
        self.ensemble = StackingEnsemble(models=self.models)
        self.ensemble.fit(X_meta, y_meta)

        print("\nTüm modeller (Stacking Ensemble dahil) başarıyla hazırlandı!")

    def predict_next(self, recent_values=None):
        if recent_values is None:
            print(f"predict_next: recent_values sağlanmadı, veritabanından son {self.training_window_size} değer yükleniyor...")
            df = self.load_data(limit=self.training_window_size)
            
            if df is None or df.empty:
                print("HATA: predict_next içinde veri yüklenemedi veya veritabanı boş. Tahmin yapılamıyor.")
                return None
            
            recent_values = df['value'].values

        MIN_DATA_FOR_PREDICTION = 250
        if len(recent_values) < MIN_DATA_FOR_PREDICTION:
            print(f"HATA: Tahmin yapmak için yeterli veri yok ({len(recent_values)}/{MIN_DATA_FOR_PREDICTION}).")
            return None

        print(f"Son {len(recent_values)} değer kullanılarak tahmin yapılıyor...")

        if not self.ensemble or not self.ensemble.is_fitted:
            print("HATA: Ensemble modeli başlatılmamış veya eğitilmemiş.")
            return None

        try:
            ensemble_value_pred, ensemble_above_prob, ensemble_confidence = self.ensemble.predict_next_value(recent_values)
            if ensemble_above_prob is None:
                ensemble_above_prob = 0.5
                ensemble_confidence = 0.0
            print(f"Ensemble Ham Tahmin Sonuçları: değer={ensemble_value_pred}, olasılık={ensemble_above_prob}, güven={ensemble_confidence}")
        except Exception as e:
            print(f"Ensemble tahmini sırasında bir hata oluştu: {e}")
            return None

        crash_risk_score = 0.0
        if hasattr(self, 'crash_detector') and self.crash_detector is not None:
            if len(recent_values) >= self.cd_lookback_period:
                crash_risk_score = self.crash_detector.predict_crash_risk(recent_values)
                print(f"CrashDetector Risk Skoru: {crash_risk_score:.2f}")

        final_confidence = self.confidence_estimator.estimate_confidence(ensemble_value_pred, ensemble_above_prob)

        if final_confidence < self.min_confidence_threshold:
            print(f"UYARI: Güven skoru ({final_confidence:.2f}) eşiğin ({self.min_confidence_threshold}) altında. Tahmin 'Belirsiz' olarak işaretlendi.")
            result = {
                'predicted_value': ensemble_value_pred,
                'above_threshold': None,
                'above_threshold_probability': ensemble_above_prob,
                'confidence_score': final_confidence,
                'crash_risk_by_special_model': crash_risk_score,
                'decision_text': 'Belirsiz (Güven Düşük)'
            }
        else:
            KARAR_ESIGI = 0.7 
            final_above_threshold_decision = ensemble_above_prob >= KARAR_ESIGI
            CRASH_DETECTOR_OVERRIDE_THRESHOLD = 0.7 
            if crash_risk_score >= CRASH_DETECTOR_OVERRIDE_THRESHOLD:
                if final_above_threshold_decision:
                    print(f"UYARI: CrashDetector YÜKSEK RİSK ({crash_risk_score:.2f}) sinyali verdi! Ensemble tahmini ({ensemble_above_prob:.2f}) geçersiz kılınıp 1.5 altı tahmin ediliyor.")
                final_above_threshold_decision = False
            result = {
                'predicted_value': ensemble_value_pred,
                'above_threshold': final_above_threshold_decision,
                'above_threshold_probability': ensemble_above_prob,
                'confidence_score': final_confidence,
                'crash_risk_by_special_model': crash_risk_score,
                'decision_text': '1.5 ÜSTÜ' if final_above_threshold_decision else '1.5 ALTI'
            }

        prediction_data = {
            'predicted_value': result['predicted_value'] if result['predicted_value'] is not None else -1.0,
            'confidence_score': result['confidence_score'] if result['confidence_score'] is not None and not np.isnan(result['confidence_score']) else 0.0,
            'above_threshold': -1 if result['above_threshold'] is None else (1 if result['above_threshold'] else 0)
        }
        try:
            prediction_id = save_prediction_to_sqlite(prediction_data, self.db_path)
            result['prediction_id'] = prediction_id
            print(f"Tahmin kaydedildi (ID: {prediction_id})")
        except Exception as e:
            print(f"Tahmin kaydetme hatası: {e}")
            result['prediction_id'] = None
            
        return result

    def update_result(self, prediction_id, actual_value):
        try:
            success = update_prediction_result(prediction_id, actual_value, self.db_path)
            if not success:
                print(f"UYARI: Tahmin ID'si {prediction_id} için sonuç güncellenemedi.")
                return False

            if self.confidence_estimator and hasattr(self.confidence_estimator, 'add_prediction'):
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT predicted_value, confidence_score FROM predictions WHERE id = ?", (prediction_id,))
                row = cursor.fetchone()
                conn.close()
                if row:
                    last_predicted_val = row[0]
                    initial_confidence = row[1] if row[1] is not None else 0.5
                    if np.isnan(initial_confidence):
                        initial_confidence = 0.5
                    self.confidence_estimator.add_prediction(
                        last_predicted_val, 
                        actual_value, 
                        initial_confidence
                    )
                else:
                    print(f"UYARI: ConfidenceEstimator için ID'si {prediction_id} olan tahmin veritabanında bulunamadı.")

            print(f"Sonuç güncellendi: Tahmin ID {prediction_id} -> Gerçekleşen Değer: {actual_value}")
            return True
            
        except Exception as e:
            print(f"Sonuç güncelleme sırasında genel hata: {e}")
            return False

    def add_new_result(self, value):
        try:
            record_id = save_result_to_sqlite(value, self.db_path)
            print(f"Yeni sonuç kaydedildi: {value} (ID: {record_id})")
            return record_id
        except Exception as e:
            print(f"Sonuç kaydetme hatası: {e}")
            return None

    def retrain_models(self):
        df = self.load_data(limit=self.training_window_size)
        if df is not None and len(df) > 0:
            self.initialize_models(df)
            return True
        print("UYARI: Yeniden eğitim için veritabanından veri yüklenemedi.")
        return False

    def get_model_info(self):
        # Stacking Ensemble'da alt modellerin ağırlığı olmadığı için bu metodun anlamı değişti.
        # Meta-modelin katsayılarını veya alt modellerin genel performansını gösterebiliriz.
        # Şimdilik boş bir sözlük döndürelim veya basit bir bilgi verelim.
        if not self.ensemble or not self.ensemble.is_fitted:
            return {}
        
        info = {}
        # Meta modelin katsayılarını (eğer LogisticRegression ise) gösterebiliriz.
        if hasattr(self.ensemble.meta_model, 'coef_'):
            sorted_model_names = sorted(self.models.keys())
            for i, name in enumerate(sorted_model_names):
                info[name] = {
                    'meta_model_coefficient': self.ensemble.meta_model.coef_[0][i]
                }
        return info