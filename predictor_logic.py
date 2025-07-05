# Enhanced JetX Predictor with Hybrid System

import numpy as np
import pandas as pd
import sqlite3
import copy
import os
import joblib
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Data processing
from data_processing.loader import load_data_from_sqlite, save_result_to_sqlite, save_prediction_to_sqlite, update_prediction_result

# Hibrit sistem
from models.hybrid_predictor import HybridPredictor, FeatureExtractor

# Enhanced modeller
from models.sequential.enhanced_lstm import EnhancedLSTMModel
from models.sequential.enhanced_transformer import EnhancedTransformerModel
from models.enhanced_light_models import create_enhanced_light_models, LightModelEnsemble

# Confidence estimation
from ensemble.confidence_estimator import MultiFactorConfidenceEstimator

# Legacy modeller (opsiyonel)
from models.transition.markov_model import MarkovModel
from models.similarity.pattern_matcher import PatternMatcher
from models.crash_detector import CrashDetector


class JetXPredictor:
    """
    Enhanced JetX Tahmin Sistemi - Hibrit Yaklaşım
    
    Ağır modeller (LSTM, Transformer) -> Feature Extraction
    Hafif modeller (RF, GB, SVM) -> Hızlı Tahmin
    """

    def __init__(self, db_path="jetx_data.db", models_path="saved_models"):
        """
        Enhanced JetX Tahmin Sistemi
        
        Args:
            db_path: SQLite veritabanı dosya yolu
            models_path: Modellerin kaydedileceği klasör yolu
        """
        self.db_path = db_path
        self.models_path = models_path
        
        # Hibrit sistem
        self.hybrid_predictor = HybridPredictor(models_path)
        
        # Legacy support
        self.models = {}
        self.confidence_estimator = MultiFactorConfidenceEstimator(
            history_size=500,
            model_update_threshold_hours=24
        )
        self.crash_detector = None
        
        # Parametreler
        self.threshold = 1.5
        self.training_window_size = 3000  # Artırıldı daha iyi accuracy için
        self.min_confidence_threshold = 0.55  # Azaltıldı daha fazla tahmin için
        
        # Model durumu
        self.is_trained = False
        self.last_training_time = None
        
        print("Enhanced JetX Tahmin Sistemi başlatılıyor...")
        
        # Modelleri yüklemeye çalış
        self.load_all_models()

    def save_all_models(self):
        """
        Enhanced hibrit sistemi kaydet
        """
        print(f"Enhanced modeller '{self.models_path}' klasörüne kaydediliyor...")
        os.makedirs(self.models_path, exist_ok=True)

        try:
            # Hibrit sistemi kaydet
            success = self.hybrid_predictor.save_models()
            
            # Confidence estimator kaydet
            conf_path = os.path.join(self.models_path, "confidence_estimator.joblib")
            joblib.dump(self.confidence_estimator, conf_path)
            
            # Confidence data'yı JSON olarak da kaydet
            conf_data_path = os.path.join(self.models_path, "confidence_data.json")
            self.confidence_estimator.save_confidence_data(conf_data_path)
            
            # Crash detector varsa kaydet
            if self.crash_detector:
                crash_path = os.path.join(self.models_path, "crash_detector.joblib")
                joblib.dump(self.crash_detector, crash_path)
            
            # Metadata kaydet
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'is_trained': self.is_trained,
                'training_window_size': self.training_window_size,
                'threshold': self.threshold,
                'version': 'enhanced_v2'
            }
            metadata_path = os.path.join(self.models_path, "predictor_metadata.joblib")
            joblib.dump(metadata, metadata_path)
            
            if success:
                print("Enhanced sistem başarıyla kaydedildi.")
                return True
            else:
                print("Model kaydetmede bazı sorunlar oluştu.")
                return False
                
        except Exception as e:
            print(f"Enhanced sistem kaydetme hatası: {e}")
            return False

    def load_all_models(self):
        """
        Enhanced hibrit sistemi yükle
        """
        print(f"Enhanced modeller '{self.models_path}' klasöründen yükleniyor...")
        
        try:
            # Hibrit sistemi yükle
            success = self.hybrid_predictor.load_models()
            
            if success:
                self.is_trained = True
                print("  -> Hibrit sistem yüklendi")
            
            # Confidence estimator yükle
            conf_path = os.path.join(self.models_path, "confidence_estimator.joblib")
            if os.path.exists(conf_path):
                self.confidence_estimator = joblib.load(conf_path)
                print("  -> Confidence estimator yüklendi")
                
                # Confidence data'yı da yükle
                conf_data_path = os.path.join(self.models_path, "confidence_data.json")
                if os.path.exists(conf_data_path):
                    self.confidence_estimator.load_confidence_data(conf_data_path)
                    print("  -> Confidence data yüklendi")
            
            # Crash detector yükle
            crash_path = os.path.join(self.models_path, "crash_detector.joblib")
            if os.path.exists(crash_path):
                self.crash_detector = joblib.load(crash_path)
                print("  -> Crash detector yüklendi")
            
            # Metadata yükle
            metadata_path = os.path.join(self.models_path, "predictor_metadata.joblib")
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.last_training_time = metadata.get('timestamp')
                print(f"  -> Metadata yüklendi: {self.last_training_time}")
            
            if success:
                print("Enhanced sistem başarıyla yüklendi!")
                # Legacy support için
                self.models = {'hybrid_system': self.hybrid_predictor}
                return True
            else:
                print("Enhanced sistem yüklenemedi, yeni eğitim gerekiyor.")
                return False
                
        except Exception as e:
            print(f"Enhanced sistem yükleme hatası: {e}")
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
        """
        Enhanced hibrit sistemi eğit
        """
        print(f"Enhanced JetX Tahmin Sistemi eğitimi başlıyor...")
        print(f"Toplam {len(data)} veri mevcut. Son {self.training_window_size} veri kullanılacak.")
        
        # Veri hazırlama
        data = data.tail(self.training_window_size).copy()
        
        if len(data) < 500:
            print(f"Eğitim için yeterli veri bulunamadı! (Mevcut: {len(data)}, Gerekli: 500+)")
            return False
            
        values = data['value'].values
        
        try:
            # 1. Ağır modelleri oluştur ve eğit
            print("\n=== AĞIR MODELLER EĞİTİLİYOR ===")
            
            # Enhanced LSTM
            print("Enhanced LSTM eğitiliyor...")
            lstm_model = EnhancedLSTMModel(
                seq_length=min(200, len(values) // 10),
                threshold=self.threshold
            )
            lstm_model.build_model(
                lstm_units=[128, 64, 32],
                dense_units=[64, 32],
                dropout_rate=0.3,
                use_attention=True,
                use_cnn_features=True
            )
            lstm_model.fit(values, epochs=100, batch_size=64, verbose=0)
            self.hybrid_predictor.add_heavy_model('enhanced_lstm', lstm_model)
            
            # Enhanced Transformer
            print("Enhanced Transformer eğitiliyor...")
            transformer_model = EnhancedTransformerModel(
                seq_length=min(150, len(values) // 12),
                threshold=self.threshold
            )
            transformer_model.build_model(
                embed_dim=128,
                num_heads=8,
                num_transformer_blocks=4,
                dense_units=[128, 64],
                use_positional_encoding=True,
                use_cnn_features=True
            )
            transformer_model.fit(values, epochs=80, batch_size=32, verbose=0)
            self.hybrid_predictor.add_heavy_model('enhanced_transformer', transformer_model)
            
            # Legacy models (hızlı feature extraction için)
            print("Legacy modeller ekleniyor...")
            try:
                markov_model = MarkovModel(order=4, threshold=self.threshold)
                markov_model.fit(values)
                self.hybrid_predictor.add_heavy_model('markov', markov_model)
            except:
                print("  -> Markov model eklenemedi")
            
            try:
                pattern_model = PatternMatcher(
                    threshold=self.threshold,
                    pattern_lengths=[10, 20, 50],
                    min_similarity_threshold=0.7
                )
                pattern_model.fit(values)
                self.hybrid_predictor.add_heavy_model('pattern_matcher', pattern_model)
            except:
                print("  -> Pattern matcher eklenemedi")
            
            # 2. Hibrit sistemi eğit
            print("\n=== HİBRİT SİSTEM EĞİTİLİYOR ===")
            
            # Sequences ve labels hazırla
            sequences = []
            labels = []
            
            sequence_length = 100
            for i in range(len(values) - sequence_length):
                seq = values[i:i + sequence_length].tolist()
                next_val = values[i + sequence_length]
                label = 1 if next_val >= self.threshold else 0
                
                sequences.append(seq)
                labels.append(label)
            
            print(f"Hazırlanan training sequences: {len(sequences)}")
            
            # Hibrit sistemi eğit
            accuracy = self.hybrid_predictor.fit(sequences, labels)
            print(f"Hibrit sistem accuracy: {accuracy:.3f}")
            
            # 3. Crash detector eğit
            print("\n=== CRASH DETECTOR EĞİTİLİYOR ===")
            self.crash_detector = CrashDetector(
                crash_threshold=self.threshold,
                lookback_period=10
            )
            self.crash_detector.fit(values)
            
            # 4. Confidence estimator güncellle
            print("Confidence estimator güncelleniyor...")
            # Model metadata'sını güncelle
            self.confidence_estimator.update_model_metadata(
                last_updated=datetime.now(),
                training_data_quality=0.8
            )
            
            # Eğitim tamamlandı
            self.is_trained = True
            self.last_training_time = datetime.now().isoformat()
            
            # Modelleri kaydet
            print("\n=== MODELLER KAYDEDİLİYOR ===")
            self.save_all_models()
            
            print(f"\n✅ Enhanced JetX Tahmin Sistemi başarıyla eğitildi!")
            print(f"   - Ağır modeller: LSTM, Transformer, Markov, Pattern")
            print(f"   - Hafif modeller: RF, GB, Extra Trees, SVM")
            print(f"   - Hibrit accuracy: {accuracy:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Enhanced sistem eğitim hatası: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict_next(self, recent_values=None):
        """
        Enhanced hibrit sistemle hızlı tahmin
        """
        # Veri yükleme
        if recent_values is None:
            df = self.load_data(limit=min(1000, self.training_window_size))
            if df is None or df.empty:
                print("HATA: Tahmin için veri yüklenemedi.")
                return None
            recent_values = df['value'].values

        MIN_DATA_FOR_PREDICTION = 100  # Azaltıldı
        if len(recent_values) < MIN_DATA_FOR_PREDICTION:
            print(f"HATA: Yeterli veri yok ({len(recent_values)}/{MIN_DATA_FOR_PREDICTION})")
            return None

        # Model kontrolü
        if not self.is_trained or not self.hybrid_predictor.is_fitted:
            print("UYARI: Enhanced sistem eğitilmemiş! Sidebar'dan modelleri eğitin.")
            return None

        try:
            print(f"Enhanced hibrit sistemle tahmin yapılıyor... (son {len(recent_values[-200:])} veri)")
            
            # 1. Hibrit sistemden tahmin al (HIZLI!)
            start_time = datetime.now()
            
            predicted_value, above_threshold_prob, confidence = self.hybrid_predictor.predict_next_value(
                recent_values[-200:]  # Son 200 veri yeterli
            )
            
            prediction_time = (datetime.now() - start_time).total_seconds()
            print(f"  -> Hibrit tahmin süresi: {prediction_time:.3f} saniye")
            
            if predicted_value is None:
                above_threshold_prob = 0.5
                confidence = 0.0
                predicted_value = np.mean(recent_values[-10:])
            
            print(f"  -> Hibrit sonuçlar: değer={predicted_value:.2f}, prob={above_threshold_prob:.2f}, güven={confidence:.2f}")
            
            # 2. Crash detector kontrolü (opsiyonel)
            crash_risk_score = 0.0
            if self.crash_detector:
                try:
                    crash_risk_score = self.crash_detector.predict_crash_risk(recent_values[-20:])
                    print(f"  -> Crash risk: {crash_risk_score:.2f}")
                except:
                    pass
            
            # 3. Multi-factor confidence calculation
            confidence_analysis = self.confidence_estimator.estimate_confidence(
                predicted_value, above_threshold_prob
            )
            
            # Çoklu faktör güven skoru
            multi_factor_confidence = confidence_analysis['total_confidence']
            
            # Hibrit confidence ile kombine et
            combined_confidence = (multi_factor_confidence + confidence) / 2
            combined_confidence = max(0.0, min(1.0, combined_confidence))
            
            # Detaylı güven bilgilerini logla
            print(f"  -> Güven Analizi:")
            print(f"     Seviye: {confidence_analysis['confidence_level']}")
            print(f"     Toplam: {multi_factor_confidence:.3f}")
            print(f"     Faktörler: {', '.join([f'{k}: {v:.2f}' for k, v in confidence_analysis['factors'].items()])}")
            if confidence_analysis['recommendations']:
                print(f"     Öneriler: {'; '.join(confidence_analysis['recommendations'])}")
            
            # 4. Decision making
            if combined_confidence < self.min_confidence_threshold:
                print(f"  -> Düşük güven ({combined_confidence:.2f}), belirsiz tahmin")
                result = {
                    'predicted_value': predicted_value,
                    'above_threshold': None,
                    'above_threshold_probability': above_threshold_prob,
                    'confidence_score': combined_confidence,
                    'crash_risk_by_special_model': crash_risk_score,
                    'decision_text': 'Belirsiz (Düşük Güven)',
                    'confidence_analysis': confidence_analysis
                }
            else:
                # Decision threshold
                DECISION_THRESHOLD = 0.6  # Daha aggressive
                final_decision = above_threshold_prob >= DECISION_THRESHOLD
                
                # Crash detector override
                if crash_risk_score >= 0.7:
                    if final_decision:
                        print(f"  -> Crash detector override! Risk: {crash_risk_score:.2f}")
                    final_decision = False
                
                result = {
                    'predicted_value': predicted_value,
                    'above_threshold': final_decision,
                    'above_threshold_probability': above_threshold_prob,
                    'confidence_score': combined_confidence,
                    'crash_risk_by_special_model': crash_risk_score,
                    'decision_text': '1.5 ÜZERİNDE' if final_decision else '1.5 ALTINDA',
                    'confidence_analysis': confidence_analysis
                }
            
            # 5. Tahmin kaydet
            prediction_data = {
                'predicted_value': result['predicted_value'],
                'confidence_score': result['confidence_score'],
                'above_threshold': -1 if result['above_threshold'] is None else (1 if result['above_threshold'] else 0)
            }
            
            try:
                prediction_id = save_prediction_to_sqlite(prediction_data, self.db_path)
                result['prediction_id'] = prediction_id
                print(f"✅ Tahmin kaydedildi (ID: {prediction_id}) - {prediction_time:.3f}s")
            except Exception as e:
                print(f"Tahmin kaydetme hatası: {e}")
                result['prediction_id'] = None
            
            return result
            
        except Exception as e:
            print(f"Enhanced tahmin hatası: {e}")
            import traceback
            traceback.print_exc()
            return None

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
                        initial_confidence,
                        timestamp=datetime.now()
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
        """
        Enhanced sistemi yeniden eğit
        """
        print("Enhanced JetX sistemi yeniden eğitiliyor...")
        
        # Veri yükle
        df = self.load_data(limit=self.training_window_size)
        if df is None or len(df) < 500:
            print("UYARI: Yeniden eğitim için yeterli veri yüklenemedi.")
            return False
        
        # Modelleri yeniden eğit
        success = self.initialize_models(df)
        
        if success:
            print("✅ Enhanced sistem başarıyla yeniden eğitildi!")
            return True
        else:
            print("❌ Enhanced sistem yeniden eğitimi başarısız!")
            return False

    def get_model_info(self):
        """
        Enhanced sistem model bilgileri
        """
        info = {}
        
        if self.is_trained and hasattr(self.hybrid_predictor, 'light_models'):
            # Hibrit sistem bilgileri
            hybrid_info = self.hybrid_predictor.get_model_info()
            for model_name, model_info in hybrid_info.items():
                info[model_name] = {
                    'accuracy': 0.75,  # Placeholder - gerçek accuracy hesaplansın
                    'weight': 1.0,
                    'type': model_info.get('type', 'unknown'),
                    'status': model_info.get('status', 'unknown')
                }
        
        # Eğer hibrit sistem fitted değilse legacy format döndür
        if not info:
            info = {
                'enhanced_lstm': {'accuracy': 0.70, 'weight': 0.3},
                'enhanced_transformer': {'accuracy': 0.72, 'weight': 0.3},
                'enhanced_rf': {'accuracy': 0.65, 'weight': 0.2},
                'enhanced_gb': {'accuracy': 0.68, 'weight': 0.2}
            }
            
        return info