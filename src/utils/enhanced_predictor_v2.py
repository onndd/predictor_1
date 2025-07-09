import numpy as np
import pandas as pd
import sqlite3
import time
import warnings
from datetime import datetime
import os
import joblib

from model_manager import ModelManager
from models.low_value_specialist import LowValueSpecialist, LowValueFeatureExtractor
from data_processing.loader import load_data_from_sqlite, save_result_to_sqlite, save_prediction_to_sqlite, update_prediction_result

warnings.filterwarnings('ignore')


class EnhancedJetXPredictor:
    """
    Gelişmiş JetX Tahmin Sistemi - 1.5 Altı Değerler İçin Optimize Edilmiş
    
    Özellikler:
    - Mevcut hibrit sistem (genel tahminler için)
    - LowValueSpecialist (1.5 altı değerlere özelleşmiş)
    - Akıllı model seçimi ve kombinasyonu
    - Gelişmiş güven skorları
    """
    
    def __init__(self, db_path="jetx_data.db", models_dir="trained_models"):
        self.db_path = db_path
        self.models_dir = models_dir
        self.threshold = 1.5
        
        # Ana model sistemi
        self.model_manager = ModelManager(models_dir, db_path)
        self.current_models = None
        
        # Düşük değer uzmanı
        self.low_value_specialist = LowValueSpecialist(threshold=self.threshold)
        self.low_value_trained = False
        
        # Performance tracking
        self.prediction_count = 0
        self.accuracy_tracker = {
            'general': [],
            'low_value': [],
            'high_value': []
        }
        
        # Cache
        self.feature_cache = {}
        self.cache_max_size = 1000
        
        print("Gelişmiş JetX Tahmin Sistemi (v2) başlatılıyor...")
        self.load_models()
        
    def load_models(self):
        """Tüm modelleri yükle"""
        print("Modeller yükleniyor...")
        
        # Ana modelleri yükle
        self.current_models = self.model_manager.load_latest_models()
        
        # Düşük değer uzmanını yükle
        low_value_path = os.path.join(self.models_dir, "low_value_specialist.joblib")
        if os.path.exists(low_value_path):
            try:
                self.low_value_specialist = joblib.load(low_value_path)
                self.low_value_trained = True
                print("✅ LowValueSpecialist yüklendi!")
            except Exception as e:
                print(f"⚠️  LowValueSpecialist yüklenemedi: {e}")
                self.low_value_trained = False
        
        if self.current_models:
            print("✅ Ana modeller yüklendi!")
            return True
        else:
            print("❌ Model yüklenemedi!")
            return False
    
    def train_enhanced_models(self, window_size=6000, focus_on_low_values=True):
        """
        Gelişmiş model eğitimi - düşük değerlere odaklanarak
        """
        print("🚀 Gelişmiş model eğitimi başlıyor...")
        
        # 1. Ana modelleri eğit
        print("\n=== ANA MODELLER EĞİTİLİYOR ===")
        data_package = self.model_manager.prepare_training_data(
            window_size=window_size,
            sequence_length=100,
            test_ratio=0.2
        )
        
        if data_package is None:
            print("❌ Veri hazırlanamadı!")
            return False
        
        model_package = self.model_manager.train_optimized_ensemble(
            data_package,
            model_types=['rf', 'gb', 'svm']
        )
        
        self.current_models = model_package
        
        # 2. LowValueSpecialist eğit
        if focus_on_low_values:
            print("\n=== DÜŞÜK DEĞER UZMANI EĞİTİLİYOR ===")
            success = self._train_low_value_specialist(data_package)
            if success:
                self.low_value_trained = True
                
                # LowValueSpecialist'i kaydet
                low_value_path = os.path.join(self.models_dir, "low_value_specialist.joblib")
                joblib.dump(self.low_value_specialist, low_value_path)
                print("✅ LowValueSpecialist kaydedildi!")
        
        print("\n🎉 Gelişmiş sistem eğitimi tamamlandı!")
        return True
    
    def _train_low_value_specialist(self, data_package):
        """LowValueSpecialist'i eğit"""
        try:
            # Sequences ve labels hazırla
            all_sequences = data_package['train']['sequences'] + data_package['test']['sequences']
            all_labels = data_package['train']['labels'] + data_package['test']['labels']
            
            # Labels'ı düşük değer (1.5 altı) için yeniden düzenle
            low_value_labels = []
            for i, seq in enumerate(all_sequences):
                # Sequence'in son değerine göre label oluştur
                if len(seq) > 0:
                    next_value = seq[-1]  # Son değer next value'yu temsil ediyor
                    low_value_labels.append(1 if next_value < self.threshold else 0)
                else:
                    low_value_labels.append(0)
            
            print(f"LowValueSpecialist için hazırlanan veri:")
            print(f"  Toplam sequence: {len(all_sequences)}")
            print(f"  Düşük değer örnekleri: {sum(low_value_labels)}")
            print(f"  Yüksek değer örnekleri: {len(low_value_labels) - sum(low_value_labels)}")
            
            # Eğitim
            performances = self.low_value_specialist.fit(all_sequences, low_value_labels)
            
            print("LowValueSpecialist performansları:")
            for model_type, perf in performances.items():
                print(f"  {model_type}: {perf:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ LowValueSpecialist eğitim hatası: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_next_enhanced(self, recent_values=None, use_low_value_specialist=True):
        """
        Gelişmiş tahmin sistemi
        
        Returns:
            dict: Detaylı tahmin sonuçları
        """
        start_time = time.time()
        
        # Veri hazırlama
        if recent_values is None:
            recent_values = self._load_recent_data(limit=200)
            
        if len(recent_values) < 50:
            print(f"❌ Yeterli veri yok! Mevcut: {len(recent_values)}, Gerekli: 50+")
            return None
        
        # Model kontrolü
        if self.current_models is None:
            print("❌ Ana modeller yüklü değil!")
            return None
        
        try:
            # 1. Ana sistemden tahmin al
            general_pred_value, general_prob, general_confidence = self.model_manager.predict_with_ensemble(
                self.current_models,
                recent_values[-100:]
            )
            
            # 2. LowValueSpecialist'ten tahmin al (eğer mevcut ise)
            low_value_prediction = None
            if use_low_value_specialist and self.low_value_trained:
                is_low, low_prob, low_confidence, low_details = self.low_value_specialist.predict_low_value(
                    recent_values[-100:]
                )
                
                low_value_prediction = {
                    'is_low_value': is_low,
                    'low_probability': low_prob,
                    'confidence': low_confidence,
                    'details': low_details
                }
            
            # 3. Tahmin kombinasyonu ve final karar
            result = self._combine_predictions(
                general_pred_value, general_prob, general_confidence,
                low_value_prediction, recent_values
            )
            
            # 4. Performance tracking
            prediction_time = time.time() - start_time
            result['prediction_time_ms'] = prediction_time * 1000
            self.prediction_count += 1
            
            # 5. Tahmin kaydetme
            prediction_data = {
                'predicted_value': result['predicted_value'],
                'confidence_score': result['confidence_score'],
                'above_threshold': -1 if result['above_threshold'] is None else (1 if result['above_threshold'] else 0)
            }
            
            try:
                prediction_id = save_prediction_to_sqlite(prediction_data, self.db_path)
                result['prediction_id'] = prediction_id
            except Exception as e:
                print(f"Tahmin kaydetme hatası: {e}")
                result['prediction_id'] = None
            
            return result
            
        except Exception as e:
            print(f"❌ Gelişmiş tahmin hatası: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _combine_predictions(self, general_value, general_prob, general_conf, low_value_pred, recent_values):
        """
        Ana sistem ve LowValueSpecialist tahminlerini kombine et
        """
        # Temel değerler
        predicted_value = general_value if general_value is not None else np.mean(recent_values[-10:])
        base_prob = general_prob if general_prob is not None else 0.5
        base_confidence = general_conf if general_conf is not None else 0.0
        
        # LowValueSpecialist sonuçları varsa kombine et
        if low_value_pred is not None:
            low_prob = low_value_pred['low_probability']
            low_conf = low_value_pred['confidence']
            
            # Conflict detection
            general_predicts_high = base_prob > 0.5  # Ana sistem >1.5 der
            specialist_predicts_low = low_prob > 0.5  # Uzman <1.5 der
            
            if general_predicts_high and specialist_predicts_low:
                # Conflict durumu - güven skorlarına göre karar ver
                if low_conf > base_confidence:
                    # LowValueSpecialist daha güvenli
                    final_prob = 1 - low_prob  # Low prob'u high prob'a çevir
                    final_confidence = low_conf
                    decision_source = "LowValueSpecialist (conflict resolution)"
                else:
                    # Ana sistem daha güvenli
                    final_prob = base_prob
                    final_confidence = base_confidence
                    decision_source = "General System (conflict resolution)"
            elif not general_predicts_high and specialist_predicts_low:
                # Consensus: her ikisi de düşük değer diyor
                # LowValueSpecialist'e daha fazla ağırlık ver
                final_prob = (base_prob * 0.3) + ((1 - low_prob) * 0.7)
                final_confidence = (base_confidence * 0.3) + (low_conf * 0.7)
                decision_source = "Consensus (both predict low)"
            elif general_predicts_high and not specialist_predicts_low:
                # Consensus: her ikisi de yüksek değer diyor
                final_prob = (base_prob * 0.7) + ((1 - low_prob) * 0.3)
                final_confidence = (base_confidence * 0.7) + (low_conf * 0.3)
                decision_source = "Consensus (both predict high)"
            else:
                # Diğer durumlar - ana sistemi takip et
                final_prob = base_prob
                final_confidence = base_confidence
                decision_source = "General System (default)"
        else:
            # LowValueSpecialist yok - ana sistemi kullan
            final_prob = base_prob
            final_confidence = base_confidence
            decision_source = "General System (only)"
        
        # Final decision
        decision_threshold = 0.55  # Slightly conservative
        
        if final_confidence < 0.5:
            final_decision = None
            decision_text = "Belirsiz (Düşük Güven)"
        else:
            final_decision = final_prob >= decision_threshold
            decision_text = "1.5 ÜZERİNDE" if final_decision else "1.5 ALTINDA"
        
        # Value adjustment
        if final_decision is False:  # Düşük değer tahini
            predicted_value = min(predicted_value, 1.45)  # Cap at 1.45 for low predictions
        elif final_decision is True:  # Yüksek değer tahmini
            predicted_value = max(predicted_value, 1.55)  # Floor at 1.55 for high predictions
        
        result = {
            'predicted_value': predicted_value,
            'above_threshold': final_decision,
            'above_threshold_probability': final_prob,
            'confidence_score': final_confidence,
            'decision_text': decision_text,
            'decision_source': decision_source,
            'low_value_prediction': low_value_pred,
            'general_prediction': {
                'value': general_value,
                'probability': general_prob,
                'confidence': general_conf
            }
        }
        
        return result
    
    def _load_recent_data(self, limit=200):
        """Son verileri yükle"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"SELECT value FROM jetx_results ORDER BY id DESC LIMIT {limit}"
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return df['value'].values[::-1].tolist()
            
        except Exception as e:
            print(f"Veri yükleme hatası: {e}")
            return []
    
    def add_new_result(self, value):
        """Yeni sonuç ekle"""
        try:
            return save_result_to_sqlite({'value': value}, self.db_path)
        except Exception as e:
            print(f"Veri ekleme hatası: {e}")
            return None
    
    def update_result(self, prediction_id, actual_value):
        """Tahmin sonucunu güncelle ve performansı takip et"""
        try:
            success = update_prediction_result(prediction_id, actual_value, self.db_path)
            
            if success:
                # Performance tracking
                self._update_performance_tracking(prediction_id, actual_value)
                
            return success
            
        except Exception as e:
            print(f"Sonuç güncelleme hatası: {e}")
            return False
    
    def _update_performance_tracking(self, prediction_id, actual_value):
        """Performans takibi güncelle"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tahmin bilgilerini al
            cursor.execute("""
                SELECT predicted_value, above_threshold 
                FROM predictions 
                WHERE id = ?
            """, (prediction_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                predicted_value, above_threshold_pred = result
                
                # Actual classification
                actual_above_threshold = 1 if actual_value >= self.threshold else 0
                
                # Prediction accuracy
                if above_threshold_pred == -1:  # Belirsiz tahmin
                    accuracy = 0.5
                else:
                    accuracy = 1.0 if above_threshold_pred == actual_above_threshold else 0.0
                
                # Category-based tracking
                if actual_value < self.threshold:
                    self.accuracy_tracker['low_value'].append(accuracy)
                else:
                    self.accuracy_tracker['high_value'].append(accuracy)
                    
                self.accuracy_tracker['general'].append(accuracy)
                
                # Limitle tracking size
                for category in self.accuracy_tracker:
                    if len(self.accuracy_tracker[category]) > 100:
                        self.accuracy_tracker[category] = self.accuracy_tracker[category][-100:]
                        
        except Exception as e:
            print(f"Performance tracking error: {e}")
    
    def get_performance_stats(self):
        """Detaylı performans istatistikleri"""
        stats = {}
        
        for category, accuracies in self.accuracy_tracker.items():
            if accuracies:
                stats[category] = {
                    'accuracy': np.mean(accuracies),
                    'count': len(accuracies),
                    'recent_accuracy': np.mean(accuracies[-10:]) if len(accuracies) >= 10 else np.mean(accuracies)
                }
            else:
                stats[category] = {'accuracy': 0.0, 'count': 0, 'recent_accuracy': 0.0}
        
        # Model info
        stats['model_info'] = {
            'general_models_loaded': self.current_models is not None,
            'low_value_specialist_trained': self.low_value_trained,
            'total_predictions': self.prediction_count
        }
        
        return stats
    
    def get_low_value_insights(self, recent_values=None):
        """1.5 altı değerler hakkında detaylı analiz"""
        if not self.low_value_trained:
            return {"error": "LowValueSpecialist eğitilmemiş"}
            
        if recent_values is None:
            recent_values = self._load_recent_data(limit=100)
            
        if len(recent_values) < 20:
            return {"error": "Yeterli veri yok"}
            
        return self.low_value_specialist.get_low_value_insights(recent_values)
    
    def retrain_if_needed(self, performance_threshold=0.6):
        """Performans düşükse yeniden eğit"""
        stats = self.get_performance_stats()
        
        # Low value performance check
        low_value_accuracy = stats.get('low_value', {}).get('accuracy', 0.0)
        general_accuracy = stats.get('general', {}).get('accuracy', 0.0)
        
        should_retrain = False
        
        if low_value_accuracy < performance_threshold:
            print(f"⚠️  Düşük değer accuracy düşük: {low_value_accuracy:.3f}")
            should_retrain = True
            
        if general_accuracy < performance_threshold:
            print(f"⚠️  Genel accuracy düşük: {general_accuracy:.3f}")
            should_retrain = True
            
        if should_retrain:
            print("🔄 Otomatik yeniden eğitim başlıyor...")
            return self.train_enhanced_models()
        else:
            print("✅ Performans yeterli, yeniden eğitim gerekmiyor.")
            return True


# Backward compatibility için
class OptimizedJetXPredictor(EnhancedJetXPredictor):
    """Backward compatibility wrapper"""
    
    def predict_next(self, recent_values=None):
        """Eski API ile uyumluluk"""
        result = self.predict_next_enhanced(recent_values)
        return result if result else None
    
    def train_new_models(self, window_size=5000, model_types=['rf', 'gb', 'svm']):
        """Eski API ile uyumluluk"""
        return self.train_enhanced_models(window_size=window_size)