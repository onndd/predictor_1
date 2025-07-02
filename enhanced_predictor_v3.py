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
from models.high_value_specialist import HighValueSpecialist, HighValueFeatureExtractor
from data_processing.loader import load_data_from_sqlite, save_result_to_sqlite, save_prediction_to_sqlite, update_prediction_result

warnings.filterwarnings('ignore')


class UltimateJetXPredictor:
    """
    Ultimate JetX Tahmin Sistemi - Tüm Değer Aralıkları İçin Optimize Edilmiş
    
    Özellikler:
    - Genel hibrit sistem (orta değerler için)
    - LowValueSpecialist (1.5 altı değerlere özelleşmiş)
    - HighValueSpecialist (10x üzeri değerlere özelleşmiş)
    - Akıllı 3-yönlü model seçimi ve kombinasyonu
    - Gelişmiş güven skorları ve conflict resolution
    """
    
    def __init__(self, db_path="jetx_data.db", models_dir="trained_models"):
        self.db_path = db_path
        self.models_dir = models_dir
        
        # Threshold'lar
        self.low_threshold = 1.5
        self.high_threshold = 10.0
        
        # Model sistemleri
        self.model_manager = ModelManager(models_dir, db_path)
        self.current_models = None
        
        # Uzman sistemler
        self.low_value_specialist = LowValueSpecialist(threshold=self.low_threshold)
        self.high_value_specialist = HighValueSpecialist(threshold=self.high_threshold)
        
        # Training status
        self.low_value_trained = False
        self.high_value_trained = False
        
        # Performance tracking
        self.prediction_count = 0
        self.accuracy_tracker = {
            'general': [],
            'low_value': [],      # <1.5
            'medium_value': [],   # 1.5-10
            'high_value': []      # 10+
        }
        
        # Cache
        self.feature_cache = {}
        self.cache_max_size = 1500
        
        print("Ultimate JetX Tahmin Sistemi (v3) başlatılıyor...")
        self.load_models()
        
    def load_models(self):
        """Tüm model sistemlerini yükle"""
        print("Tüm model sistemleri yükleniyor...")
        
        # Ana modelleri yükle
        self.current_models = self.model_manager.load_latest_models()
        if self.current_models:
            print("✅ Ana modeller yüklendi!")
        
        # LowValueSpecialist yükle
        low_value_path = os.path.join(self.models_dir, "low_value_specialist.joblib")
        if os.path.exists(low_value_path):
            try:
                self.low_value_specialist = joblib.load(low_value_path)
                self.low_value_trained = True
                print("✅ LowValueSpecialist yüklendi!")
            except Exception as e:
                print(f"⚠️  LowValueSpecialist yüklenemedi: {e}")
                self.low_value_trained = False
        
        # HighValueSpecialist yükle
        high_value_path = os.path.join(self.models_dir, "high_value_specialist.joblib")
        if os.path.exists(high_value_path):
            try:
                self.high_value_specialist = joblib.load(high_value_path)
                self.high_value_trained = True
                print("✅ HighValueSpecialist yüklendi!")
            except Exception as e:
                print(f"⚠️  HighValueSpecialist yüklenemedi: {e}")
                self.high_value_trained = False
        
        return self.current_models is not None
    
    def train_ultimate_models(self, window_size=7000, focus_on_specialists=True):
        """
        Ultimate model eğitimi - tüm uzman sistemler
        """
        print("🚀 Ultimate model eğitimi başlıyor...")
        print(f"   📏 Veri penceresi: {window_size}")
        print(f"   🎯 Uzman sistemler: {focus_on_specialists}")
        
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
        
        if focus_on_specialists:
            # 2. LowValueSpecialist eğit
            print("\n=== DÜŞÜK DEĞER UZMANI EĞİTİLİYOR ===")
            low_success = self._train_low_value_specialist(data_package)
            if low_success:
                self.low_value_trained = True
                
                # LowValueSpecialist'i kaydet
                low_value_path = os.path.join(self.models_dir, "low_value_specialist.joblib")
                joblib.dump(self.low_value_specialist, low_value_path)
                print("✅ LowValueSpecialist kaydedildi!")
            
            # 3. HighValueSpecialist eğit
            print("\n=== YÜKSEK DEĞER UZMANI EĞİTİLİYOR ===")
            high_success = self._train_high_value_specialist(data_package)
            if high_success:
                self.high_value_trained = True
                
                # HighValueSpecialist'i kaydet
                high_value_path = os.path.join(self.models_dir, "high_value_specialist.joblib")
                joblib.dump(self.high_value_specialist, high_value_path)
                print("✅ HighValueSpecialist kaydedildi!")
        
        print("\n🎉 Ultimate sistem eğitimi tamamlandı!")
        print(f"   ✅ Ana sistem: Aktif")
        print(f"   ✅ Düşük değer uzmanı: {'Aktif' if self.low_value_trained else 'Pasif'}")
        print(f"   ✅ Yüksek değer uzmanı: {'Aktif' if self.high_value_trained else 'Pasif'}")
        
        return True
    
    def _train_low_value_specialist(self, data_package):
        """LowValueSpecialist'i eğit"""
        try:
            all_sequences = data_package['train']['sequences'] + data_package['test']['sequences']
            
            # Low value labels oluştur (next value < 1.5)
            low_value_labels = []
            
            # Veritabanından actual next values'ları al
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM jetx_results ORDER BY id DESC LIMIT ?", (len(all_sequences) + 100,))
            actual_values = [row[0] for row in cursor.fetchall()]
            actual_values.reverse()  # Doğru sıralama
            conn.close()
            
            # Labels oluştur
            for i, seq in enumerate(all_sequences):
                if i < len(actual_values) - 100:  # Safety margin
                    next_value = actual_values[i + 100]  # Next value after sequence
                    low_value_labels.append(1 if next_value < self.low_threshold else 0)
                else:
                    # Fallback
                    low_value_labels.append(0)
            
            print(f"LowValueSpecialist eğitim verisi:")
            print(f"  Sequence sayısı: {len(all_sequences)}")
            print(f"  Düşük değer örnekleri: {sum(low_value_labels)} ({sum(low_value_labels)/len(low_value_labels)*100:.1f}%)")
            
            if sum(low_value_labels) < 10:
                print("❌ Yeterli düşük değer örneği yok!")
                return False
            
            # Eğitim
            performances = self.low_value_specialist.fit(all_sequences, low_value_labels)
            
            print("LowValueSpecialist performansları:")
            for model_type, perf in performances.items():
                print(f"  {model_type}: {perf:.3f}")
            
            return len(performances) > 0
            
        except Exception as e:
            print(f"❌ LowValueSpecialist eğitim hatası: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _train_high_value_specialist(self, data_package):
        """HighValueSpecialist'i eğit"""
        try:
            all_sequences = data_package['train']['sequences'] + data_package['test']['sequences']
            
            # High value labels oluştur (next value >= 10.0)
            high_value_labels = []
            
            # Veritabanından actual next values'ları al
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM jetx_results ORDER BY id DESC LIMIT ?", (len(all_sequences) + 100,))
            actual_values = [row[0] for row in cursor.fetchall()]
            actual_values.reverse()  # Doğru sıralama
            conn.close()
            
            # Labels oluştur
            for i, seq in enumerate(all_sequences):
                if i < len(actual_values) - 100:  # Safety margin
                    next_value = actual_values[i + 100]  # Next value after sequence
                    high_value_labels.append(1 if next_value >= self.high_threshold else 0)
                else:
                    # Fallback
                    high_value_labels.append(0)
            
            print(f"HighValueSpecialist eğitim verisi:")
            print(f"  Sequence sayısı: {len(all_sequences)}")
            print(f"  Yüksek değer örnekleri: {sum(high_value_labels)} ({sum(high_value_labels)/len(high_value_labels)*100:.1f}%)")
            
            if sum(high_value_labels) < 5:
                print("❌ Yeterli yüksek değer örneği yok!")
                return False
            
            # Eğitim
            performances = self.high_value_specialist.fit(all_sequences, high_value_labels)
            
            print("HighValueSpecialist performansları:")
            for model_type, perf in performances.items():
                print(f"  {model_type}: {perf:.3f}")
            
            return len(performances) > 0
            
        except Exception as e:
            print(f"❌ HighValueSpecialist eğitim hatası: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_ultimate(self, recent_values=None, use_all_specialists=True):
        """
        Ultimate tahmin sistemi - 3 sistemin intelligent kombinasyonu
        
        Returns:
            dict: Comprehensive prediction results
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
            
            # 2. LowValueSpecialist'ten tahmin al
            low_value_prediction = None
            if use_all_specialists and self.low_value_trained:
                is_low, low_prob, low_confidence, low_details = self.low_value_specialist.predict_low_value(
                    recent_values[-100:]
                )
                
                low_value_prediction = {
                    'is_low_value': is_low,
                    'low_probability': low_prob,
                    'confidence': low_confidence,
                    'details': low_details
                }
            
            # 3. HighValueSpecialist'ten tahmin al
            high_value_prediction = None
            if use_all_specialists and self.high_value_trained:
                is_high, high_prob, high_confidence, high_details = self.high_value_specialist.predict_high_value(
                    recent_values[-100:]
                )
                
                high_value_prediction = {
                    'is_high_value': is_high,
                    'high_probability': high_prob,
                    'confidence': high_confidence,
                    'details': high_details
                }
            
            # 4. Ultimate combination - 3-way intelligent fusion
            result = self._ultimate_combination(
                general_pred_value, general_prob, general_confidence,
                low_value_prediction, high_value_prediction, recent_values
            )
            
            # 5. Performance tracking
            prediction_time = time.time() - start_time
            result['prediction_time_ms'] = prediction_time * 1000
            self.prediction_count += 1
            
            # 6. Tahmin kaydetme
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
            print(f"❌ Ultimate tahmin hatası: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _ultimate_combination(self, general_value, general_prob, general_conf, 
                             low_pred, high_pred, recent_values):
        """
        3 sistemin ultimate kombinasyonu
        """
        # Base values
        predicted_value = general_value if general_value is not None else np.mean(recent_values[-10:])
        base_prob = general_prob if general_prob is not None else 0.5
        base_confidence = general_conf if general_conf is not None else 0.0
        
        # Specialist predictions
        low_prob = low_pred['low_probability'] if low_pred else 0.0
        low_conf = low_pred['confidence'] if low_pred else 0.0
        high_prob = high_pred['high_probability'] if high_pred else 0.0
        high_conf = high_pred['confidence'] if high_pred else 0.0
        
        # Decision logic matrix
        decisions = {
            'general_high': base_prob > 0.5,  # >1.5 tahmini
            'specialist_low': low_prob > 0.5,  # <1.5 tahmini
            'specialist_high': high_prob > 0.4  # >=10.0 tahmini (lower threshold for rare events)
        }
        
        # Confidence weights
        total_confidence = base_confidence + low_conf + high_conf
        if total_confidence > 0:
            weights = {
                'general': base_confidence / total_confidence,
                'low': low_conf / total_confidence,
                'high': high_conf / total_confidence
            }
        else:
            weights = {'general': 0.5, 'low': 0.25, 'high': 0.25}
        
        # Ultimate decision making
        if decisions['specialist_high'] and high_conf > 0.5:
            # High value specialist çok güvenli - öncelik ver
            final_decision = "HIGH"  # 10x+
            final_prob = high_prob
            final_confidence = high_conf
            predicted_value = max(predicted_value, 12.0)  # Minimum 12x for high predictions
            decision_source = f"HighValueSpecialist (conf: {high_conf:.2f})"
            
        elif decisions['specialist_low'] and low_conf > base_confidence:
            # Low value specialist daha güvenli
            final_decision = "LOW"  # <1.5
            final_prob = 1 - low_prob  # Convert to "above threshold" probability
            final_confidence = low_conf
            predicted_value = min(predicted_value, 1.45)
            decision_source = f"LowValueSpecialist (conf: {low_conf:.2f})"
            
        elif not decisions['general_high'] and decisions['specialist_low']:
            # Consensus: Both predict low
            final_decision = "LOW"
            final_prob = (base_prob * weights['general']) + ((1-low_prob) * weights['low'])
            final_confidence = (base_confidence * weights['general']) + (low_conf * weights['low'])
            predicted_value = min(predicted_value, 1.45)
            decision_source = "Consensus (Low)"
            
        elif decisions['general_high'] and not decisions['specialist_low'] and not decisions['specialist_high']:
            # General system confident, specialists agree or neutral
            final_decision = "MEDIUM"  # 1.5-10x
            final_prob = base_prob
            final_confidence = base_confidence
            decision_source = "General System"
            
        else:
            # Complex conflict or uncertain situation
            # Use weighted combination
            if decisions['specialist_high']:
                # High value specialist says yes, combine carefully
                combined_prob = (base_prob * weights['general'] + 
                               (1-low_prob) * weights['low'] + 
                               high_prob * weights['high'])
                final_decision = "HIGH" if combined_prob > 0.6 else "MEDIUM"
            else:
                # Standard combination
                combined_prob = (base_prob * weights['general'] + 
                               (1-low_prob) * weights['low'])
                final_decision = "LOW" if combined_prob < 0.4 else "MEDIUM"
            
            final_prob = combined_prob
            final_confidence = (base_confidence * weights['general'] + 
                              low_conf * weights['low'] + 
                              high_conf * weights['high'])
            decision_source = "Weighted Combination"
        
        # Map to threshold-based decision
        if final_decision == "LOW":
            above_threshold = False
            decision_text = "1.5 ALTINDA"
        elif final_decision == "HIGH":
            above_threshold = True
            decision_text = "10X ÜZERİNDE"
        else:  # MEDIUM
            above_threshold = final_prob > 0.55
            decision_text = "1.5 ÜZERİNDE" if above_threshold else "1.5 ALTINDA"
        
        # Final confidence check
        if final_confidence < 0.4:
            above_threshold = None
            decision_text = "Belirsiz (Düşük Güven)"
        
        result = {
            'predicted_value': predicted_value,
            'above_threshold': above_threshold,
            'above_threshold_probability': final_prob,
            'confidence_score': final_confidence,
            'decision_text': decision_text,
            'decision_source': decision_source,
            'category_prediction': final_decision,
            'low_value_prediction': low_pred,
            'high_value_prediction': high_pred,
            'general_prediction': {
                'value': general_value,
                'probability': general_prob,
                'confidence': general_conf
            },
            'decision_weights': weights
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
                # Enhanced performance tracking
                self._update_ultimate_performance_tracking(prediction_id, actual_value)
                
            return success
            
        except Exception as e:
            print(f"Sonuç güncelleme hatası: {e}")
            return False
    
    def _update_ultimate_performance_tracking(self, prediction_id, actual_value):
        """Enhanced performance tracking with categories"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
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
                actual_above_threshold = 1 if actual_value >= self.low_threshold else 0
                
                # Prediction accuracy
                if above_threshold_pred == -1:  # Belirsiz tahmin
                    accuracy = 0.5
                else:
                    accuracy = 1.0 if above_threshold_pred == actual_above_threshold else 0.0
                
                # Category-based tracking
                if actual_value < self.low_threshold:
                    self.accuracy_tracker['low_value'].append(accuracy)
                elif actual_value >= self.high_threshold:
                    self.accuracy_tracker['high_value'].append(accuracy)
                else:
                    self.accuracy_tracker['medium_value'].append(accuracy)
                    
                self.accuracy_tracker['general'].append(accuracy)
                
                # Limit tracking size
                for category in self.accuracy_tracker:
                    if len(self.accuracy_tracker[category]) > 100:
                        self.accuracy_tracker[category] = self.accuracy_tracker[category][-100:]
                        
        except Exception as e:
            print(f"Performance tracking error: {e}")
    
    def get_ultimate_performance_stats(self):
        """Ultimate performans istatistikleri"""
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
            'high_value_specialist_trained': self.high_value_trained,
            'total_predictions': self.prediction_count
        }
        
        return stats
    
    def get_ultimate_insights(self, recent_values=None):
        """Ultimate analiz - tüm kategoriler"""
        if recent_values is None:
            recent_values = self._load_recent_data(limit=100)
            
        if len(recent_values) < 20:
            return {"error": "Yeterli veri yok"}
        
        insights = {}
        
        # Low value insights
        if self.low_value_trained:
            insights['low_value'] = self.low_value_specialist.get_low_value_insights(recent_values)
        
        # High value insights
        if self.high_value_trained:
            insights['high_value'] = self.high_value_specialist.get_high_value_insights(recent_values)
        
        # General insights
        insights['general'] = {
            'recent_avg': np.mean(recent_values[-10:]),
            'volatility': np.std(recent_values[-20:]),
            'trend': np.mean(recent_values[-5:]) - np.mean(recent_values[-15:-5]),
            'max_recent': np.max(recent_values[-20:]),
            'min_recent': np.min(recent_values[-20:])
        }
        
        return insights


# Backward compatibility wrapper
class EnhancedJetXPredictor(UltimateJetXPredictor):
    """Backward compatibility wrapper"""
    
    def predict_next_enhanced(self, recent_values=None):
        """Eski API ile uyumluluk"""
        return self.predict_ultimate(recent_values)
    
    def train_enhanced_models(self, window_size=6000, focus_on_low_values=True):
        """Eski API ile uyumluluk"""
        return self.train_ultimate_models(window_size=window_size, focus_on_specialists=True)