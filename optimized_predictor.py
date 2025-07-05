import numpy as np
import pandas as pd
import sqlite3
import time
import warnings
from datetime import datetime

from model_manager import ModelManager
from data_processing.loader import load_data_from_sqlite, save_result_to_sqlite, save_prediction_to_sqlite, update_prediction_result

warnings.filterwarnings('ignore')

class OptimizedJetXPredictor:
    """
    Optimize edilmiş JetX Tahmin Sistemi
    - Önceden eğitilmiş modelleri hızlı yükleme
    - Millisaniye cinsinden tahmin süresi
    - Gelişmiş feature engineering
    - Akıllı model yenileme sistemi
    """
    
    def __init__(self, db_path="jetx_data.db", models_dir="trained_models"):
        self.db_path = db_path
        self.models_dir = models_dir
        self.threshold = 1.5
        
        # Model manager
        self.model_manager = ModelManager(models_dir, db_path)
        
        # Loaded models
        self.current_models = None
        self.last_model_load_time = None
        self.prediction_count = 0
        
        # Performance tracking
        self.prediction_times = []
        self.accuracy_tracker = []
        
        # Cache for faster predictions
        self.feature_cache = {}
        self.cache_max_size = 1000
        
        print("Optimize edilmiş JetX Tahmin Sistemi başlatılıyor...")
        
        # Try to load existing models
        self.load_models()
        
        # If no models exist, suggest training
        if self.current_models is None:
            print("❌ Eğitilmiş model bulunamadı!")
            print("✅ Önce modelleri eğitmek için: predictor.train_new_models() çalıştırın")
            
    def load_models(self):
        """Mevcut modelleri yükle"""
        print("Mevcut modeller yükleniyor...")
        
        start_time = time.time()
        self.current_models = self.model_manager.load_latest_models()
        load_time = time.time() - start_time
        
        if self.current_models:
            self.last_model_load_time = datetime.now()
            print(f"✅ Modeller yüklendi! ({load_time:.2f} saniye)")
            
            # Performance info
            metadata = self.current_models['metadata']
            print(f"   📊 Ensemble Accuracy: {metadata['ensemble_performance']['accuracy']:.3f}")
            print(f"   📅 Eğitim Tarihi: {metadata['metadata']['trained_at']}")
            
            return True
        else:
            print("❌ Model yüklenemedi!")
            return False
            
    def train_new_models(self, window_size=5000, model_types=['rf', 'gb', 'svm']):
        """
        Yeni modelleri eğit ve kaydet
        """
        print("🚀 Yeni model eğitimi başlıyor...")
        print(f"   📏 Veri penceresi: {window_size}")
        print(f"   🔧 Model türleri: {model_types}")
        
        # Veri hazırlama
        print("\n1️⃣ Eğitim verisi hazırlanıyor...")
        data_package = self.model_manager.prepare_training_data(
            window_size=window_size, 
            sequence_length=100, 
            test_ratio=0.2
        )
        
        if data_package is None:
            print("❌ Veri hazırlanamadı!")
            return False
        
        # Model eğitimi
        print("\n2️⃣ Modeller eğitiliyor...")
        start_time = time.time()
        
        model_package = self.model_manager.train_optimized_ensemble(
            data_package, 
            model_types=model_types
        )
        
        training_time = time.time() - start_time
        print(f"\n✅ Eğitim tamamlandı! ({training_time/60:.1f} dakika)")
        
        # Yeni modelleri yükle
        print("\n3️⃣ Yeni modeller yükleniyor...")
        success = self.load_models()
        
        if success:
            print("🎉 Sistem hazır! Artık hızlı tahminler yapabilirsiniz.")
            return True
        else:
            print("❌ Yeni modeller yüklenemedi!")
            return False
    
    def predict_next(self, recent_values=None, use_cache=True):
        """
        Optimize edilmiş hızlı tahmin
        """
        # Model kontrolü
        if self.current_models is None:
            print("❌ Model yüklü değil! Önce train_new_models() çalıştırın.")
            return None
        
        # Veri hazırlama
        if recent_values is None:
            recent_values = self._load_recent_data(limit=200)
            
        if len(recent_values) < 150:
            print(f"❌ Yeterli veri yok! Mevcut: {len(recent_values)}, Gerekli: 150+")
            return None
        
        # Hızlı tahmin
        start_time = time.time()
        
        try:
            # Cache kontrolü
            cache_key = tuple(recent_values[-100:])  # Son 100 değerle cache
            if use_cache and cache_key in self.feature_cache:
                features = self.feature_cache[cache_key]
                cache_hit = True
            else:
                features = None
                cache_hit = False
            
            # Ensemble prediction
            predicted_value, probability, confidence = self.model_manager.predict_with_ensemble(
                self.current_models, 
                recent_values[-200:]  # Son 200 değer ile daha stabil tahmin
            )
            
            prediction_time = time.time() - start_time
            self.prediction_times.append(prediction_time)
            
            # Cache güncelleme
            if not cache_hit and use_cache:
                if len(self.feature_cache) < self.cache_max_size:
                    self.feature_cache[cache_key] = features
            
            # Decision making
            decision_threshold = 0.65  # Daha conservative
            final_decision = probability >= decision_threshold
            
            # Confidence adjustment
            if confidence < 0.6:
                final_decision = None  # Belirsiz
                decision_text = "Belirsiz (Düşük Güven)"
            else:
                decision_text = "1.5 ÜZERİNDE" if final_decision else "1.5 ALTINDA"
            
            result = {
                'predicted_value': predicted_value,
                'above_threshold': final_decision,
                'above_threshold_probability': probability,
                'confidence_score': confidence,
                'decision_text': decision_text,
                'prediction_time_ms': prediction_time * 1000,
                'cache_hit': cache_hit
            }
            
            # Tahmin kaydetme
            try:
                prediction_data = {
                    'predicted_value': predicted_value,
                    'confidence_score': confidence,
                    'above_threshold': -1 if final_decision is None else (1 if final_decision else 0)
                }
                
                prediction_id = save_prediction_to_sqlite(prediction_data, self.db_path)
                result['prediction_id'] = prediction_id
                
            except Exception as e:
                print(f"Tahmin kaydetme hatası: {e}")
                result['prediction_id'] = None
            
            # Performance tracking
            self.prediction_count += 1
            
            # Performance log (her 10 tahminde bir)
            if self.prediction_count % 10 == 0:
                avg_time = np.mean(self.prediction_times[-10:]) * 1000
                print(f"📊 Ortalama tahmin süresi (son 10): {avg_time:.1f}ms")
            
            return result
            
        except Exception as e:
            print(f"❌ Tahmin hatası: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_recent_data(self, limit=200):
        """Son verileri hızlı yükle"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"SELECT value FROM jetx_results ORDER BY id DESC LIMIT {limit}"
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return df['value'].values[::-1].tolist()  # Doğru sıralama
            
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
        """Tahmin sonucunu güncelle"""
        try:
            return update_prediction_result(prediction_id, actual_value, self.db_path)
        except Exception as e:
            print(f"Sonuç güncelleme hatası: {e}")
            return False
    
    def get_performance_stats(self):
        """Performans istatistikleri"""
        if not self.prediction_times:
            return None
        
        stats = {
            'total_predictions': len(self.prediction_times),
            'avg_prediction_time_ms': np.mean(self.prediction_times) * 1000,
            'min_prediction_time_ms': np.min(self.prediction_times) * 1000,
            'max_prediction_time_ms': np.max(self.prediction_times) * 1000,
            'cache_hit_ratio': len(self.feature_cache) / max(1, self.prediction_count),
            'model_info': self.current_models['metadata'] if self.current_models else None
        }
        
        return stats
    
    def should_retrain(self, new_data_threshold=1000):
        """Model yenileme gerekip gerekmediğini kontrol et"""
        if self.current_models is None:
            return True
        
        # Son model eğitim zamanından bu yana eklenen veri sayısı
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Last training time
            last_training_time = self.current_models['metadata']['metadata']['trained_at']
            
            # Count new data
            query = """
            SELECT COUNT(*) FROM jetx_results 
            WHERE created_at > ?
            """
            cursor = conn.cursor()
            cursor.execute(query, (last_training_time,))
            new_data_count = cursor.fetchone()[0]
            conn.close()
            
            return new_data_count >= new_data_threshold
            
        except Exception as e:
            print(f"Retrain kontrolü hatası: {e}")
            return False
    
    def auto_retrain_if_needed(self, force=False):
        """Gerekiyorsa otomatik yeniden eğitim"""
        if force or self.should_retrain():
            print("🔄 Otomatik yeniden eğitim başlıyor...")
            
            success = self.train_new_models(
                window_size=6000,  # Biraz daha büyük pencere
                model_types=['rf', 'gb', 'svm']
            )
            
            if success:
                print("✅ Otomatik yeniden eğitim tamamlandı!")
                return True
            else:
                print("❌ Otomatik yeniden eğitim başarısız!")
                return False
        else:
            print("ℹ️  Yeniden eğitim gerekmiyor.")
            return True
    
    def get_model_info(self):
        """Model bilgilerini döndür"""
        if self.current_models is None:
            return {}
        
        metadata = self.current_models['metadata']
        info = {}
        
        # Individual model performances
        for model_name, perf in metadata['performances'].items():
            info[model_name] = {
                'accuracy': perf['accuracy'],
                'auc': perf['auc'],
                'weight': self.current_models['ensemble']['weights'].get(model_name, 0.0)
            }
        
        # Ensemble performance
        ensemble_perf = metadata['ensemble_performance']
        info['ensemble'] = {
            'accuracy': ensemble_perf['accuracy'],
            'auc': ensemble_perf['auc'],
            'weight': 1.0
        }
        
        return info
    
    def benchmark_prediction_speed(self, num_tests=100):
        """Tahmin hızını benchmark et"""
        print(f"Tahmin hızı benchmark'i çalışıyor... ({num_tests} test)")
        
        if self.current_models is None:
            print("❌ Model yüklü değil!")
            return None
        
        # Test verisi hazırla
        test_sequence = self._load_recent_data(limit=100)
        if len(test_sequence) < 100:
            print("❌ Yeterli test verisi yok!")
            return None
        
        times = []
        
        for i in range(num_tests):
            start_time = time.time()
            
            # Tahmin yap
            self.model_manager.predict_with_ensemble(
                self.current_models, 
                test_sequence
            )
            
            prediction_time = time.time() - start_time
            times.append(prediction_time)
            
            if i % 20 == 0:
                print(f"  Test {i+1}/{num_tests} tamamlandı")
        
        results = {
            'num_tests': num_tests,
            'avg_time_ms': np.mean(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'predictions_per_second': 1.0 / np.mean(times)
        }
        
        print(f"\n📊 Benchmark Sonuçları:")
        print(f"   ⚡ Ortalama: {results['avg_time_ms']:.1f}ms")
        print(f"   🚀 En hızlı: {results['min_time_ms']:.1f}ms") 
        print(f"   🐌 En yavaş: {results['max_time_ms']:.1f}ms")
        print(f"   📈 Saniyede: {results['predictions_per_second']:.1f} tahmin")
        
        return results