#!/usr/bin/env python3
"""
JetX Optimize Tahmin Sistemi - Eğitim ve Test Script'i

Kullanım:
python train_and_test.py --action train --window_size 5000
python train_and_test.py --action test --num_tests 100
python train_and_test.py --action benchmark
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime

from optimized_predictor import OptimizedJetXPredictor

def train_models(window_size=5000, model_types=['rf', 'gb', 'svm']):
    """Model eğitimi fonksiyonu"""
    print("🚀 JetX Model Eğitimi Başlıyor...")
    print(f"   📏 Veri penceresi: {window_size}")
    print(f"   🔧 Model türleri: {model_types}")
    print("=" * 60)
    
    # Predictor oluştur
    predictor = OptimizedJetXPredictor()
    
    # Eğitim yap
    start_time = time.time()
    success = predictor.train_new_models(
        window_size=window_size,
        model_types=model_types
    )
    training_time = time.time() - start_time
    
    if success:
        print("=" * 60)
        print("✅ EĞİTİM BAŞARILI!")
        print(f"   ⏱️  Toplam süre: {training_time/60:.1f} dakika")
        
        # Model bilgilerini göster
        model_info = predictor.get_model_info()
        if model_info:
            print("\n📊 Model Performansları:")
            for model_name, info in model_info.items():
                print(f"   {model_name}: Accuracy={info['accuracy']:.3f}, AUC={info.get('auc', 0):.3f}")
        
        return True
    else:
        print("❌ EĞİTİM BAŞARISIZ!")
        return False

def test_predictions(num_tests=50):
    """Tahmin testleri"""
    print(f"🧪 JetX Tahmin Testleri ({num_tests} test)")
    print("=" * 60)
    
    # Predictor yükle
    predictor = OptimizedJetXPredictor()
    
    if predictor.current_models is None:
        print("❌ Eğitilmiş model bulunamadı!")
        print("💡 Önce 'python train_and_test.py --action train' çalıştırın")
        return False
    
    # Test verileri al
    recent_data = predictor._load_recent_data(limit=500)
    if len(recent_data) < 200:
        print("❌ Yeterli test verisi yok!")
        return False
    
    print(f"📊 Test verisi: {len(recent_data)} değer")
    print("\n🎯 Tahmin testleri yapılıyor...")
    
    # Test döngüsü
    predictions = []
    prediction_times = []
    
    for i in range(num_tests):
        # Random starting point
        start_idx = np.random.randint(100, len(recent_data) - 100)
        test_sequence = recent_data[start_idx:start_idx + 100]
        
        # Tahmin yap
        start_time = time.time()
        result = predictor.predict_next(test_sequence)
        pred_time = time.time() - start_time
        
        if result:
            predictions.append(result)
            prediction_times.append(pred_time)
            
            if (i + 1) % 10 == 0:
                avg_time = np.mean(prediction_times[-10:]) * 1000
                print(f"   Test {i+1}/{num_tests} - Ortalama: {avg_time:.1f}ms")
    
    # Sonuçları analiz et
    if predictions:
        print("\n" + "=" * 60)
        print("📈 TEST SONUÇLARI:")
        
        # Tahmin süreleri
        avg_time = np.mean(prediction_times) * 1000
        min_time = np.min(prediction_times) * 1000
        max_time = np.max(prediction_times) * 1000
        
        print(f"\n⚡ Performans:")
        print(f"   Ortalama tahmin süresi: {avg_time:.1f}ms")
        print(f"   En hızlı tahmin: {min_time:.1f}ms")
        print(f"   En yavaş tahmin: {max_time:.1f}ms")
        print(f"   Saniyede tahmin: {1000/avg_time:.1f}")
        
        # Tahmin dağılımı
        above_threshold_count = sum(1 for p in predictions if p.get('above_threshold', False))
        below_threshold_count = sum(1 for p in predictions if p.get('above_threshold', False) == False)
        uncertain_count = sum(1 for p in predictions if p.get('above_threshold') is None)
        
        print(f"\n🎯 Tahmin Dağılımı:")
        print(f"   1.5 Üzeri: {above_threshold_count} ({above_threshold_count/len(predictions)*100:.1f}%)")
        print(f"   1.5 Altı: {below_threshold_count} ({below_threshold_count/len(predictions)*100:.1f}%)")
        print(f"   Belirsiz: {uncertain_count} ({uncertain_count/len(predictions)*100:.1f}%)")
        
        # Güven skorları
        confidences = [p.get('confidence_score', 0) for p in predictions]
        avg_confidence = np.mean(confidences)
        
        print(f"\n🎖️  Güven Skorları:")
        print(f"   Ortalama güven: {avg_confidence:.3f}")
        print(f"   Yüksek güven (>0.8): {sum(1 for c in confidences if c > 0.8)}")
        print(f"   Düşük güven (<0.5): {sum(1 for c in confidences if c < 0.5)}")
        
        return True
    else:
        print("❌ Hiç tahmin yapılamadı!")
        return False

def benchmark_system():
    """Sistem benchmark'i"""
    print("🏁 JetX Sistem Benchmark'i")
    print("=" * 60)
    
    predictor = OptimizedJetXPredictor()
    
    if predictor.current_models is None:
        print("❌ Eğitilmiş model bulunamadı!")
        return False
    
    # Model bilgileri
    print("📊 Model Bilgileri:")
    model_info = predictor.get_model_info()
    for model_name, info in model_info.items():
        print(f"   {model_name}: {info['accuracy']:.3f}")
    
    # Hız benchmark'i
    print("\n⚡ Hız Benchmark'i yapılıyor...")
    results = predictor.benchmark_prediction_speed(100)
    
    if results:
        print(f"\n🚀 Hız Sonuçları:")
        print(f"   Test sayısı: {results['num_tests']}")
        print(f"   Ortalama: {results['avg_time_ms']:.1f}ms")
        print(f"   En hızlı: {results['min_time_ms']:.1f}ms")
        print(f"   En yavaş: {results['max_time_ms']:.1f}ms")
        print(f"   Saniyede: {results['predictions_per_second']:.1f} tahmin")
        
        # Performance rating
        if results['avg_time_ms'] < 50:
            rating = "🚀 Çok Hızlı"
        elif results['avg_time_ms'] < 100:
            rating = "⚡ Hızlı"
        elif results['avg_time_ms'] < 200:
            rating = "✅ İyi"
        else:
            rating = "⚠️ Yavaş"
        
        print(f"   Rating: {rating}")
    
    # Sistem durumu
    print("\n🔧 Sistem Durumu:")
    performance_stats = predictor.get_performance_stats()
    
    if performance_stats:
        print(f"   Toplam tahmin: {performance_stats['total_predictions']}")
        print(f"   Cache hit ratio: {performance_stats.get('cache_hit_ratio', 0):.2f}")
    
    # Memory ve model durumu
    needs_retrain = predictor.should_retrain()
    print(f"   Model durumu: {'🔄 Yenileme gerekli' if needs_retrain else '✅ Güncel'}")
    
    return True

def check_data_status():
    """Veri durumu kontrolü"""
    print("📊 Veri Durumu Kontrolü")
    print("=" * 60)
    
    try:
        predictor = OptimizedJetXPredictor()
        recent_data = predictor._load_recent_data(limit=10000)
        
        print(f"📈 Toplam veri: {len(recent_data)}")
        
        if len(recent_data) > 0:
            values = np.array(recent_data)
            
            print(f"   Ortalama: {np.mean(values):.2f}")
            print(f"   Maksimum: {np.max(values):.2f}")
            print(f"   Minimum: {np.min(values):.2f}")
            print(f"   1.5+ oranı: {np.mean(values >= 1.5):.2%}")
            
            # Son 100 veri analizi
            if len(recent_data) >= 100:
                last_100 = values[-100:]
                print(f"\n📊 Son 100 veri:")
                print(f"   Ortalama: {np.mean(last_100):.2f}")
                print(f"   1.5+ oranı: {np.mean(last_100 >= 1.5):.2%}")
                print(f"   Volatilite: {np.std(last_100):.2f}")
            
            # Model eğitimi için yeterli mi?
            if len(recent_data) >= 500:
                print("\n✅ Model eğitimi için yeterli veri mevcut")
            else:
                print(f"\n⚠️  Model eğitimi için daha fazla veri gerekli (Mevcut: {len(recent_data)}, Gerekli: 500+)")
                
        else:
            print("❌ Hiç veri bulunamadı!")
            
    except Exception as e:
        print(f"❌ Veri kontrolü hatası: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="JetX Optimize Tahmin Sistemi - Eğitim ve Test")
    parser.add_argument('--action', choices=['train', 'test', 'benchmark', 'check'], 
                       required=True, help='Yapılacak işlem')
    parser.add_argument('--window_size', type=int, default=5000, 
                       help='Eğitim veri pencere boyutu (default: 5000)')
    parser.add_argument('--models', nargs='+', default=['rf', 'gb', 'svm'],
                       choices=['rf', 'gb', 'svm', 'lstm'],
                       help='Eğitilecek model türleri (default: rf gb svm)')
    parser.add_argument('--num_tests', type=int, default=50,
                       help='Test sayısı (default: 50)')
    
    args = parser.parse_args()
    
    print(f"""
🚀 JetX Optimize Tahmin Sistemi
═══════════════════════════════
Tarih: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
İşlem: {args.action.upper()}
""")
    
    success = False
    
    if args.action == 'train':
        success = train_models(args.window_size, args.models)
        
    elif args.action == 'test':
        success = test_predictions(args.num_tests)
        
    elif args.action == 'benchmark':
        success = benchmark_system()
        
    elif args.action == 'check':
        success = check_data_status()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ İşlem başarıyla tamamlandı!")
    else:
        print("❌ İşlem başarısız!")
        
    print(f"🕒 Tamamlanma zamanı: {datetime.now().strftime('%H:%M:%S')}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())