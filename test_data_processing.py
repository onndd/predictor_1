#!/usr/bin/env python3
"""
Veri işleme sistemini test etmek için kapsamlı test scripti
"""

import os
import sys
import numpy as np
import sqlite3
import tempfile
import traceback

# Proje dizinini sys.path'e ekle
sys.path.insert(0, 'src')

def test_data_loader():
    """SQLite veri yükleme/kaydetme fonksiyonlarını test et"""
    print("🧪 SQLite Veri Yükleme/Kaydetme Testleri")
    print("=" * 50)
    
    try:
        from data_processing.loader import (
            load_data_from_sqlite, 
            save_result_to_sqlite,
            save_prediction_to_sqlite,
            update_prediction_result,
            DataFrameAlternative
        )
        
        # Geçici veritabanı dosyası oluştur
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            test_db = tmp_file.name
        
        # Test verileri
        test_values = [1.2, 1.5, 2.3, 1.1, 3.4, 1.0, 2.1, 1.8, 4.2, 1.3]
        
        print("✅ Modüller başarıyla import edildi")
        
        # Veri kaydetme testi
        print("\n📝 Veri Kaydetme Testi:")
        for i, value in enumerate(test_values):
            record_id = save_result_to_sqlite(value, test_db)
            print(f"  - Değer {value} kaydedildi (ID: {record_id})")
        
        # Veri yükleme testi
        print("\n📖 Veri Yükleme Testi:")
        df = load_data_from_sqlite(test_db)
        print(f"  - Toplam {len(df)} kayıt yüklendi")
        
        if len(df) > 0:
            print(f"  - Son 3 kayıt: {df.tail(3).data}")
            
            # DataFrameAlternative test
            print(f"  - Columns: {df.columns}")
            print(f"  - İlk değer: {df['value'][0]}")
            print(f"  - Empty: {df.empty}")
        
        # Prediction kaydetme testi
        print("\n🔮 Prediction Kaydetme Testi:")
        pred_data = {
            'predicted_value': 1.7,
            'confidence_score': 0.85,
            'above_threshold': True
        }
        pred_id = save_prediction_to_sqlite(pred_data, test_db)
        print(f"  - Prediction kaydedildi (ID: {pred_id})")
        
        # Prediction güncelleme testi
        update_result = update_prediction_result(pred_id, 1.9, test_db)
        print(f"  - Prediction güncellendi: {update_result}")
        
        # Temizlik
        os.unlink(test_db)
        print("\n✅ SQLite testleri başarıyla tamamlandı!")
        
    except Exception as e:
        print(f"❌ SQLite test hatası: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_data_transformer():
    """Veri dönüşüm fonksiyonlarını test et"""
    print("\n🔄 Veri Dönüşüm Testleri")
    print("=" * 50)
    
    try:
        from data_processing.transformer import (
            get_value_category,
            get_step_category,
            transform_to_categories,
            transform_to_step_categories,
            fuzzy_membership,
            transform_to_category_ngrams
        )
        
        # Test verileri
        test_values = [1.0, 1.2, 1.5, 2.3, 5.7, 15.2, 100.5, 250.0]
        
        print("✅ Transformer modülleri başarıyla import edildi")
        
        # Kategori dönüşüm testleri
        print("\n📊 Kategori Dönüşüm Testleri:")
        for value in test_values:
            val_cat = get_value_category(value)
            step_cat = get_step_category(value)
            print(f"  - {value} -> Value: {val_cat}, Step: {step_cat}")
        
        # Batch dönüşüm testleri
        print("\n📦 Batch Dönüşüm Testleri:")
        categories = transform_to_categories(test_values)
        step_categories = transform_to_step_categories(test_values)
        print(f"  - Value Categories: {categories}")
        print(f"  - Step Categories: {step_categories}")
        
        # Fuzzy membership testi
        print("\n🌟 Fuzzy Membership Testi:")
        test_memberships = [
            (1.15, 'LOW_110_115'),
            (1.48, 'LOW_145_149'),
            (2.1, 'EARLY_2X')
        ]
        
        for value, category in test_memberships:
            membership = fuzzy_membership(value, category)
            print(f"  - {value} -> {category}: {membership:.3f}")
        
        # N-gram testi
        print("\n🔗 N-gram Testi:")
        ngrams = transform_to_category_ngrams(categories, n=2)
        print(f"  - İlk 5 bigram: {ngrams[:5]}")
        
        print("\n✅ Transformer testleri başarıyla tamamlandı!")
        
    except Exception as e:
        print(f"❌ Transformer test hatası: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_data_splitter():
    """Veri bölme fonksiyonlarını test et"""
    print("\n✂️ Veri Bölme Testleri")
    print("=" * 50)
    
    try:
        from data_processing.splitter import (
            create_sequences,
            create_above_threshold_target,
            split_recent,
            time_series_split
        )
        
        # Test verileri
        test_data = np.random.uniform(1.0, 5.0, 100)
        
        print("✅ Splitter modülleri başarıyla import edildi")
        
        # Sequence oluşturma testi
        print("\n📊 Sequence Oluşturma Testi:")
        X, y = create_sequences(test_data, seq_length=10)
        print(f"  - X shape: {X.shape}")
        print(f"  - y shape: {y.shape}")
        print(f"  - İlk sequence: {X[0]}")
        print(f"  - İlk target: {y[0]}")
        
        # Threshold target testi
        print("\n🎯 Threshold Target Testi:")
        binary_targets = create_above_threshold_target(test_data, threshold=1.5)
        above_count = np.sum(binary_targets)
        print(f"  - Toplam {len(binary_targets)} target")
        print(f"  - Eşik üstü: {above_count} ({above_count/len(binary_targets)*100:.1f}%)")
        
        # Recent split testi
        print("\n📅 Recent Split Testi:")
        older, recent = split_recent(test_data, recent_size=20)
        print(f"  - Older data: {len(older)} kayıt")
        print(f"  - Recent data: {len(recent)} kayıt")
        
        # Time series split testi (sklearn gerektiriyor)
        print("\n⏰ Time Series Split Testi:")
        try:
            splits = time_series_split(test_data, n_splits=3)
            print(f"  - {len(splits)} split oluşturuldu")
            for i, (train_idx, test_idx) in enumerate(splits):
                print(f"    Split {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
        except ImportError:
            print("  - sklearn yok, time series split atlandı")
        
        print("\n✅ Splitter testleri başarıyla tamamlandı!")
        
    except Exception as e:
        print(f"❌ Splitter test hatası: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_data_bootstrapper():
    """Bootstrap örnekleme fonksiyonlarını test et"""
    print("\n🔄 Bootstrap Örnekleme Testleri")
    print("=" * 50)
    
    try:
        from data_processing.bootstrapper import (
            block_bootstrap,
            weighted_bootstrap,
            time_based_bootstrap
        )
        
        # Test verileri
        test_data = np.random.uniform(1.0, 5.0, 50)
        
        print("✅ Bootstrapper modülleri başarıyla import edildi")
        
        # Block bootstrap testi
        print("\n📦 Block Bootstrap Testi:")
        block_samples = block_bootstrap(test_data, block_size=5, num_samples=3)
        print(f"  - {len(block_samples)} örnek oluşturuldu")
        print(f"  - İlk örnek uzunluğu: {len(block_samples[0])}")
        print(f"  - İlk örnek: {block_samples[0][:10]}...")
        
        # Weighted bootstrap testi
        print("\n⚖️ Weighted Bootstrap Testi:")
        weighted_samples = weighted_bootstrap(test_data, block_size=5, num_samples=3)
        print(f"  - {len(weighted_samples)} ağırlıklı örnek oluşturuldu")
        print(f"  - İlk örnek uzunluğu: {len(weighted_samples[0])}")
        
        # Time-based bootstrap testi
        print("\n⏰ Time-based Bootstrap Testi:")
        time_samples = time_based_bootstrap(test_data, block_size=5, num_samples=3)
        print(f"  - {len(time_samples)} zaman-bazlı örnek oluşturuldu")
        
        print("\n✅ Bootstrapper testleri başarıyla tamamlandı!")
        
    except Exception as e:
        print(f"❌ Bootstrapper test hatası: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_integration():
    """Entegrasyon testleri - tüm bileşenleri birlikte test et"""
    print("\n🔗 Entegrasyon Testleri")
    print("=" * 50)
    
    try:
        # Geçici veritabanı
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            test_db = tmp_file.name
        
        # Örnek JetX verileri oluştur
        np.random.seed(42)
        jetx_data = []
        for _ in range(200):
            # Gerçekçi JetX dağılımı
            if np.random.random() < 0.3:  # %30 crash (< 1.5)
                value = np.random.uniform(1.0, 1.49)
            elif np.random.random() < 0.5:  # %35 düşük (1.5-3.0)
                value = np.random.uniform(1.5, 3.0)
            else:  # %35 yüksek (3.0+)
                value = np.random.exponential(2.0) + 3.0
            jetx_data.append(round(value, 2))
        
        print(f"✅ {len(jetx_data)} test verisi oluşturuldu")
        
        # Veri kaydetme
        print("\n💾 Veri Kaydetme:")
        from data_processing.loader import save_result_to_sqlite
        for value in jetx_data:
            save_result_to_sqlite(value, test_db)
        print(f"  - {len(jetx_data)} kayıt veritabanına kaydedildi")
        
        # Veri yükleme
        print("\n📖 Veri Yükleme:")
        from data_processing.loader import load_data_from_sqlite
        df = load_data_from_sqlite(test_db)
        loaded_values = [row[1] for row in df.data]  # value column
        print(f"  - {len(loaded_values)} kayıt yüklendi")
        
        # Kategori dönüşümleri
        print("\n🔄 Kategori Dönüşümleri:")
        from data_processing.transformer import transform_to_categories
        categories = transform_to_categories(loaded_values)
        unique_categories = set(categories)
        print(f"  - {len(unique_categories)} farklı kategori tespit edildi")
        print(f"  - Kategoriler: {sorted(unique_categories)}")
        
        # Sequence oluşturma
        print("\n📊 Sequence Oluşturma:")
        from data_processing.splitter import create_sequences
        X, y = create_sequences(loaded_values, seq_length=20)
        print(f"  - {len(X)} sequence oluşturuldu")
        print(f"  - Sequence shape: {X.shape}")
        
        # Bootstrap örnekleme
        print("\n🔄 Bootstrap Örnekleme:")
        from data_processing.bootstrapper import weighted_bootstrap
        bootstrap_samples = weighted_bootstrap(loaded_values, num_samples=5)
        print(f"  - {len(bootstrap_samples)} bootstrap örneği oluşturuldu")
        
        # İstatistikler
        print("\n📈 İstatistikler:")
        print(f"  - Min değer: {min(loaded_values):.2f}")
        print(f"  - Max değer: {max(loaded_values):.2f}")
        print(f"  - Ortalama: {np.mean(loaded_values):.2f}")
        print(f"  - Crash oranı (<1.5): {sum(1 for v in loaded_values if v < 1.5)/len(loaded_values)*100:.1f}%")
        
        # Temizlik
        os.unlink(test_db)
        print("\n✅ Entegrasyon testleri başarıyla tamamlandı!")
        
    except Exception as e:
        print(f"❌ Entegrasyon test hatası: {e}")
        traceback.print_exc()
        return False
    
    return True

def main():
    """Ana test fonksiyonu"""
    print("🧪 VERİ İŞLEME SİSTEMİ KAPSAMLI TEST")
    print("=" * 60)
    
    # Test sonuçları
    results = []
    
    # Test sırası
    tests = [
        ("SQLite Veri Yükleme/Kaydetme", test_data_loader),
        ("Veri Dönüşümleri", test_data_transformer), 
        ("Veri Bölme", test_data_splitter),
        ("Bootstrap Örnekleme", test_data_bootstrapper),
        ("Entegrasyon", test_integration)
    ]
    
    # Testleri çalıştır
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test hatası: {e}")
            results.append((test_name, False))
    
    # Sonuçları özet
    print(f"\n{'='*60}")
    print("🎯 TEST SONUÇLARI")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ BAŞARILI" if result else "❌ BAŞARISIZ"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📊 ÖZET:")
    print(f"  - Toplam Test: {len(results)}")
    print(f"  - Başarılı: {passed}")
    print(f"  - Başarısız: {failed}")
    print(f"  - Başarı Oranı: {passed/len(results)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 TÜM TESTLER BAŞARILI!")
        print("   Veri işleme sistemi tamamen çalışıyor!")
    else:
        print(f"\n⚠️  {failed} test başarısız!")
        print("   Lütfen hataları düzeltin.")

if __name__ == "__main__":
    main()
