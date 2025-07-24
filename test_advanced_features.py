#!/usr/bin/env python3
"""
Gelişmiş İstatistiksel Özellikler Test Sistemi
==============================================

Bu script gelişmiş istatistiksel özelliklerin:
1. Düzgün çalışıp çalışmadığını
2. Mevcut modellerle uyumlu olup olmadığını
3. Performans etkisini
test eder.
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from feature_engineering.advanced_statistical_features import (
        AdvancedStatisticalFeatures,
        extract_advanced_statistical_features
    )
    from feature_engineering.unified_extractor import UnifiedFeatureExtractor
    print("✅ Temel modüller başarıyla import edildi")
    
    # DataLoader optional import
    try:
        from data_processing.loader import DataLoader
        DATA_LOADER_AVAILABLE = True
        print("✅ DataLoader da mevcut")
    except ImportError as e:
        DATA_LOADER_AVAILABLE = False
        print(f"⚠️ DataLoader import hatası: {e}")
        print("DataLoader olmadan test devam edecek")
        
except ImportError as e:
    print(f"❌ Kritik import hatası: {e}")
    sys.exit(1)

def generate_test_data(n_samples: int = 1000) -> List[float]:
    """Test verisi üretir"""
    np.random.seed(42)
    
    # Karmaşık finansal zaman serisi benzeri veri
    trends = np.cumsum(np.random.randn(n_samples) * 0.01)
    volatility = np.random.gamma(2, 0.5, n_samples)
    noise = np.random.randn(n_samples) * 0.1
    
    # Rejim değişiklikleri ekle
    regime_changes = [300, 600, 800]
    for change_point in regime_changes:
        if change_point < n_samples:
            trends[change_point:] += np.random.choice([-1, 1]) * 0.5
    
    # JetX benzeri değerler (1.0 - 10.0 arası)
    base_values = 1.0 + 9.0 * (1 / (1 + np.exp(-(trends + volatility + noise))))
    
    # Anomaliler ekle
    anomaly_indices = np.random.choice(n_samples, size=n_samples//50, replace=False)
    base_values[anomaly_indices] *= np.random.uniform(0.1, 5.0, len(anomaly_indices))
    
    return base_values.tolist()

def test_advanced_features_basic():
    """Temel gelişmiş özellik testleri"""
    print("\n🔬 Temel Gelişmiş Özellik Testleri")
    print("=" * 50)
    
    # Test verisi oluştur
    test_data = generate_test_data(500)
    print(f"Test verisi oluşturuldu: {len(test_data)} sample")
    
    # Advanced features sınıfını test et
    extractor = AdvancedStatisticalFeatures([10, 20, 50])
    
    try:
        # Hurst exponent testi
        start_time = time.time()
        hurst = extractor.compute_hurst_exponent(np.array(test_data))
        hurst_time = time.time() - start_time
        print(f"✅ Hurst Exponent: {hurst:.4f} (Süre: {hurst_time:.2f}s)")
        
        # Fractal dimension testi
        start_time = time.time()
        fractal_features = extractor.compute_fractal_dimension(np.array(test_data))
        fractal_time = time.time() - start_time
        print(f"✅ Fractal Dimensions: {fractal_features} (Süre: {fractal_time:.2f}s)")
        
        # RQA testi
        start_time = time.time()
        rqa_features = extractor.compute_rqa_features(np.array(test_data))
        rqa_time = time.time() - start_time
        print(f"✅ RQA Features: {list(rqa_features.keys())} (Süre: {rqa_time:.2f}s)")
        
        # Entropy testi
        start_time = time.time()
        entropy_features = extractor.compute_entropy_features(np.array(test_data))
        entropy_time = time.time() - start_time
        print(f"✅ Entropy Features: {list(entropy_features.keys())} (Süre: {entropy_time:.2f}s)")
        
        # Rejim değişiklik testi
        start_time = time.time()
        regime_features = extractor.compute_regime_change_indicators(np.array(test_data))
        regime_time = time.time() - start_time
        print(f"✅ Regime Change Features: {list(regime_features.keys())} (Süre: {regime_time:.2f}s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Temel test hatası: {e}")
        return False

def test_advanced_features_extraction():
    """Ana çıkarım fonksiyonu testi"""
    print("\n🔧 Gelişmiş Özellik Çıkarım Testi")
    print("=" * 50)
    
    test_data = generate_test_data(200)
    
    try:
        start_time = time.time()
        feature_matrix = extract_advanced_statistical_features(
            test_data, 
            window_sizes=[10, 20, 50]
        )
        extraction_time = time.time() - start_time
        
        print(f"✅ Feature matrix oluşturuldu: {feature_matrix.shape}")
        print(f"   - {feature_matrix.shape[0]} sample")
        print(f"   - {feature_matrix.shape[1]} özellik")
        print(f"   - Süre: {extraction_time:.2f}s")
        
        # NaN/Inf kontrolü
        nan_count = np.isnan(feature_matrix).sum()
        inf_count = np.isinf(feature_matrix).sum()
        print(f"   - NaN sayısı: {nan_count}")
        print(f"   - Inf sayısı: {inf_count}")
        
        if nan_count == 0 and inf_count == 0:
            print("✅ Temiz feature matrix (NaN/Inf yok)")
        else:
            print("⚠️ Feature matrix'te NaN/Inf değerler var")
            
        return True
        
    except Exception as e:
        print(f"❌ Çıkarım test hatası: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_unified_extractor_integration():
    """Unified extractor entegrasyonu testi"""
    print("\n🔗 Unified Extractor Entegrasyon Testi")
    print("=" * 50)
    
    test_data = generate_test_data(300)
    
    try:
        # Unified extractor oluştur
        extractor = UnifiedFeatureExtractor(
            feature_windows=[5, 10, 20],
            lag_windows=[5, 10],
            lags=[1, 2, 3],
            model_sequence_length=10
        )
        
        print("Unified extractor oluşturuldu")
        
        # Fit
        start_time = time.time()
        extractor.fit(test_data)
        fit_time = time.time() - start_time
        print(f"✅ Fit tamamlandı (Süre: {fit_time:.2f}s)")
        
        # Transform
        start_time = time.time()
        feature_matrix = extractor.transform(test_data)
        transform_time = time.time() - start_time
        
        print(f"✅ Transform tamamlandı:")
        print(f"   - Shape: {feature_matrix.shape}")
        print(f"   - Süre: {transform_time:.2f}s")
        
        # Feature names kontrol
        feature_names = extractor.get_feature_names()
        print(f"   - Total feature sayısı: {len(feature_names)}")
        
        # Gelişmiş özellik sayısını kontrol et
        advanced_feature_count = sum(1 for name in feature_names if any(
            keyword in name for keyword in [
                'hurst_', 'higuchi_', 'katz_', 'petrosian_',
                'shannon_', 'approximate_', 'sample_', 'permutation_', 'spectral_',
                'rqa_', 'cusum_', 'variance_ratio', 'trend_change', 'recent_change'
            ]
        ))
        print(f"   - Gelişmiş özellik sayısı: {advanced_feature_count}")
        
        # NaN/Inf kontrolü
        nan_count = np.isnan(feature_matrix).sum()
        inf_count = np.isinf(feature_matrix).sum()
        print(f"   - NaN: {nan_count}, Inf: {inf_count}")
        
        if nan_count == 0 and inf_count == 0:
            print("✅ Entegrasyon başarılı!")
            return True, feature_matrix, feature_names
        else:
            print("⚠️ NaN/Inf değerler tespit edildi")
            return False, None, None
            
    except Exception as e:
        print(f"❌ Entegrasyon test hatası: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_performance_comparison():
    """Performans karşılaştırması"""
    print("\n⚡ Performans Karşılaştırması")
    print("=" * 50)
    
    test_sizes = [100, 200, 500]
    
    for size in test_sizes:
        print(f"\nTest boyutu: {size} sample")
        test_data = generate_test_data(size)
        
        # Sadece temel özellikler
        try:
            from feature_engineering.statistical_features import extract_statistical_features
            
            start_time = time.time()
            basic_features = extract_statistical_features(
                test_data,
                feature_windows=[5, 10, 20],
                lag_windows=[5, 10],
                lags=[1, 2, 3]
            )
            basic_time = time.time() - start_time
            
            print(f"  📊 Temel özellikler: {basic_features.shape[1]} özellik, {basic_time:.3f}s")
            
        except Exception as e:
            print(f"  ❌ Temel özellik hatası: {e}")
            basic_time = float('inf')
        
        # Gelişmiş özellikler
        try:
            start_time = time.time()
            advanced_features = extract_advanced_statistical_features(
                test_data,
                window_sizes=[5, 10, 20]
            )
            advanced_time = time.time() - start_time
            
            print(f"  🧠 Gelişmiş özellikler: {advanced_features.shape[1]} özellik, {advanced_time:.3f}s")
            
            if basic_time != float('inf'):
                ratio = advanced_time / basic_time
                print(f"  ⚖️ Süre oranı (gelişmiş/temel): {ratio:.2f}x")
            
        except Exception as e:
            print(f"  ❌ Gelişmiş özellik hatası: {e}")

def test_real_data_compatibility():
    """Gerçek veri uyumluluğu testi"""
    print("\n📋 Gerçek Veri Uyumluluğu Testi")
    print("=" * 50)
    
    if not DATA_LOADER_AVAILABLE:
        print("⚠️ DataLoader mevcut değil, simülasyon verisi kullanılıyor")
        return test_unified_extractor_integration()[0]
    
    try:
        # Mevcut data loader ile test
        data_loader = DataLoader()
        
        # Cache'den veri yükle (varsa)
        cache_file = "data/cache/jetx_data_full_data.pkl"
        if os.path.exists(cache_file):
            print("Cache dosyası bulundu, yükleniyor...")
            import pickle
            with open(cache_file, 'rb') as f:
                real_data = pickle.load(f)
            
            if isinstance(real_data, list) and len(real_data) > 100:
                test_data = real_data[:500]  # İlk 500 sample al
                print(f"Gerçek veri yüklendi: {len(test_data)} sample")
                
                # Unified extractor ile test
                extractor = UnifiedFeatureExtractor(
                    feature_windows=[10, 20, 50],
                    lag_windows=[10, 20],
                    lags=[1, 2, 3, 5],
                    model_sequence_length=20
                )
                
                start_time = time.time()
                extractor.fit(test_data)
                feature_matrix = extractor.transform(test_data)
                total_time = time.time() - start_time
                
                print(f"✅ Gerçek veri işleme başarılı:")
                print(f"   - Feature matrix: {feature_matrix.shape}")
                print(f"   - Toplam süre: {total_time:.2f}s")
                print(f"   - Sample başına süre: {total_time/len(test_data)*1000:.2f}ms")
                
                return True
                
        print("⚠️ Cache dosyası bulunamadı, simülasyon verisi kullanılıyor")
        return test_unified_extractor_integration()[0]
        
    except Exception as e:
        print(f"❌ Gerçek veri test hatası: {e}")
        return False

def main():
    """Ana test fonksiyonu"""
    print("🚀 Gelişmiş İstatistiksel Özellikler Test Başlatılıyor")
    print("=" * 60)
    
    results = {}
    
    # 1. Temel testler
    results['basic'] = test_advanced_features_basic()
    
    # 2. Özellik çıkarım testi
    results['extraction'] = test_advanced_features_extraction()
    
    # 3. Entegrasyon testi
    results['integration'], feature_matrix, feature_names = test_unified_extractor_integration()
    
    # 4. Performans testi
    test_performance_comparison()
    
    # 5. Gerçek veri uyumluluğu
    results['real_data'] = test_real_data_compatibility()
    
    # Sonuçları özetle
    print("\n📋 TEST SONUÇLARI")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ BAŞARILI" if result else "❌ BAŞARISIZ"
        print(f"{test_name.upper():20} : {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 TÜM TESTLER BAŞARILI!")
        print("Gelişmiş istatistiksel özellikler sisteme başarıyla entegre edildi.")
        
        if feature_names:
            print(f"\nToplam özellik sayısı: {len(feature_names)}")
            
            # Özellik kategorilerini say
            categories = {
                'Temel İstatistik': len([n for n in feature_names if n.startswith('stat_')]),
                'Hurst Exponent': len([n for n in feature_names if 'hurst_' in n]),
                'Fractal Dimension': len([n for n in feature_names if any(x in n for x in ['higuchi_', 'katz_', 'petrosian_'])]),
                'Entropy': len([n for n in feature_names if any(x in n for x in ['shannon_', 'approximate_', 'sample_', 'permutation_', 'spectral_'])]),
                'RQA': len([n for n in feature_names if 'rqa_' in n]),
                'Rejim Değişiklik': len([n for n in feature_names if any(x in n for x in ['cusum_', 'variance_ratio', 'trend_change', 'recent_change'])]),
                'Kategorik': len([n for n in feature_names if n.startswith('cat_')]),
                'Pattern': len([n for n in feature_names if n.startswith('ngram_')]),
                'Benzerlik': len([n for n in feature_names if n.startswith('sim_')])
            }
            
            print("\nÖzellik kategorileri:")
            for category, count in categories.items():
                print(f"  {category:20}: {count:3d} özellik")
                
    else:
        print("\n❌ BAZI TESTLER BAŞARISIZ!")
        print("Lütfen hataları gözden geçirin ve düzeltin.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
