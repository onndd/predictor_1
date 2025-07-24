#!/usr/bin/env python3
"""
Unified Extractor Entegrasyon Testi
"""

import sys
import os
import numpy as np
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from feature_engineering.unified_extractor import UnifiedFeatureExtractor
    from feature_engineering.statistical_features import extract_statistical_features
    from feature_engineering.advanced_statistical_features import extract_advanced_statistical_features
    print("✅ Tüm modüller import edildi")
except ImportError as e:
    print(f"❌ Import hatası: {e}")
    sys.exit(1)

def generate_test_data(n_samples: int = 200) -> list:
    """Test verisi üretir"""
    np.random.seed(42)
    
    # JetX benzeri basit test verisi
    base = np.random.uniform(1.0, 5.0, n_samples)
    noise = np.random.normal(0, 0.1, n_samples)
    trend = np.linspace(0, 0.5, n_samples)
    
    values = base + noise + trend
    values = np.clip(values, 1.0, 10.0)  # JetX range
    
    return values.tolist()

def test_individual_features():
    """Bireysel özellik sistemlerini test et"""
    print("\n🔬 Bireysel Özellik Sistemleri Testi")
    print("=" * 50)
    
    test_data = generate_test_data(100)
    
    # Temel istatistiksel özellikler
    try:
        basic_features = extract_statistical_features(
            test_data,
            feature_windows=[5, 10],
            lag_windows=[5, 10],
            lags=[1, 2]
        )
        print(f"✅ Temel istatistikler: {basic_features.shape}")
    except Exception as e:
        print(f"❌ Temel istatistik hatası: {e}")
        return False
    
    # Gelişmiş istatistiksel özellikler
    try:
        advanced_features = extract_advanced_statistical_features(
            test_data,
            window_sizes=[5, 10]
        )
        print(f"✅ Gelişmiş istatistikler: {advanced_features.shape}")
    except Exception as e:
        print(f"❌ Gelişmiş istatistik hatası: {e}")
        return False
    
    return True

def test_unified_extractor():
    """Unified extractor'ı test et"""
    print("\n🔗 Unified Extractor Testi")
    print("=" * 50)
    
    test_data = generate_test_data(150)
    
    try:
        # Unified extractor oluştur
        extractor = UnifiedFeatureExtractor(
            feature_windows=[5, 10],
            lag_windows=[5, 10],
            lags=[1, 2],
            model_sequence_length=10
        )
        
        print("Unified extractor oluşturuldu")
        
        # Fit
        start_time = time.time()
        extractor.fit(test_data)
        fit_time = time.time() - start_time
        print(f"✅ Fit tamamlandı: {fit_time:.2f}s")
        
        # Transform
        start_time = time.time()
        feature_matrix = extractor.transform(test_data)
        transform_time = time.time() - start_time
        
        print(f"✅ Transform tamamlandı:")
        print(f"   - Shape: {feature_matrix.shape}")
        print(f"   - Transform süresi: {transform_time:.2f}s")
        
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
        
        # Özellik kategorilerini say
        categories = {
            'Temel İstatistik': len([n for n in feature_names if n.startswith('stat_')]),
            'Hurst Exponent': len([n for n in feature_names if 'hurst_' in n]),
            'Fractal': len([n for n in feature_names if any(x in n for x in ['higuchi_', 'katz_', 'petrosian_'])]),
            'Entropy': len([n for n in feature_names if any(x in n for x in ['shannon_', 'approximate_', 'sample_', 'permutation_', 'spectral_'])]),
            'RQA': len([n for n in feature_names if 'rqa_' in n]),
            'Rejim': len([n for n in feature_names if any(x in n for x in ['cusum_', 'variance_ratio', 'trend_change', 'recent_change'])]),
            'Kategorik': len([n for n in feature_names if n.startswith('cat_')]),
            'Pattern': len([n for n in feature_names if n.startswith('ngram_')]),
            'Benzerlik': len([n for n in feature_names if n.startswith('sim_')])
        }
        
        print("\n   Özellik kategorileri:")
        for category, count in categories.items():
            if count > 0:
                print(f"     {category:15}: {count:3d} özellik")
        
        if nan_count == 0 and inf_count == 0:
            print("\n✅ Unified Extractor başarıyla çalışıyor!")
            return True, feature_matrix, feature_names
        else:
            print("\n⚠️ NaN/Inf değerler var")
            return False, None, None
            
    except Exception as e:
        print(f"❌ Unified extractor hatası: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_performance():
    """Performans testi"""
    print("\n⚡ Performans Testi")
    print("=" * 50)
    
    sizes = [50, 100, 200]
    
    for size in sizes:
        print(f"\nTest boyutu: {size} sample")
        test_data = generate_test_data(size)
        
        extractor = UnifiedFeatureExtractor(
            feature_windows=[5, 10],
            lag_windows=[5, 10],
            lags=[1, 2, 3],
            model_sequence_length=10
        )
        
        try:
            start_time = time.time()
            extractor.fit(test_data)
            features = extractor.transform(test_data)
            total_time = time.time() - start_time
            
            print(f"  ✅ {features.shape[1]} özellik, {total_time:.3f}s")
            print(f"      Sample başına: {total_time/size*1000:.1f}ms")
            
        except Exception as e:
            print(f"  ❌ Hata: {e}")

def main():
    """Ana test"""
    print("🚀 Unified Extractor Entegrasyon Testi")
    print("=" * 50)
    
    # Test 1: Bireysel sistemler
    test1 = test_individual_features()
    
    # Test 2: Unified extractor
    test2, feature_matrix, feature_names = test_unified_extractor()
    
    # Test 3: Performans
    if test2:
        test_performance()
    
    print("\n📋 SONUÇLAR")
    print("=" * 50)
    print(f"Bireysel özellikler: {'✅' if test1 else '❌'}")
    print(f"Unified entegrasyon: {'✅' if test2 else '❌'}")
    
    if test1 and test2:
        print("\n🎉 ENTEGRASYON BAŞARILI!")
        print("Gelişmiş istatistiksel özellikler sisteme başarıyla entegre edildi!")
        print("\nSistem artık şunları içeriyor:")
        print("• Hurst Exponent (trend persistence)")
        print("• Fractal Dimension (complexity analysis)")
        print("• RQA (pattern recurrence)")
        print("• Entropy measures (predictability)")
        print("• Regime change indicators")
        
        if feature_names:
            total_advanced = len([n for n in feature_names if any(
                keyword in n for keyword in [
                    'hurst_', 'higuchi_', 'katz_', 'petrosian_',
                    'shannon_', 'approximate_', 'sample_', 'permutation_', 'spectral_',
                    'rqa_', 'cusum_', 'variance_ratio', 'trend_change', 'recent_change'
                ]
            )])
            print(f"\nToplam gelişmiş özellik sayısı: {total_advanced}")
        
    else:
        print("\n❌ ENTEGRASYON BAŞARISIZ!")
    
    return test1 and test2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
