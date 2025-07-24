#!/usr/bin/env python3
"""
Basit Gelişmiş Özellik Testi
"""

import sys
import os
import numpy as np
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from feature_engineering.advanced_statistical_features import (
        AdvancedStatisticalFeatures,
        extract_advanced_statistical_features
    )
    print("✅ Gelişmiş özellik modülü import edildi")
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

def test_basic_features():
    """Temel özellik testleri"""
    print("\n🔬 Temel Gelişmiş Özellik Testleri")
    print("=" * 50)
    
    test_data = generate_test_data(100)
    print(f"Test verisi: {len(test_data)} sample")
    
    extractor = AdvancedStatisticalFeatures([10, 20])
    
    try:
        # Hurst exponent
        hurst = extractor.compute_hurst_exponent(np.array(test_data))
        print(f"✅ Hurst Exponent: {hurst:.4f}")
        
        # Fractal dimension
        fractal = extractor.compute_fractal_dimension(np.array(test_data))
        print(f"✅ Fractal Dimensions: {fractal}")
        
        # Entropy
        entropy = extractor.compute_entropy_features(np.array(test_data))
        print(f"✅ Entropy Features: {entropy}")
        
        # RQA
        rqa = extractor.compute_rqa_features(np.array(test_data))
        print(f"✅ RQA Features: {rqa}")
        
        # Regime change
        regime = extractor.compute_regime_change_indicators(np.array(test_data))
        print(f"✅ Regime Features: {regime}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_extraction_function():
    """Ana çıkarım fonksiyonu testi"""
    print("\n🔧 Çıkarım Fonksiyon Testi")
    print("=" * 50)
    
    test_data = generate_test_data(150)
    
    try:
        start_time = time.time()
        features = extract_advanced_statistical_features(test_data, [10, 20])
        end_time = time.time()
        
        print(f"✅ Feature matrix: {features.shape}")
        print(f"   Süre: {end_time - start_time:.2f}s")
        print(f"   NaN count: {np.isnan(features).sum()}")
        print(f"   Inf count: {np.isinf(features).sum()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Ana test"""
    print("🚀 Basit Gelişmiş Özellik Testi")
    print("=" * 40)
    
    # Test 1
    test1 = test_basic_features()
    
    # Test 2  
    test2 = test_extraction_function()
    
    print("\n📋 SONUÇLAR")
    print("=" * 40)
    print(f"Temel testler    : {'✅' if test1 else '❌'}")
    print(f"Çıkarım testleri : {'✅' if test2 else '❌'}")
    
    if test1 and test2:
        print("\n🎉 TÜM TESTLER BAŞARILI!")
        print("Gelişmiş özellikler çalışıyor!")
    else:
        print("\n❌ TESTLER BAŞARISIZ!")
    
    return test1 and test2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
