"""
Demo: 1.5 Altı Protein Değerleri İçin Gelişmiş Tahmin Sistemi

Bu script, protein değerleri 1.5'in altında olan durumlar için özelleştirilmiş
tahmin sisteminin nasıl çalıştığını ve performansını gösterir.
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Gelişmiş tahmin sistemi
from enhanced_predictor_v2 import EnhancedJetXPredictor
from models.low_value_specialist import LowValueSpecialist, LowValueFeatureExtractor


def analyze_low_value_data(db_path="jetx_data.db"):
    """
    Veritabanındaki 1.5 altı değerleri analiz et
    """
    print("=== 1.5 ALTI DEĞERLER ANALİZİ ===")
    
    conn = sqlite3.connect(db_path)
    query = "SELECT value FROM jetx_results ORDER BY id"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    values = df['value'].values
    low_values = values[values < 1.5]
    
    print(f"📊 Toplam veri sayısı: {len(values)}")
    print(f"📉 1.5 altı değer sayısı: {len(low_values)} ({len(low_values)/len(values)*100:.1f}%)")
    print(f"📈 1.5 üstü değer sayısı: {len(values) - len(low_values)} ({(len(values) - len(low_values))/len(values)*100:.1f}%)")
    
    if len(low_values) > 0:
        print(f"\n1.5 Altı Değerler İstatistikleri:")
        print(f"  Ortalama: {np.mean(low_values):.3f}")
        print(f"  Medyan: {np.median(low_values):.3f}")
        print(f"  Minimum: {np.min(low_values):.3f}")
        print(f"  Maksimum: {np.max(low_values):.3f}")
        print(f"  Std sapma: {np.std(low_values):.3f}")
        
        # Dağılım analizi
        very_low = np.sum(low_values < 1.2)
        medium_low = np.sum((low_values >= 1.2) & (low_values < 1.35))
        close_low = np.sum(low_values >= 1.35)
        
        print(f"\n1.5 Altı Değerlerin Dağılımı:")
        print(f"  Çok düşük (<1.2): {very_low} ({very_low/len(low_values)*100:.1f}%)")
        print(f"  Orta-düşük (1.2-1.35): {medium_low} ({medium_low/len(low_values)*100:.1f}%)")
        print(f"  1.5'e yakın (1.35-1.5): {close_low} ({close_low/len(low_values)*100:.1f}%)")
    
    return values, low_values


def demonstrate_enhanced_prediction():
    """
    Gelişmiş tahmin sistemini demo et
    """
    print("\n=== GELİŞMİŞ TAHMİN SİSTEMİ DEMO ===")
    
    # Enhanced predictor'ı başlat
    predictor = EnhancedJetXPredictor()
    
    # Eğer modeller yoksa eğit
    if not predictor.load_models():
        print("🔧 Modeller bulunamadı, eğitim başlatılıyor...")
        success = predictor.train_enhanced_models(window_size=4000, focus_on_low_values=True)
        if not success:
            print("❌ Model eğitimi başarısız!")
            return
    
    # Test tahminleri
    print("\n📍 Test Tahminleri:")
    
    for i in range(5):
        print(f"\n--- Tahmin {i+1} ---")
        
        # Enhanced prediction
        result = predictor.predict_next_enhanced()
        
        if result:
            print(f"🎯 Tahmin Değeri: {result['predicted_value']:.3f}")
            print(f"📊 Karar: {result['decision_text']}")
            print(f"🎚️  Olasılık: {result['above_threshold_probability']:.3f}")
            print(f"💪 Güven: {result['confidence_score']:.3f}")
            print(f"🔍 Karar Kaynağı: {result['decision_source']}")
            
            # Low value prediction detayları
            if result.get('low_value_prediction'):
                low_pred = result['low_value_prediction']
                print(f"🔻 Düşük Değer Uzmanı:")
                print(f"   - Düşük tahmin: {low_pred['is_low_value']}")
                print(f"   - Düşük olasılık: {low_pred['low_probability']:.3f}")
                print(f"   - Güven: {low_pred['confidence']:.3f}")
        else:
            print("❌ Tahmin alınamadı!")


def demonstrate_low_value_specialist():
    """
    LowValueSpecialist'in özelleştirilmiş yeteneklerini göster
    """
    print("\n=== DÜŞÜK DEĞER UZMANI DEMO ===")
    
    # Test verisi oluştur (1.5 altı değerler ağırlıklı)
    test_sequences = [
        # Düşük değer trendi
        [1.45, 1.38, 1.25, 1.15, 1.35, 1.28, 1.42, 1.33, 1.41, 1.37] * 10,
        
        # Crash sonrası düşük değerler
        [3.2, 5.1, 1.1, 1.2, 1.35, 1.28, 1.41, 1.33, 1.25, 1.39] * 10,
        
        # Karışık düşük değerler
        [1.25, 1.48, 1.33, 1.41, 1.27, 1.45, 1.32, 1.38, 1.29, 1.44] * 10,
        
        # Yüksek değerlerden düşük değere geçiş
        [2.1, 1.8, 1.6, 1.5, 1.45, 1.38, 1.33, 1.41, 1.36, 1.42] * 10,
        
        # Stabil düşük değerler
        [1.35, 1.38, 1.42, 1.36, 1.41, 1.37, 1.44, 1.39, 1.43, 1.40] * 10
    ]
    
    # LowValueSpecialist oluştur ve eğit
    specialist = LowValueSpecialist(threshold=1.5)
    
    # Labels oluştur (sequence sonundaki değer 1.5 altı mı?)
    labels = []
    for seq in test_sequences:
        # Son değer 1.5 altındaysa 1, değilse 0
        labels.append(1 if seq[-1] < 1.5 else 0)
    
    print(f"📚 Eğitim verisi hazırlandı:")
    print(f"  Sequence sayısı: {len(test_sequences)}")
    print(f"  Düşük değer örnekleri: {sum(labels)}")
    print(f"  Yüksek değer örnekleri: {len(labels) - sum(labels)}")
    
    # Eğitim
    print("\n🏋️ LowValueSpecialist eğitiliyor...")
    performances = specialist.fit(test_sequences, labels)
    
    print("\n📈 Model performansları:")
    for model_type, perf in performances.items():
        print(f"  {model_type}: {perf:.3f}")
    
    # Test tahminleri
    print("\n🔬 Test Tahminleri:")
    
    test_cases = [
        "Düşük değer trendi",
        "Crash sonrası",
        "Karışık pattern",
        "Yüksekten düşüğe",
        "Stabil düşük"
    ]
    
    for i, (seq, case_name) in enumerate(zip(test_sequences, test_cases)):
        print(f"\n{i+1}. {case_name}:")
        print(f"   Son değerler: {seq[-5:]}")
        
        is_low, prob, conf, details = specialist.predict_low_value(seq)
        
        print(f"   🎯 Tahmin: {'DÜŞÜK' if is_low else 'YÜKSEK'}")
        print(f"   📊 Düşük olasılık: {prob:.3f}")
        print(f"   💪 Güven: {conf:.3f}")
        
        # Detaylı analiz
        insights = specialist.get_low_value_insights(seq)
        print(f"   🔍 Ardışık düşük: {insights['consecutive_low']}")
        print(f"   📉 Düşük trend: {insights['low_trend']:.3f}")
        print(f"   💥 Crash pattern: {insights['crash_patterns']}")


def demonstrate_feature_extraction():
    """
    LowValueFeatureExtractor'ın özelleştirilmiş özellik çıkarımını göster
    """
    print("\n=== ÖZELLEŞTİRİLMİŞ ÖZELLİK ÇIKARIMI ===")
    
    extractor = LowValueFeatureExtractor(threshold=1.5)
    
    # Test sequences
    test_cases = {
        "Düşük değer ağırlıklı": [1.35, 1.28, 1.41, 1.33, 1.25, 1.39, 1.42, 1.36, 1.44, 1.38] * 5,
        "Yüksek değer ağırlıklı": [2.1, 1.8, 3.2, 2.5, 1.9, 2.7, 1.6, 2.3, 1.7, 2.1] * 5,
        "Crash pattern": [5.2, 8.1, 1.1, 1.2, 3.5, 1.3, 7.2, 1.15, 2.8, 1.25] * 5,
        "Karışık pattern": [1.45, 2.1, 1.33, 1.8, 1.28, 2.3, 1.41, 1.9, 1.36, 2.0] * 5
    }
    
    for case_name, sequence in test_cases.items():
        print(f"\n📊 {case_name}:")
        print(f"   Sequence uzunluğu: {len(sequence)}")
        print(f"   Son 10 değer: {sequence[-10:]}")
        
        features = extractor.extract_specialized_features(sequence)
        
        print(f"   🔧 Çıkarılan özellik sayısı: {len(features)}")
        print(f"   📉 1.5 altı oranı: {np.mean(np.array(sequence) < 1.5):.3f}")
        print(f"   📈 Düşük değer yoğunluğu (son 20): {features[12]:.3f}")  # Feature index 12
        print(f"   🔄 Ardışık düşük değer: {features[-7]:.0f}")  # Consecutive low feature
        print(f"   💥 Crash pattern: {features[13]:.3f}")  # Crash pattern feature


def create_performance_report(predictor):
    """
    Performans raporu oluştur
    """
    print("\n=== PERFORMANS RAPORU ===")
    
    stats = predictor.get_performance_stats()
    
    print(f"📊 Genel İstatistikler:")
    print(f"  Toplam tahmin: {stats['model_info']['total_predictions']}")
    print(f"  Ana modeller: {'✅' if stats['model_info']['general_models_loaded'] else '❌'}")
    print(f"  Düşük değer uzmanı: {'✅' if stats['model_info']['low_value_specialist_trained'] else '❌'}")
    
    for category in ['general', 'low_value', 'high_value']:
        if category in stats:
            cat_stats = stats[category]
            print(f"\n📈 {category.replace('_', ' ').title()} Performans:")
            print(f"  Accuracy: {cat_stats['accuracy']:.3f}")
            print(f"  Örneklem sayısı: {cat_stats['count']}")
            print(f"  Son 10 accuracy: {cat_stats['recent_accuracy']:.3f}")


def main():
    """
    Ana demo fonksiyonu
    """
    print("🚀 1.5 ALTI PROTEIN DEĞERLERİ İÇİN GELİŞMİŞ TAHMİN SİSTEMİ")
    print("=" * 70)
    
    try:
        # 1. Veri analizi
        values, low_values = analyze_low_value_data()
        
        # 2. Enhanced prediction demo
        demonstrate_enhanced_prediction()
        
        # 3. Low value specialist demo
        demonstrate_low_value_specialist()
        
        # 4. Feature extraction demo
        demonstrate_feature_extraction()
        
        # 5. Performance insights
        predictor = EnhancedJetXPredictor()
        insights = predictor.get_low_value_insights()
        
        if 'error' not in insights:
            print("\n=== DÜŞÜK DEĞER ANALİZ SONUÇLARI ===")
            for key, value in insights.items():
                if key != 'prediction_summary':
                    print(f"📊 {key}: {value}")
        
        print("\n✅ Demo tamamlandı!")
        print("\n🎯 Geliştirilmiş özellikler:")
        print("  • 1.5 altı değerlere özelleştirilmiş feature extraction")
        print("  • Multiple model ensemble (RF, GB, NN, Ensemble)")
        print("  • Cross-validation based model weighting")
        print("  • Conflict resolution between general and specialist models")
        print("  • Enhanced confidence scoring")
        print("  • Automatic performance tracking and retraining")
        
    except Exception as e:
        print(f"❌ Demo sırasında hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()