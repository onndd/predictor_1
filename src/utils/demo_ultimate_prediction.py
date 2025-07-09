"""
Demo: Ultimate Protein Tahmin Sistemi
Tüm Değer Aralıkları için Gelişmiş Tahmin

Bu script, düşük değerler (<1.5), orta değerler (1.5-10x) ve yüksek değerler (10x+)
için özelleştirilmiş tahmin sisteminin nasıl çalıştığını gösterir.
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime

# Ultimate tahmin sistemi
from enhanced_predictor_v3 import UltimateJetXPredictor
from models.low_value_specialist import LowValueSpecialist, LowValueFeatureExtractor
from models.high_value_specialist import HighValueSpecialist, HighValueFeatureExtractor


def analyze_all_value_ranges(db_path="jetx_data.db"):
    """
    Tüm değer aralıklarını analiz et
    """
    print("=== TÜM DEĞER ARALIKLARI ANALİZİ ===")
    
    conn = sqlite3.connect(db_path)
    query = "SELECT value FROM jetx_results ORDER BY id"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    values = df['value'].values
    
    print(f"📊 Toplam veri sayısı: {len(values)}")
    
    # Kategoriler
    categories = {
        'düşük (<1.5)': values < 1.5,
        'orta (1.5-10x)': (values >= 1.5) & (values < 10.0),
        'yüksek (10x+)': values >= 10.0,
        'çok yüksek (50x+)': values >= 50.0,
        'ekstrem (100x+)': values >= 100.0
    }
    
    for name, mask in categories.items():
        count = np.sum(mask)
        percentage = count / len(values) * 100
        if count > 0:
            category_values = values[mask]
            avg_val = np.mean(category_values)
            max_val = np.max(category_values)
            print(f"📈 {name}: {count} ({percentage:.2f}%) - Avg: {avg_val:.2f}, Max: {max_val:.2f}")
        else:
            print(f"📈 {name}: {count} ({percentage:.2f}%)")
    
    return values


def demonstrate_ultimate_prediction():
    """
    Ultimate tahmin sistemini demo et
    """
    print("\n=== ULTIMATE TAHMİN SİSTEMİ DEMO ===")
    
    # Ultimate predictor'ı başlat
    predictor = UltimateJetXPredictor()
    
    # Eğer modeller yoksa eğit
    if not predictor.load_models():
        print("🔧 Modeller bulunamadı, eğitim başlatılıyor...")
        success = predictor.train_ultimate_models(window_size=5000, focus_on_specialists=True)
        if not success:
            print("❌ Model eğitimi başarısız!")
            return
    
    # Test tahminleri
    print("\n📍 Ultimate Test Tahminleri:")
    
    for i in range(5):
        print(f"\n--- Ultimate Tahmin {i+1} ---")
        
        # Ultimate prediction
        result = predictor.predict_ultimate()
        
        if result:
            print(f"🎯 Tahmin Değeri: {result['predicted_value']:.3f}")
            print(f"📊 Karar: {result['decision_text']}")
            print(f"🎚️  Olasılık: {result['above_threshold_probability']:.3f}")
            print(f"💪 Güven: {result['confidence_score']:.3f}")
            print(f"🔍 Karar Kaynağı: {result['decision_source']}")
            print(f"📂 Kategori: {result['category_prediction']}")
            
            # Specialist detayları
            if result.get('low_value_prediction'):
                low_pred = result['low_value_prediction']
                print(f"🔻 Düşük Değer Uzmanı:")
                print(f"   - Düşük tahmin: {low_pred['is_low_value']}")
                print(f"   - Düşük olasılık: {low_pred['low_probability']:.3f}")
                print(f"   - Güven: {low_pred['confidence']:.3f}")
            
            if result.get('high_value_prediction'):
                high_pred = result['high_value_prediction']
                print(f"🔺 Yüksek Değer Uzmanı:")
                print(f"   - Yüksek tahmin: {high_pred['is_high_value']}")
                print(f"   - Yüksek olasılık: {high_pred['high_probability']:.3f}")
                print(f"   - Güven: {high_pred['confidence']:.3f}")
        else:
            print("❌ Tahmin alınamadı!")


def demonstrate_high_value_specialist():
    """
    HighValueSpecialist'in özelleştirilmiş yeteneklerini göster
    """
    print("\n=== YÜKSEK DEĞER UZMANI DEMO ===")
    
    # Test verisi oluştur (10x üzeri değerler ağırlıklı)
    test_sequences = [
        # Buildup to high value pattern
        [1.8, 2.3, 3.1, 4.2, 5.8, 7.5, 8.9, 12.3, 15.7, 11.2] * 10,
        
        # Volatile high values
        [15.2, 8.1, 22.3, 5.2, 35.8, 12.8, 41.2, 18.3, 25.4, 30.9] * 10,
        
        # Growth pattern
        [2.1, 2.8, 3.5, 4.9, 6.7, 9.2, 13.4, 18.7, 25.3, 32.1] * 10,
        
        # Mixed with extreme values
        [45.2, 125.1, 5.1, 78.2, 15.5, 8.8, 234.1, 33.3, 12.5, 89.4] * 10,
        
        # Medium to high transition
        [3.5, 4.8, 6.2, 5.6, 8.1, 7.7, 9.4, 11.9, 14.3, 16.0] * 10
    ]
    
    # HighValueSpecialist oluştur ve eğit
    specialist = HighValueSpecialist(threshold=10.0)
    
    # Labels oluştur (sequence sonundaki değer 10x üzeri mi?)
    labels = []
    for seq in test_sequences:
        labels.append(1 if seq[-1] >= 10.0 else 0)
    
    print(f"📚 Eğitim verisi hazırlandı:")
    print(f"  Sequence sayısı: {len(test_sequences)}")
    print(f"  Yüksek değer örnekleri: {sum(labels)}")
    print(f"  Düşük değer örnekleri: {len(labels) - sum(labels)}")
    
    # Eğitim
    print("\n🏋️ HighValueSpecialist eğitiliyor...")
    performances = specialist.fit(test_sequences, labels)
    
    print("\n📈 Model performansları:")
    for model_type, perf in performances.items():
        print(f"  {model_type}: {perf:.3f}")
    
    # Test tahminleri
    print("\n🔬 Test Tahminleri:")
    
    test_cases = [
        "Buildup pattern",
        "Volatile yüksek",
        "Growth pattern",
        "Extreme değerler",
        "Orta-yüksek geçiş"
    ]
    
    for i, (seq, case_name) in enumerate(zip(test_sequences, test_cases)):
        print(f"\n{i+1}. {case_name}:")
        print(f"   Son değerler: {seq[-5:]}")
        
        is_high, prob, conf, details = specialist.predict_high_value(seq)
        
        print(f"   🎯 Tahmin: {'YÜKSEK' if is_high else 'DÜŞÜK'}")
        print(f"   📊 Yüksek olasılık: {prob:.3f}")
        print(f"   💪 Güven: {conf:.3f}")
        
        # Detaylı analiz
        insights = specialist.get_high_value_insights(seq)
        print(f"   🔍 Momentum trend: {insights['momentum_trend']:.3f}")
        print(f"   📈 Ardışık büyüme: {insights['consecutive_growth']}")
        print(f"   🏗️  Buildup pattern: {insights['buildup_patterns']}")
        print(f"   📊 Son maksimum: {insights['max_recent_value']:.2f}")


def demonstrate_high_value_features():
    """
    HighValueFeatureExtractor'ın özelleştirilmiş özellik çıkarımını göster
    """
    print("\n=== YÜKSEK DEĞER ÖZELLİK ÇIKARIMI ===")
    
    extractor = HighValueFeatureExtractor(threshold=10.0)
    
    # Test sequences
    test_cases = {
        "Yüksek değer ağırlıklı": [12.5, 8.8, 15.1, 22.3, 18.5, 25.9, 31.2, 19.6, 27.4, 33.8] * 5,
        "Düşük değer ağırlıklı": [1.2, 1.8, 2.3, 1.5, 1.9, 2.7, 1.6, 2.3, 1.7, 2.1] * 5,
        "Buildup pattern": [2.1, 3.2, 4.5, 6.8, 9.1, 12.4, 15.7, 19.3, 24.1, 28.6] * 5,
        "Extreme volatilite": [145.2, 1.1, 89.3, 2.8, 234.5, 1.3, 67.2, 5.15, 128.8, 3.25] * 5
    }
    
    for case_name, sequence in test_cases.items():
        print(f"\n📊 {case_name}:")
        print(f"   Sequence uzunluğu: {len(sequence)}")
        print(f"   Son 10 değer: {sequence[-10:]}")
        
        features = extractor.extract_specialized_features(sequence)
        
        print(f"   🔧 Çıkarılan özellik sayısı: {len(features)}")
        print(f"   📈 10x üzeri oranı: {np.mean(np.array(sequence) >= 10.0):.3f}")
        print(f"   🚀 5x üzeri yoğunluğu: {features[10]:.3f}")  # Feature index for 5x+
        print(f"   📊 EMA ratio: {features[33]:.3f}")  # EMA ratio feature
        print(f"   🎯 Momentum değişim: {features[13]:.3f}")  # Momentum change
        print(f"   ⚡ Büyüme streak: {features[32]:.0f}")  # Growth streaks


def compare_specialist_systems():
    """
    Düşük ve yüksek değer uzmanlarını karşılaştır
    """
    print("\n=== UZMAN SİSTEMLER KARŞILAŞTIRMASI ===")
    
    # Test sequence'ları
    sequences = {
        "Çok düşük değerler": [1.1, 1.2, 1.15, 1.3, 1.25, 1.18, 1.35, 1.28, 1.22, 1.31] * 10,
        "Çok yüksek değerler": [15.2, 28.1, 12.3, 45.2, 18.5, 33.8, 21.2, 38.3, 25.4, 41.9] * 10,
        "Karışık değerler": [1.2, 15.8, 1.4, 22.3, 1.3, 8.7, 1.45, 12.1, 1.35, 18.4] * 10,
        "Orta değerler": [2.3, 3.8, 4.2, 5.6, 3.9, 6.1, 4.7, 5.8, 4.1, 6.3] * 10
    }
    
    # Uzmanları oluştur
    low_specialist = LowValueSpecialist(threshold=1.5)
    high_specialist = HighValueSpecialist(threshold=10.0)
    
    print("Uzman sistemler hazırlanıyor...")
    
    for seq_name, sequence in sequences.items():
        print(f"\n📊 {seq_name}:")
        print(f"   Son 10 değer: {sequence[-10:]}")
        
        # Low value specialist için basit eğitim
        low_labels = [1 if val < 1.5 else 0 for val in sequence[-100:]]
        high_labels = [1 if val >= 10.0 else 0 for val in sequence[-100:]]
        
        # Sadece yeterli örneklem varsa eğit
        if sum(low_labels) > 5:
            try:
                low_specialist.fit([sequence], [1 if sequence[-1] < 1.5 else 0])
                is_low, low_prob, low_conf, _ = low_specialist.predict_low_value(sequence)
                print(f"   🔻 Düşük uzman: {is_low} (prob: {low_prob:.3f}, conf: {low_conf:.3f})")
            except:
                print(f"   🔻 Düşük uzman: Eğitilemedi")
        
        if sum(high_labels) > 2:
            try:
                high_specialist.fit([sequence], [1 if sequence[-1] >= 10.0 else 0])
                is_high, high_prob, high_conf, _ = high_specialist.predict_high_value(sequence)
                print(f"   🔺 Yüksek uzman: {is_high} (prob: {high_prob:.3f}, conf: {high_conf:.3f})")
            except:
                print(f"   🔺 Yüksek uzman: Eğitilemedi")


def test_ultimate_performance():
    """
    Ultimate sistemin performansını test et
    """
    print("\n=== ULTIMATE SİSTEM PERFORMANS TESTİ ===")
    
    predictor = UltimateJetXPredictor()
    
    # Performance stats
    stats = predictor.get_ultimate_performance_stats()
    
    print(f"📊 Ultimate Sistem Durumu:")
    print(f"  Ana modeller: {'✅' if stats['model_info']['general_models_loaded'] else '❌'}")
    print(f"  Düşük değer uzmanı: {'✅' if stats['model_info']['low_value_specialist_trained'] else '❌'}")
    print(f"  Yüksek değer uzmanı: {'✅' if stats['model_info']['high_value_specialist_trained'] else '❌'}")
    print(f"  Toplam tahmin: {stats['model_info']['total_predictions']}")
    
    for category in ['general', 'low_value', 'medium_value', 'high_value']:
        if category in stats and stats[category]['count'] > 0:
            cat_stats = stats[category]
            print(f"\n📈 {category.replace('_', ' ').title()} Performans:")
            print(f"  Accuracy: {cat_stats['accuracy']:.3f}")
            print(f"  Örneklem sayısı: {cat_stats['count']}")
            print(f"  Son 10 accuracy: {cat_stats['recent_accuracy']:.3f}")
    
    # Ultimate insights
    try:
        insights = predictor.get_ultimate_insights()
        
        if 'error' not in insights:
            print(f"\n=== ULTIMATE INSIGHT'LAR ===")
            
            if 'general' in insights:
                gen = insights['general']
                print(f"📊 Genel Analiz:")
                print(f"  Son ortalama: {gen['recent_avg']:.2f}")
                print(f"  Volatilite: {gen['volatility']:.2f}")
                print(f"  Trend: {gen['trend']:.2f}")
                print(f"  Son maksimum: {gen['max_recent']:.2f}")
            
            if 'low_value' in insights:
                print(f"🔻 Düşük değer insights mevcut")
            
            if 'high_value' in insights:
                print(f"🔺 Yüksek değer insights mevcut")
        
    except Exception as e:
        print(f"Insights hatası: {e}")


def main():
    """
    Ana demo fonksiyonu
    """
    print("🚀 ULTIMATE PROTEIN TAHMİN SİSTEMİ")
    print("Tüm Değer Aralıkları için Gelişmiş AI Tahmin")
    print("=" * 70)
    
    try:
        # 1. Veri analizi - tüm aralıklar
        values = analyze_all_value_ranges()
        
        # 2. Ultimate prediction demo
        demonstrate_ultimate_prediction()
        
        # 3. High value specialist demo
        demonstrate_high_value_specialist()
        
        # 4. High value feature extraction demo
        demonstrate_high_value_features()
        
        # 5. Specialist comparison
        compare_specialist_systems()
        
        # 6. Performance test
        test_ultimate_performance()
        
        print("\n✅ Ultimate Demo tamamlandı!")
        print("\n🎯 Ultimate Sistem Özellikleri:")
        print("  • 3 uzman sistem entegrasyonu")
        print("  • Düşük değerler (<1.5) için özelleşmiş tahmin")
        print("  • Yüksek değerler (10x+) için özelleşmiş tahmin")
        print("  • Intelligent 3-way conflict resolution")
        print("  • 50+ özelleştirilmiş feature per specialist")
        print("  • Probability calibration for rare events")
        print("  • Category-based performance tracking")
        print("  • Auto-retraining with specialist focus")
        
        print("\n🏆 Tahmin Kategorileri:")
        print("  📉 DÜŞÜK (<1.5): LowValueSpecialist öncelik")
        print("  📊 ORTA (1.5-10x): General sistem")
        print("  📈 YÜKSEK (10x+): HighValueSpecialist öncelik")
        print("  🚀 EXTREME (50x+): Enhanced high value detection")
        
    except Exception as e:
        print(f"❌ Demo sırasında hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()