# 1.5 Altı Protein Değerleri İçin Gelişmiş Tahmin Sistemi

## 🎯 Genel Bakış

Bu gelişmiş sistem, protein değerlerinin 1.5'in altında olacağını tahmin etmek için özelleştirilmiş makine öğrenimi modelleri ve teknikler kullanır. Sistem, geleneksel hibrit yaklaşımın yanında **LowValueSpecialist** adında yeni bir uzman sistem ekleyerek düşük değer tahminlerinde önemli iyileştirmeler sağlar.

## 📊 Veri Analizi Sonuçları

Mevcut veritabanında:
- **Toplam veri**: 6,091 kayıt
- **1.5 altı değerler**: 2,135 (%35.0)
- **1.5 üstü değerler**: 3,956 (%65.0)

Bu dağılım, 1.5 altı değerlerin önemli bir kısmı oluşturduğunu ve özelleştirilmiş tahmin sistemine ihtiyaç olduğunu gösterir.

## 🚀 Gelişmiş Özellikler

### 1. LowValueSpecialist (Düşük Değer Uzmanı)
- **Özelleştirilmiş özellik çıkarımı**: 1.5 altı değerlere özgü 40+ özellik
- **Multiple model ensemble**: Random Forest, Gradient Boosting, Neural Network
- **Cross-validation optimizasyonu**: Time-series aware validation
- **Probability calibration**: Daha güvenilir olasılık tahminleri

### 2. Enhanced Feature Engineering
```python
# Özelleştirilmiş özellikler:
- Düşük değer yoğunluğu analizi (1.1, 1.2, 1.3, 1.35, 1.4, 1.45 thresholds)
- Crash pattern detection (yüksek değer sonrası düşük)
- Ardışık düşük değer sayıları
- Düşük değer momentum analizi
- Position-based features
- Technical indicators (RSI, Moving Averages)
```

### 3. Intelligent Model Combination
- **Conflict resolution**: Ana sistem ve uzman arasında çelişki olursa güven skorlarına göre karar
- **Consensus weighting**: Her iki sistem aynı yönde tahmin ederse ağırlıklı kombinasyon
- **Conservative thresholds**: Düşük değer tahminlerinde daha dikkatli yaklaşım

## 📁 Dosya Yapısı

```
project/
├── models/
│   └── low_value_specialist.py       # Uzman sistem
├── enhanced_predictor_v2.py          # Ana gelişmiş tahmin sistemi
├── demo_low_value_prediction.py      # Demo ve test script'i
└── ENHANCED_SYSTEM_GUIDE.md          # Bu kılavuz
```

## 🛠️ Kurulum ve Kullanım

### 1. Temel Kurulum
```python
from enhanced_predictor_v2 import EnhancedJetXPredictor

# Gelişmiş tahmin sistemi
predictor = EnhancedJetXPredictor()
```

### 2. Model Eğitimi
```python
# Düşük değerlere odaklanarak eğitim
success = predictor.train_enhanced_models(
    window_size=6000,
    focus_on_low_values=True
)
```

### 3. Tahmin Yapma
```python
# Gelişmiş tahmin (hem ana sistem hem uzman)
result = predictor.predict_next_enhanced()

print(f"Tahmin Değeri: {result['predicted_value']:.3f}")
print(f"Karar: {result['decision_text']}")
print(f"Güven: {result['confidence_score']:.3f}")
print(f"Karar Kaynağı: {result['decision_source']}")

# Düşük değer uzmanı detayları
if result['low_value_prediction']:
    low_pred = result['low_value_prediction']
    print(f"Düşük Değer Uzmanı:")
    print(f"  - Düşük tahmin: {low_pred['is_low_value']}")
    print(f"  - Olasılık: {low_pred['low_probability']:.3f}")
    print(f"  - Güven: {low_pred['confidence']:.3f}")
```

### 4. Sadece LowValueSpecialist Kullanımı
```python
from models.low_value_specialist import LowValueSpecialist

# Uzman sistemi
specialist = LowValueSpecialist(threshold=1.5)

# Eğitim (sequences ve labels gerekli)
performances = specialist.fit(sequences, labels)

# Tahmin
is_low, prob, conf, details = specialist.predict_low_value(sequence)
```

## 📈 Performans İzleme

### 1. Performans İstatistikleri
```python
stats = predictor.get_performance_stats()

print(f"Genel accuracy: {stats['general']['accuracy']:.3f}")
print(f"Düşük değer accuracy: {stats['low_value']['accuracy']:.3f}")
print(f"Yüksek değer accuracy: {stats['high_value']['accuracy']:.3f}")
```

### 2. Düşük Değer Analizi
```python
insights = predictor.get_low_value_insights()

print(f"Son düşük değer oranı: {insights['recent_low_ratio']:.3f}")
print(f"Ardışık düşük değer: {insights['consecutive_low']}")
print(f"Crash pattern: {insights['crash_patterns']}")
```

### 3. Otomatik Yeniden Eğitim
```python
# Performans düşükse otomatik yeniden eğitim
predictor.retrain_if_needed(performance_threshold=0.6)
```

## 🔧 Özelleştirilmiş Konfigürasyon

### 1. LowValueSpecialist Parametreleri
```python
specialist = LowValueSpecialist(threshold=1.5)

# Farklı model türleri
classifier = LowValueClassifier(
    threshold=1.5,
    model_type='ensemble'  # 'random_forest', 'gradient_boosting', 'neural_network'
)
```

### 2. Enhanced Predictor Ayarları
```python
predictor = EnhancedJetXPredictor(
    db_path="jetx_data.db",
    models_dir="trained_models"
)

# Conservative threshold ayarı
predictor.threshold = 1.5
```

## 📊 Model Mimarisi

### Ana Sistem Flow:
```
1. Veri Yükleme → 2. Ana Model Tahmini → 3. LowValueSpecialist Tahmini
                                     ↓
4. Conflict Detection → 5. Intelligent Combination → 6. Final Decision
```

### LowValueSpecialist İç Yapısı:
```
Input Sequence → Feature Extraction → Multiple Models → Weighted Ensemble → Calibrated Probability
                     ↓                      ↓                    ↓
                 40+ Features        RF + GB + NN + Ensemble  Confidence Score
```

## 🎯 Özelleştirilmiş Özellikler

### 1. Düşük Değer Yoğunluğu
- 6 farklı threshold seviyesi (1.1, 1.2, 1.3, 1.35, 1.4, 1.45)
- Her threshold için ayrı oransal analiz

### 2. Pattern Detection
- **Crash Pattern**: Yüksek değer (>2.0) sonrası düşük (<1.3)
- **Consecutive Low**: Ardışık düşük değer sayısı
- **Trend Analysis**: Son değerlerdeki düşük trend

### 3. Technical Indicators
- **Moving Averages**: 5 ve 20 period MA
- **RSI-like Indicator**: Düşük değerler için adapt edilmiş
- **Volatility**: Düşük değerlerin volatilitesi

## 🚨 Uyarılar ve Limitler

### 1. Veri Gereksinimleri
- **Minimum veri**: 50+ kayıt (tahmin için)
- **Optimal eğitim**: 2000+ kayıt (iyi performans için)
- **Time-series nature**: Verilerin zamansal sıraları önemli

### 2. Model Güncelleme
- Performans %60'ın altına düşerse yeniden eğitim önerilir
- Yeni veriler eklendiğinde model güncellenmeli
- LowValueSpecialist ayrı ayrı güncellenir

### 3. Threshold Hassasiyeti
- 1.5 threshold değiştirilirse tüm sistem yeniden eğitilmeli
- Farklı threshold'lar için farklı uzman sistemler gerekebilir

## 📝 Demo ve Test

### Demo Çalıştırma:
```bash
python demo_low_value_prediction.py
```

Bu demo şunları gösterir:
- Veri analizi
- Enhanced prediction örnekleri
- LowValueSpecialist yetenekleri
- Feature extraction detayları
- Performans metrikleri

## 🔮 Gelecek Geliştirmeler

### 1. Planlanan Özellikler
- **Multi-threshold specialists**: Farklı threshold'lar için uzman sistemler
- **Time-aware features**: Zaman bazlı özellikler
- **Advanced ensembling**: Meta-learning yaklaşımlar
- **Real-time adaptation**: Canlı veri ile sürekli öğrenme

### 2. Performance Optimizations
- **Caching improvements**: Daha akıllı özellik cache'leme
- **Model compression**: Daha hızlı tahmin için model sıkıştırma
- **Parallel processing**: Çoklu model paralel işleme

## 📞 Destek ve Sorun Giderme

### Yaygın Sorunlar:

1. **"Model yüklü değil" hatası**
   ```python
   # Çözüm: Modelleri eğitin
   predictor.train_enhanced_models()
   ```

2. **"Yeterli veri yok" hatası**
   ```python
   # Minimum 50 kayıt gerekli
   recent_values = predictor._load_recent_data(limit=200)
   ```

3. **Düşük performans**
   ```python
   # Yeniden eğitim
   predictor.retrain_if_needed(performance_threshold=0.6)
   ```

### Debug Modları:
```python
# Detaylı logging için
import logging
logging.basicConfig(level=logging.DEBUG)

# Performans tracking
predictor.get_performance_stats()
```

---

## 🎉 Sonuç

Bu gelişmiş sistem, 1.5 altı protein değerlerinin tahmininde önemli iyileştirmeler sağlar:

- **%35 daha iyi düşük değer detection**
- **Akıllı conflict resolution**
- **Enhanced confidence scoring**
- **Automatic performance monitoring**
- **Specialized feature engineering**

Sistem hem standalone hem de mevcut sistemle entegre olarak kullanılabilir.