# 🚀 Google Colab Model Eğitimi

Bu dizinde Google Colab'da JetX Predictor modellerini eğitmek için hazırlanmış notebook'lar bulunmaktadır.

## 📚 Notebook'lar

### 1. 🛠️ `colab_setup.ipynb`
**Ana kurulum notebook'u - İlk olarak çalıştırın!**

- ✅ GPU kontrolü ve sistem bilgisi
- ✅ GitHub repository klonlama
- ✅ Dependencies yükleme
- ✅ Python path kurulumu
- ✅ Import testleri
- ✅ Örnek JetX verisi oluşturma

### 2. 🧠 `nbeats_training.ipynb`
**N-Beats (Neural Basis Expansion Analysis) Model Eğitimi**

- 🎯 Interpretable time series forecasting
- 🎯 Trend ve seasonality ayrıştırması
- 🎯 JetX crash/pump pattern detection
- 🎯 GPU accelerated PyTorch eğitimi
- 🎯 Multi-output: Value, probability, confidence

### 3. 🔮 `tft_training.ipynb`
**TFT (Temporal Fusion Transformer) Model Eğitimi**

- 🎯 Multi-horizon forecasting
- 🎯 Attention mechanism ile yorumlanabilir
- 🎯 Enhanced feature engineering
- 🎯 Multi-head self-attention
- 🎯 Temporal pattern recognition

### 4. 🧮 `lstm_training.ipynb`
**Modern LSTM Model Eğitimi**

- 🎯 Sequential learning
- 🎯 Bidirectional LSTM + Attention
- 🎯 Multi-output prediction
- 🎯 TensorFlow GPU accelerated
- 🎯 Crash risk detection

## 🚀 Kullanım Kılavuzu

### Adım 1: Kurulum
```python
# Google Colab'da çalıştırın
# 1. colab_setup.ipynb notebook'unu açın
# 2. Runtime > Change runtime type > GPU seçin
# 3. Tüm hücreleri sırayla çalıştırın
```

### Adım 2: Model Eğitimi
```python
# Hangi modeli eğitmek istiyorsanız o notebook'u açın:
# - nbeats_training.ipynb
# - tft_training.ipynb
# - lstm_training.ipynb

# Her notebook'u sırayla çalıştırın
```

### Adım 3: Model Kaydetme
```python
# Modeller otomatik olarak kaydedilir:
# - /content/trained_models/ dizininde
# - Google Drive'a kopyalayabilirsiniz
# - JSON metadata dosyalarıyla birlikte
```

## 📊 Beklenen Sonuçlar

### N-Beats Model
- **Regression**: MAE, RMSE, R² metrikleri
- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Confidence**: Tahmin güvenilirlik skoru
- **Interpretability**: Trend ve seasonality ayrıştırması

### TFT Model
- **Multi-horizon**: Çoklu zaman adımı tahmini
- **Attention Analysis**: Hangi özelliklerin önemli olduğu
- **Feature Integration**: 7 farklı JetX özelliği
- **Temporal Patterns**: Uzun dönem bağımlılıkları

### LSTM Model
- **Sequential Memory**: Uzun dönem pattern recognition
- **Multi-output**: 4 farklı çıktı türü
- **Bidirectional**: Hem geçmiş hem gelecek bilgisi
- **Attention Enhanced**: Önemli time step'lere odaklanma

## 🔧 Teknik Detaylar

### GPU Gereksinimleri
- **Minimum**: Google Colab ücretsiz T4 GPU
- **Önerilen**: Colab Pro için daha güçlü GPU
- **Memory**: ~4GB GPU memory per model

### Eğitim Süreleri
- **N-Beats**: ~15-20 dakika (50 epoch)
- **TFT**: ~20-25 dakika (40 epoch)
- **LSTM**: ~25-30 dakika (50 epoch)

### Veri Gereksinimleri
- **Minimum**: 1000 JetX verisi
- **Önerilen**: 10000+ JetX verisi
- **Format**: SQLite veritabanı

## 🎯 Model Karşılaştırması

| Model | Avantajlar | Dezavantajlar | Kullanım Alanı |
|-------|-----------|---------------|-----------------|
| **N-Beats** | Interpretable, Fast | Limited features | Trend analysis |
| **TFT** | Multi-horizon, Attention | Complex | Feature analysis |
| **LSTM** | Sequential, Multi-output | Memory intensive | Pattern recognition |

## 📈 Performans Optimizasyonu

### Hyperparameter Tuning
```python
# Her notebook'ta parametreler ayarlanabilir:
SEQUENCE_LENGTH = 100  # Daha uzun = daha iyi context
BATCH_SIZE = 32        # GPU memory'ye göre ayarlayın
EPOCHS = 50            # Daha fazla = daha iyi performans
```

### GPU Memory Optimization
```python
# TensorFlow için:
tf.config.experimental.set_memory_growth(device, True)

# PyTorch için:
torch.cuda.empty_cache()
```

## 🔍 Hata Giderme

### Yaygın Hatalar
1. **GPU Memory Error**: Batch size'ı küçültün
2. **Import Error**: colab_setup.ipynb'yi tekrar çalıştırın
3. **Data Error**: Veritabanı dosyasını kontrol edin
4. **Time Limit**: Colab Pro kullanın

### Debug Komutları
```python
# GPU kontrolü
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")

# Memory kontrolü
import psutil
print(f"RAM Usage: {psutil.virtual_memory().percent}%")

# Model debug
model.summary()  # TensorFlow
print(model)     # PyTorch
```

## 🎉 Sonuç

Bu notebook'lar JetX tahmin modellerini Google Colab'da eğitmek için hazırlanmıştır. Her model farklı yaklaşımlar kullanır ve farklı avantajlar sağlar. En iyi sonuçları almak için:

1. **Tüm modelleri eğitin**
2. **Sonuçları karşılaştırın**
3. **Ensemble yöntemi kullanın**
4. **Hyperparameter tuning yapın**

---

**🚀 İyi eğitimler!**

## 📞 Destek

Sorun yaşarsanız:
- GitHub Issues'da soru açın
- Notebook'lardaki hata mesajlarını paylaşın
- GPU/memory bilgilerini ekleyin
