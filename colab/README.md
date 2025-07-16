# 🚀 JetX Model Trainer - Google Colab Interface

Bu klasör, JetX tahmin modellerini Google Colab ortamında interaktif bir arayüz ile eğitmenizi sağlayan unified notebook'u içerir.

## 📁 Dosya Yapısı

```
colab/
├── jetx_model_trainer.ipynb    # Ana unified training notebook
├── README.md                   # Bu dosya
└── models/                     # Eğitilmiş modeller için klasör
```

## 🎯 Özellikler

### ✨ **Interactive Interface**
- 🎮 **Model Seçimi**: Dropdown ile N-Beats, TFT, LSTM, Ensemble
- ⚙️ **Parametre Ayarlama**: Slider'lar ile real-time parametre optimizasyonu
- 📊 **Real-time Progress**: Eğitim sürecini canlı takip
- 📈 **Model Karşılaştırma**: Performans dashboard'u
- 💾 **Otomatik Kaydetme**: Modeller otomatik kaydedilir

### 🔧 **Desteklenen Modeller**
- **N-Beats**: Interpretable time series forecasting
- **TFT**: Temporal Fusion Transformer with attention
- **LSTM**: Modern LSTM with attention mechanism
- **Ensemble**: Birden fazla modelin kombinasyonu

### 📊 **Performans Metrikleri**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- Accuracy (Classification accuracy)
- F1-Score (Balanced precision/recall)

## 🚀 Hızlı Başlangıç

### 1. **Google Colab'da Açma**
```
1. https://colab.research.google.com/ adresine gidin
2. File > Open notebook > GitHub
3. URL: https://github.com/onndd/predictor_1.git
4. colab/jetx_model_trainer.ipynb dosyasını seçin
```

### 2. **GPU Aktivasyonu**
```
1. Runtime > Change runtime type
2. Hardware accelerator > GPU
3. GPU type > T4 (ücretsiz) veya daha güçlü
```

### 3. **Notebook Çalıştırma**
```
1. Tüm hücreleri sırayla çalıştırın (Ctrl+F9)
2. Kurulum hücrelerini bekleyin
3. Interactive arayüz otomatik açılacak
```

## 📚 Kullanım Kılavuzu

### 🎯 **Model Seçimi**

#### N-Beats
- **En İyi**: Trend analizi, hızlı eğitim
- **Önerilen**: Başlangıç için ideal
- **Parametreler**: Sequence length 50-200, epochs 30-100

#### TFT (Temporal Fusion Transformer)
- **En İyi**: Karmaşık pattern'lar, attention analizi
- **Önerilen**: Yüksek performans istiyorsanız
- **Parametreler**: Sequence length 100-200, epochs 20-60

#### LSTM
- **En İyi**: Sequential pattern'lar, çoklu çıktı
- **Önerilen**: Balanced performance
- **Parametreler**: Sequence length 50-150, epochs 30-80

#### Ensemble
- **En İyi**: Maksimum performans
- **Önerilen**: Diğer modeller eğitildikten sonra
- **Gereksinim**: En az 2 model eğitilmiş olmalı

### ⚙️ **Parametre Optimizasyonu**

#### Sequence Length
- **50-100**: Hızlı eğitim, temel pattern'lar
- **100-150**: Balanced performance
- **150-200**: Karmaşık pattern'lar, daha uzun eğitim

#### Epochs
- **10-30**: Hızlı test
- **30-60**: Normal eğitim
- **60-100**: Maksimum performans

#### Batch Size
- **16**: Daha az GPU memory, daha stabil
- **32**: Balanced (önerilen)
- **64-128**: Daha hızlı eğitim, daha fazla memory

#### Learning Rate
- **0.0001**: Konservatif, stabil
- **0.001**: Önerilen başlangıç
- **0.01**: Agresif, hızlı öğrenme

## 🔧 Adım Adım Eğitim

### 1. **Kurulum**
```python
# Sistem kurulumu hücresi
# Import'lar ve temel kurulum
# Repository kurulumu
# Veri hazırlama
```

### 2. **Model Seçimi**
```python
# Model dropdown'dan seç
# Parametreleri ayarla
# Model bilgilerini kontrol et
```

### 3. **Eğitim**
```python
# "🚀 Eğitimi Başlat" butonuna tıkla
# Progress bar'ı takip et
# Real-time loss değerlerini izle
```

### 4. **Sonuçlar**
```python
# Performans metriklerini görüntüle
# Model karşılaştırma dashboard'unu yenile
# En iyi modeli belirle
```

## 📊 Beklenen Performans

### N-Beats
- **Eğitim süresi**: ~15-20 dakika
- **GPU memory**: ~2GB
- **Interpretable results**: ✅

### TFT
- **Eğitim süresi**: ~20-25 dakika
- **GPU memory**: ~3GB
- **Attention analysis**: ✅

### LSTM
- **Eğitim süresi**: ~25-30 dakika
- **GPU memory**: ~4GB
- **Multi-output predictions**: ✅

## 🎮 Interface Özellikleri

### 🎯 **Model Seçimi Bölümü**
- Model dropdown
- Model bilgileri
- Parametre otomatik güncelleme

### ⚙️ **Parametre Ayarlama Bölümü**
- Sequence Length slider
- Epochs slider
- Batch Size dropdown
- Learning Rate log slider

### 🚀 **Eğitim Kontrolleri**
- Başlat/Durdur butonları
- Progress bar
- Real-time eğitim çıktısı

### 📊 **Dashboard Bölümü**
- Model listesi
- Performans karşılaştırma
- En iyi model seçimi
- Interactive grafikler

## 🔍 Troubleshooting

### GPU İle İlgili Sorunlar
```
Problem: GPU tanınmıyor
Çözüm: Runtime > Change runtime type > GPU
```

### Memory Sorunları
```
Problem: GPU memory yetersiz
Çözüm: Batch size'ı küçült (16'ya düşür)
```

### Import Hataları
```
Problem: Modül bulunamadı
Çözüm: Kurulum hücrelerini tekrar çalıştır
```

### Repository Sorunları
```
Problem: Git clone hatası
Çözüm: Internet bağlantısını kontrol et
```

## 🎉 Başarı İpuçları

### 1. **GPU Kontrolü**
```python
# GPU'nun çalıştığını kontrol et
import torch
print(torch.cuda.is_available())
```

### 2. **Veri Hazırlığı**
```python
# Veri hazırlama hücrelerini önce çalıştır
# SQLite veritabanının oluştuğunu kontrol et
```

### 3. **Parametre Tuning**
```python
# Küçük dataset'te hızlı test yap
# Epochs'u düşük tutarak dene
```

### 4. **Model Karşılaştırma**
```python
# Birden fazla model eğit
# Dashboard'u yenile
# En iyi modeli seç
```

## 📈 Örnek Workflow

```python
# 1. Kurulum
# Tüm kurulum hücrelerini çalıştır

# 2. Model Seçimi
# N-Beats seç
# Parametreleri ayarla

# 3. Eğitim
# Eğitimi başlat
# Progress'i takip et

# 4. Sonuçlar
# Performans metriklerini kontrol et
# Dashboard'u yenile

# 5. Karşılaştırma
# Farklı modeller dene
# En iyi modeli belirle
```

## 🔗 Faydalı Linkler

- **GitHub Repository**: https://github.com/onndd/predictor_1.git
- **Google Colab**: https://colab.research.google.com/
- **PyTorch Documentation**: https://pytorch.org/docs/
- **TensorFlow Documentation**: https://www.tensorflow.org/

## 🆘 Destek

Herhangi bir sorun yaşarsanız:
1. GitHub Issues'da rapor edin
2. Documentation'ı kontrol edin
3. Troubleshooting bölümünü inceleyin

---

**🎯 Başarılı eğitimler!**

> Bu notebook ile JetX tahmin modellerini profesyonel seviyede eğitebilir ve karşılaştırabilirsiniz.
