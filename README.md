# Gelişmiş JetX Tahmin Sistemi v4.1

JetX oyunu için gelişmiş, modüler ve test edilebilir makine öğrenmesi tabanlı tahmin sistemi. Bu proje, PyTorch tabanlı derin öğrenme modellerini, Optuna ile hiperparametre optimizasyonunu ve MLflow ile deney takibini bir araya getirerek sağlam bir eğitim ve tahmin altyapısı sunar.

## 🚀 Temel Mimari ve Felsefe

Bu projenin temel felsefesi, **sorumlulukların ayrılması (Separation of Concerns)** ilkesine dayanmaktadır. Her bileşen, iyi tanımlanmış tek bir görevi yerine getirir:

- **`DataManager`**: Tüm veri işlemlerinden sorumludur (yükleme, önbellekleme, dizi hazırlama).
- **`PipelineManager`**: Uçtan uca tüm eğitim akışını yönetir (veri, HPO, eğitim, raporlama).
- **`RollingTrainer`**: Belirli bir modelin artımlı veya kayan pencere ile eğitimini gerçekleştirir.
- **`BasePredictor`**: Tüm derin öğrenme modelleri için ortak bir arayüz ve temel eğitim döngüsü sağlar.
- **Model Sınıfları** (`NBeatsPredictor`, `EnhancedTFTPredictor` vb.): Sadece kendi ağ mimarilerini ve kayıp fonksiyonlarını tanımlarlar.

Bu modüler yapı, sistemin test edilmesini, bakımını ve yeni modellerle genişletilmesini son derece kolaylaştırır.

## ✨ Ana Özellikler

- **Merkezi Veri Yönetimi**: `DataManager` ile verimli ve tutarlı veri işleme.
- **Soyutlanmış Eğitim Akışı**: `PipelineManager` ile yönetilen, yapılandırılabilir ve tekrarlanabilir eğitim süreçleri.
- **Genişletilebilir Model Mimarisi**: `BasePredictor` sayesinde yeni derin öğrenme modellerini kolayca ekleme imkanı.
- **Otomatik HPO**: `Optuna` entegrasyonu ile her model için en iyi hiperparametrelerin otomatik olarak bulunması.
- **Deney Takibi**: `MLflow` ile tüm eğitim süreçlerinin, parametrelerin ve sonuçların kaydedilmesi ve izlenmesi.
- **İnteraktif Colab Arayüzü**: Modelleri kod yazmadan, sadece bir arayüz üzerinden seçip eğitebilme imkanı.
- **Kapsamlı Testler**: `pytest` ile yazılmış birim ve entegrasyon testleri ile sistemin güvenilirliğinin sağlanması.

## 📁 Proje Yapısı

```
predictor_1/
├── colab/
│   └── jetx_model_trainer.ipynb      # İnteraktif Colab eğitim not defteri
├── data/
│   ├── jetx_data.db                  # SQLite veritabanı
│   └── cache/                        # Veri önbellek dosyaları
├── docs/                             # Proje dokümantasyonu
├── mlruns/                           # MLflow deney kayıtları
├── reports/                          # Eğitim sonrası raporlar
├── src/
│   ├── config/
│   │   └── settings.py               # Merkezi konfigürasyon yükleyici
│   ├── data_processing/
│   │   ├── manager.py                # ⭐️ Merkezi DataManager
│   │   └── ...                       # Diğer veri işleme modülleri
│   ├── evaluation/
│   │   └── ...                       # Metrik, raporlama ve test modülleri
│   ├── models/
│   │   ├── base_predictor.py         # ⭐️ Tüm modeller için soyut temel sınıf
│   │   └── deep_learning/
│   │       ├── n_beats_model.py
│   │       ├── enhanced_tft_model.py
│   │       └── ...
│   └── training/
│       ├── pipeline_manager.py       # ⭐️ Tüm eğitim akışını yöneten ana sınıf
│       ├── rolling_trainer.py        # Kayan pencere eğitim mantığı
│       └── model_registry.py         # Eğitimli modelleri takip eden kayıt defteri
├── tests/
│   ├── data_processing/
│   │   └── test_manager.py           # DataManager için birim testleri
│   └── training/
│       └── test_pipeline.py          # PipelineManager için entegrasyon testi
├── config.yaml                       # ⭐️ Merkezi yapılandırma dosyası
└── README.md                         # Bu dosya
```

## 🚀 Hızlı Başlangıç (Google Colab)

Bu proje, en kolay şekilde Google Colab üzerinde çalıştırılmak üzere tasarlanmıştır.

1.  **Colab'da Açın**: [`colab/jetx_model_trainer.ipynb`](colab/jetx_model_trainer.ipynb) dosyasını Google Colab'da açın.
2.  **GPU'yu Etkinleştirin**: `Runtime > Change runtime type` menüsünden `GPU` seçeneğini etkinleştirin.
3.  **Hücreleri Çalıştırın**: Not defterindeki hücreleri sırayla çalıştırın.
    - **1. Hücre**: Gerekli ortamı kurar ve projeyi klonlar.
    - **2. Hücre**: MLflow arayüzünü başlatmak için sizden bir `ngrok` authtoken isteyecektir.
    - **3. Hücre**: Eğitmek istediğiniz modelleri seçip "Eğitimi Başlat" butonuna tıklayarak tüm süreci başlatabileceğiniz interaktif bir arayüz sunar.

## 🔧 Yerel Ortamda Çalıştırma

### Kurulum

```bash
# Projeyi klonlayın
git clone https://github.com/onndd/predictor_1.git
cd predictor_1

# Gerekli kütüphaneleri yükleyin
pip install -r requirements_enhanced.txt
```

### Testleri Çalıştırma

Sistemin doğru çalıştığından emin olmak için testleri çalıştırın:

```bash
pytest
```

### Eğitim Pipeline'ını Başlatma

Tüm modeller için tam bir eğitim, HPO ve raporlama sürecini başlatmak için:

```bash
# src/training/pipeline_manager.py dosyasını doğrudan çalıştırabilirsiniz
# (veya kendi başlangıç script'inizi oluşturabilirsiniz)

# Örnek: Sadece N-Beats ve TFT modellerini eğitmek için
python -c "from src.training.pipeline_manager import PipelineManager; PipelineManager(models_to_train=['N-Beats', 'TFT']).run_pipeline()"
```

## ⚙️ Merkezi Yapılandırma (`config.yaml`)

Projenin tüm davranışları [`config.yaml`](config.yaml:1) dosyası üzerinden kontrol edilir. Bu dosya üzerinden:
-   Veritabanı ve diğer dosya yollarını,
-   Genel eğitim parametrelerini (epoch, batch size vb.),
-   Her bir modelin varsayılan hiperparametrelerini,
-   Optuna için HPO arama uzaylarını
kolayca değiştirebilirsiniz. Bu, kodda herhangi bir değişiklik yapmadan sistemi farklı veri setleri ve senaryolar için ayarlamanıza olanak tanır.
