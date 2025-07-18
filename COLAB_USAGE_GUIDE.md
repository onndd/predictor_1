# Colab'da İstenilen Modeli Eğitme ve İndirme Kılavuzu

Bu kılavuz, `master_trainer.py` betiğini kullanarak sadece belirli modelleri nasıl eğiteceğinizi ve eğitilen en iyi modeli nasıl indireceğinizi açıklar.

## 1. Belirli Modelleri Eğitme

Eğitim betiğini, `--models` argümanı ile çalıştırarak sadece istediğiniz modelleri eğitebilirsiniz. Bu argüman, bir veya daha fazla modelin adını alabilir.

### Örnekler:

**Sadece TFT modelini eğitmek için:**
```bash
!python src/training/master_trainer.py --models TFT
```

**Hem TFT hem de N-Beats modellerini eğitmek için:**
```bash
!python src/training/master_trainer.py --models TFT N-Beats
```

**Tüm mevcut modelleri eğitmek için (varsayılan davranış):**
```bash
!python src/training/master_trainer.py
```

## 2. Eğitilen Modeli İndirme

Eğitim süreci tamamlandığında, betik, eğitilen her modelin en iyi performans gösteren versiyonunun (`.pth` dosyası) tam yolunu konsola yazdıracaktır.

### Örnek Çıktı:

```
📥 --- Download Best Models ---
To download the best performing model for each type, use the following paths:

  - Model: TFT
    Path: /content/predictor_1/trained_models/TFT_cycle_5_20250718_193518.pth

Example Colab download command: from google.colab import files; files.download('path/to/your/model.pth')
```

### Modeli İndirme Adımları:

1.  Yukarıdaki çıktıdan indirmek istediğiniz modelin **tam yolunu** kopyalayın.
2.  Yeni bir Colab hücresine aşağıdaki kodu yapıştırın ve `'path/to/your/model.pth'` kısmını kopyaladığınız yol ile değiştirin.
3.  Hücreyi çalıştırın. Modeliniz bilgisayarınıza indirilecektir.

```python
from google.colab import files

# Yukarıdaki çıktıdan kopyaladığınız model yolunu buraya yapıştırın
model_path_to_download = '/content/predictor_1/trained_models/TFT_cycle_5_20250718_193518.pth'

files.download(model_path_to_download)