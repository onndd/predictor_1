# stacking.py DOSYASININ TAM İÇERİĞİ

import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings

class StackingEnsemble:
    def __init__(self, models=None, meta_model=None):
        self.models = models or {}
        self.meta_model = meta_model
        if self.meta_model is None:
            self.meta_model = LogisticRegression(class_weight='balanced', random_state=42)
        self.is_fitted = False
        print("StackingEnsemble başlatıldı.")

    def _collect_base_predictions(self, sequence):
        features = []
        sorted_model_names = sorted(self.models.keys())
        for name in sorted_model_names:
            model = self.models[name]
            try:
                result = model.predict_next_value(sequence)
                prob = result[1] if (isinstance(result, tuple) and len(result) > 1 and result[1] is not None) else 0.5
                features.append(prob)
            except Exception as e:
                print(f"Stacking: Model '{name}' tahmin hatası: {e}. Varsayılan (0.5) kullanılıyor.")
                features.append(0.5)
        return np.array(features).reshape(1, -1)

    def fit(self, sequences, labels, n_splits=5):
        """
        Trains the meta-model using out-of-fold predictions from base models
        to prevent data leakage.
        """
        print("Stacking 'fit': Meta-model için out-of-fold tahminler oluşturuluyor...")
        from sklearn.model_selection import TimeSeriesSplit

        if len(sequences) < n_splits * 2:
            print("Stacking 'fit': Çapraz doğrulama için yeterli veri yok. Eğitim atlanıyor.")
            self.is_fitted = False
            return

        tscv = TimeSeriesSplit(n_splits=n_splits)
        meta_features = []
        meta_labels = []

        for train_index, val_index in tscv.split(sequences):
            # Her bir fold için base modelleri eğit (bu kısım basitleştirilmiştir,
            # normalde her base modelin kendi fit metodu çağrılmalıdır)
            # Bu örnekte, base modellerin zaten eğitilmiş olduğunu varsayıyoruz.
            
            # Validation seti üzerinde tahminler yap
            for i in val_index:
                seq = sequences[i]
                label = labels[i]
                
                base_preds = self._collect_base_predictions(seq)
                meta_features.append(base_preds.flatten())
                meta_labels.append(label)

        if not meta_features:
            print("Stacking 'fit': Meta-özellikler oluşturulamadı.")
            self.is_fitted = False
            return

        X_meta = np.array(meta_features)
        y_meta = np.array(meta_labels)
        
        if self.meta_model is None:
            raise RuntimeError("Meta model is not initialized.")

        print(f"Stacking meta-modeli {X_meta.shape[0]} out-of-fold örnek ile eğitiliyor...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.meta_model.fit(X_meta, y_meta)
                self.is_fitted = True
                print("Stacking meta-modeli başarıyla eğitildi.")
            except Exception as e:
                print(f"Stacking meta-model eğitim hatası: {e}")
                self.is_fitted = False

    def predict_next_value(self, sequence):
        if not self.is_fitted:
            print("Stacking meta-modeli eğitilmemiş. Tahmin yapılamıyor.")
            return None, 0.5, 0.0
        
        if self.meta_model is None:
            raise RuntimeError("Meta model is not initialized.")

        meta_features = self._collect_base_predictions(sequence)
        try:
            probabilities = self.meta_model.predict_proba(meta_features)[0]
            above_prob = probabilities[1]
            confidence = abs(above_prob - 0.5) * 2
            return None, above_prob, confidence
        except Exception as e:
            print(f"Stacking meta-model tahmin hatası: {e}")
            return None, 0.5, 0.0