import math
import statistics
from datetime import datetime, timedelta
import json
import os

class MultiFactorConfidenceEstimator:
    """
    Çoklu Faktör Güven Skoru Sistemi
    
    Factors:
    1. Model Performance History (Model Performans Geçmişi)
    2. Data Quality Index (Veri Kalitesi İndeksi)
    3. Temporal Consistency (Zamansal Tutarlılık)
    4. Market Volatility Adaptation (Piyasa Volatilitesi Adaptasyonu)
    5. Prediction Certainty (Tahmin Kesinliği)
    6. Model Agreement (Model Uyumu)
    7. Recent Performance Trend (Son Performans Trendi)
    """
    
    def __init__(self, history_size=500, model_update_threshold_hours=24):
        """
        Çoklu Faktör Güven Tahmincisi

        Args:
            history_size: Geçmiş tahmin sayısı (artırıldı)
            model_update_threshold_hours: Model güncelleme eşiği (saat)
        """
        self.history_size = history_size
        self.model_update_threshold_hours = model_update_threshold_hours
        
        # Geçmiş veriler
        self.prediction_history = []
        self.actual_history = []
        self.confidence_history = []
        self.timestamp_history = []
        self.volatility_history = []
        
        # Model metadata
        self.model_last_updated = None
        self.model_training_data_quality = 0.8  # Başlangıç değeri
        
        # Performans tracking
        self.recent_accuracy_window = 50
        self.performance_trend_window = 100
        
        # Weights for different factors
        self.factor_weights = {
            'model_performance': 0.25,      # Model performansı
            'data_quality': 0.15,           # Veri kalitesi
            'temporal_consistency': 0.15,   # Zamansal tutarlılık
            'market_volatility': 0.15,      # Piyasa volatilitesi
            'prediction_certainty': 0.15,   # Tahmin kesinliği
            'model_freshness': 0.10,        # Model tazeliği
            'trend_alignment': 0.05         # Trend uyumu
        }

    def add_prediction(self, prediction, actual_value, confidence, timestamp=None, market_conditions=None):
        """
        Yeni bir tahmin-sonuç çiftini gelişmiş bilgilerle kaydeder

        Args:
            prediction: Tahmin edilen değer
            actual_value: Gerçek değer
            confidence: Tahmin güveni
            timestamp: Tahmin zamanı
            market_conditions: Piyasa koşulları (dict)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.prediction_history.append(prediction)
        self.actual_history.append(actual_value)
        self.confidence_history.append(confidence)
        self.timestamp_history.append(timestamp)
        
        # Market volatility hesapla
        if len(self.actual_history) >= 2:
            recent_values = self.actual_history[-10:]
            volatility = statistics.stdev(recent_values) if len(recent_values) > 1 else 0.0
            self.volatility_history.append(volatility)
        else:
            self.volatility_history.append(0.0)

        # Geçmiş boyutunu sınırla
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
            self.actual_history.pop(0)
            self.confidence_history.pop(0)
            self.timestamp_history.pop(0)
            self.volatility_history.pop(0)

    def calculate_model_performance_factor(self):
        """
        Model Performans Faktörü (0.0 - 1.0)
        Son performansa odaklanır
        """
        if len(self.prediction_history) < 10:
            return 0.5
        
        # Son tahminlerin doğruluğu
        recent_predictions = self.prediction_history[-self.recent_accuracy_window:]
        recent_actuals = self.actual_history[-self.recent_accuracy_window:]
        
        binary_predictions = [1 if p >= 1.5 else 0 for p in recent_predictions]
        binary_actuals = [1 if a >= 1.5 else 0 for a in recent_actuals]
        
        accuracy = sum(1 for p, a in zip(binary_predictions, binary_actuals) if p == a) / len(binary_predictions)
        
        # Exponential smoothing - son tahminler daha ağır
        weights = []
        for i in range(len(binary_predictions)):
            weight = math.exp(i / len(binary_predictions))
            weights.append(weight)
        
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        weighted_accuracy = sum(w * (1 if p == a else 0) for w, p, a in zip(weights, binary_predictions, binary_actuals))
        
        return weighted_accuracy

    def calculate_data_quality_factor(self):
        """
        Veri Kalitesi Faktörü (0.0 - 1.0)
        Veri tutarlılığı, eksik değerler, aykırı değerler
        """
        if len(self.actual_history) < 10:
            return 0.5
        
        recent_values = self.actual_history[-50:]
        
        # Aykırı değer kontrolü
        recent_values_sorted = sorted(recent_values)
        n = len(recent_values_sorted)
        q1 = recent_values_sorted[n//4] if n >= 4 else recent_values_sorted[0]
        q3 = recent_values_sorted[3*n//4] if n >= 4 else recent_values_sorted[-1]
        iqr = q3 - q1
        
        outliers = [v for v in recent_values if v < q1 - 1.5 * iqr or v > q3 + 1.5 * iqr]
        outlier_ratio = len(outliers) / len(recent_values)
        outlier_score = max(0.0, 1.0 - outlier_ratio * 2)
        
        # Veri tutarlılığı (çok fazla 0 veya negatif değer var mı?)
        valid_values = [v for v in recent_values if v > 0]
        validity_score = len(valid_values) / len(recent_values)
        
        # Veri çeşitliliği (tek bir değer mi tekrar ediyor?)
        unique_values = len(set(recent_values))
        diversity_score = min(1.0, unique_values / (len(recent_values) * 0.3))
        
        # Birleşik veri kalitesi
        data_quality = (outlier_score * 0.4 + validity_score * 0.4 + diversity_score * 0.2)
        
        return data_quality

    def calculate_temporal_consistency_factor(self):
        """
        Zamansal Tutarlılık Faktörü (0.0 - 1.0)
        Tahminlerin zaman içindeki tutarlılığı
        """
        if len(self.prediction_history) < 20:
            return 0.5
        
        # Son 20 tahmindeki tutarlılık
        recent_predictions = self.prediction_history[-20:]
        recent_confidences = self.confidence_history[-20:]
        
        # Güven seviyesi tutarlılığı
        confidence_std = statistics.stdev(recent_confidences) if len(recent_confidences) > 1 else 0.0
        confidence_consistency = max(0.0, 1.0 - confidence_std * 2)
        
        # Tahmin değeri tutarlılığı
        prediction_changes = [abs(recent_predictions[i] - recent_predictions[i-1]) 
                             for i in range(1, len(recent_predictions))]
        avg_change = statistics.mean(prediction_changes) if prediction_changes else 0.0
        prediction_consistency = max(0.0, 1.0 - avg_change / 2.0)
        
        return (confidence_consistency + prediction_consistency) / 2

    def calculate_market_volatility_factor(self):
        """
        Piyasa Volatilitesi Adaptasyon Faktörü (0.0 - 1.0)
        Yüksek volatilite = düşük güven
        """
        if len(self.volatility_history) < 10:
            return 0.7
        
        current_volatility = statistics.mean(self.volatility_history[-10:])
        historical_volatility = statistics.mean(self.volatility_history)
        
        # Normalize volatility (0-1 arası)
        normalized_volatility = min(1.0, current_volatility / max(0.1, historical_volatility))
        
        # Yüksek volatilite = düşük güven
        volatility_factor = max(0.0, 1.0 - normalized_volatility * 0.5)
        
        return volatility_factor

    def calculate_model_freshness_factor(self):
        """
        Model Tazeliği Faktörü (0.0 - 1.0)
        Model ne kadar güncel?
        """
        if self.model_last_updated is None:
            return 0.5
        
        hours_since_update = (datetime.now() - self.model_last_updated).total_seconds() / 3600
        
        if hours_since_update <= self.model_update_threshold_hours:
            return 1.0
        elif hours_since_update <= self.model_update_threshold_hours * 2:
            return 0.8
        elif hours_since_update <= self.model_update_threshold_hours * 7:
            return 0.6
        else:
            return 0.3

    def calculate_trend_alignment_factor(self):
        """
        Trend Uyum Faktörü (0.0 - 1.0)
        Tahminler mevcut trend ile uyumlu mu?
        """
        if len(self.actual_history) < 20:
            return 0.5
        
        # Son 20 değerdeki trend
        recent_values = self.actual_history[-20:]
        
        # Basit trend hesapla (lineer regresyon)
        n = len(recent_values)
        x_sum = sum(range(n))
        y_sum = sum(recent_values)
        xy_sum = sum(i * recent_values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        # Slope hesaplama
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum) if (n * x2_sum - x_sum * x_sum) != 0 else 0
        
        # R-squared hesaplama (basitleştirilmiş)
        y_mean = statistics.mean(recent_values)
        ss_tot = sum((y - y_mean) ** 2 for y in recent_values)
        
        # Predicted values
        y_pred = [slope * i + (y_sum - slope * x_sum) / n for i in range(n)]
        ss_res = sum((recent_values[i] - y_pred[i]) ** 2 for i in range(n))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        trend_strength = abs(r_squared)
        
        # Son tahmin trend ile uyumlu mu?
        if len(self.prediction_history) >= 2:
            last_prediction = self.prediction_history[-1]
            second_last_prediction = self.prediction_history[-2]
            prediction_trend = last_prediction - second_last_prediction
            
            # Trend yönü uyumu
            if slope > 0 and prediction_trend > 0:
                alignment = 1.0
            elif slope < 0 and prediction_trend < 0:
                alignment = 1.0
            elif abs(slope) < 0.1:  # Nötr trend
                alignment = 0.8
            else:
                alignment = 0.3
        else:
            alignment = 0.5
        
        return trend_strength * alignment

    def estimate_confidence(self, prediction, above_prob, uncertainty=None, market_conditions=None):
        """
        Çoklu faktör güven skorunu hesaplar

        Args:
            prediction: Tahmin edilen değer
            above_prob: Eşik üstü olasılığı
            uncertainty: Tahmin belirsizliği
            market_conditions: Piyasa koşulları

        Returns:
            dict: Detaylı güven analizi
        """
        # Faktörleri hesapla
        factors = {
            'model_performance': self.calculate_model_performance_factor(),
            'data_quality': self.calculate_data_quality_factor(),
            'temporal_consistency': self.calculate_temporal_consistency_factor(),
            'market_volatility': self.calculate_market_volatility_factor(),
            'model_freshness': self.calculate_model_freshness_factor(),
            'trend_alignment': self.calculate_trend_alignment_factor()
        }
        
        # Prediction certainty hesapla
        if above_prob is not None and not math.isnan(above_prob):
            certainty = 1.0 - 2.0 * abs(above_prob - 0.5)
        else:
            certainty = 0.5
        
        if uncertainty is not None:
            certainty *= (1.0 - min(1.0, uncertainty))
        
        factors['prediction_certainty'] = certainty
        
        # Ağırlıklı toplam güven skoru
        total_confidence = sum(
            factors[factor] * self.factor_weights[factor] 
            for factor in factors.keys()
        )
        
        # 0-1 arasında normalize et
        total_confidence = max(0.0, min(1.0, total_confidence))
        
        # Güven seviyesi kategorisi
        if total_confidence >= 0.85:
            confidence_level = "Çok Yüksek"
        elif total_confidence >= 0.7:
            confidence_level = "Yüksek"
        elif total_confidence >= 0.55:
            confidence_level = "Orta"
        elif total_confidence >= 0.4:
            confidence_level = "Düşük"
        else:
            confidence_level = "Çok Düşük"
        
        return {
            'total_confidence': total_confidence,
            'confidence_level': confidence_level,
            'factors': factors,
            'factor_weights': self.factor_weights,
            'recommendations': self._generate_recommendations(factors, total_confidence)
        }

    def _generate_recommendations(self, factors, total_confidence):
        """
        Güven skoruna göre öneriler üret
        """
        recommendations = []
        
        if total_confidence < 0.5:
            recommendations.append("⚠️ Çok düşük güven - Bu tahmini kullanmayın")
        
        if factors['model_performance'] < 0.6:
            recommendations.append("📊 Model performansı düşük - Yeniden eğitim gerekli")
        
        if factors['data_quality'] < 0.7:
            recommendations.append("🔍 Veri kalitesi düşük - Veri kontrolü yapın")
        
        if factors['model_freshness'] < 0.5:
            recommendations.append("🔄 Model eski - Güncelleme gerekli")
        
        if factors['market_volatility'] < 0.5:
            recommendations.append("📈 Yüksek volatilite - Dikkatli olun")
        
        if factors['temporal_consistency'] < 0.6:
            recommendations.append("⏱️ Tutarsız tahminler - Daha fazla veri gerekli")
        
        if total_confidence >= 0.8:
            recommendations.append("✅ Yüksek güven - Güvenle kullanabilirsiniz")
        
        return recommendations

    def update_model_metadata(self, last_updated=None, training_data_quality=None):
        """
        Model metadata'sını güncelle
        """
        if last_updated:
            self.model_last_updated = last_updated
        if training_data_quality:
            self.model_training_data_quality = training_data_quality

    def get_confidence_summary(self):
        """
        Güven sistemi özeti
        """
        if len(self.prediction_history) < 5:
            return "Yetersiz veri"
        
        recent_accuracy = self.calculate_model_performance_factor()
        avg_confidence = statistics.mean(self.confidence_history[-20:]) if len(self.confidence_history) >= 20 else 0.5
        
        return {
            'total_predictions': len(self.prediction_history),
            'recent_accuracy': recent_accuracy,
            'average_confidence': avg_confidence,
            'data_quality': self.calculate_data_quality_factor(),
            'model_age_hours': (datetime.now() - self.model_last_updated).total_seconds() / 3600 if self.model_last_updated else None
        }

    def save_confidence_data(self, filepath):
        """
        Güven verilerini dosyaya kaydet
        """
        data = {
            'prediction_history': self.prediction_history,
            'actual_history': self.actual_history,
            'confidence_history': self.confidence_history,
            'timestamp_history': [t.isoformat() for t in self.timestamp_history],
            'volatility_history': self.volatility_history,
            'model_last_updated': self.model_last_updated.isoformat() if self.model_last_updated else None,
            'model_training_data_quality': self.model_training_data_quality,
            'factor_weights': self.factor_weights
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_confidence_data(self, filepath):
        """
        Güven verilerini dosyadan yükle
        """
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.prediction_history = data.get('prediction_history', [])
            self.actual_history = data.get('actual_history', [])
            self.confidence_history = data.get('confidence_history', [])
            self.timestamp_history = [datetime.fromisoformat(t) for t in data.get('timestamp_history', [])]
            self.volatility_history = data.get('volatility_history', [])
            self.model_last_updated = datetime.fromisoformat(data['model_last_updated']) if data.get('model_last_updated') else None
            self.model_training_data_quality = data.get('model_training_data_quality', 0.8)
            self.factor_weights = data.get('factor_weights', self.factor_weights)
            
            return True
        except Exception as e:
            print(f"Güven verisi yükleme hatası: {e}")
            return False


# Geriye uyumluluk için alias
ConfidenceEstimator = MultiFactorConfidenceEstimator
