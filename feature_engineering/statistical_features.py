# statistical_features.py DOSYASININ OLMASI GEREKEN TAM VE DOĞRU HALİ

import numpy as np
from scipy import stats

def calculate_basic_stats(values, window_sizes=[10, 20, 50, 100, 200]):
    """
    Temel istatistiksel özellikleri hesaplar
    """
    n_samples = len(values)
    n_features = len(window_sizes) * 7
    features = np.zeros((n_samples, n_features))
    
    for i in range(n_samples):
        feature_idx = 0
        for window in window_sizes:
            start = max(0, i - window)
            window_vals = values[start:i]
            
            if len(window_vals) == 0:
                feature_idx += 7
                continue
            
            mean = np.mean(window_vals)
            median = np.median(window_vals)
            std = np.std(window_vals)
            min_val = np.min(window_vals)
            max_val = np.max(window_vals)
            skewness = stats.skew(window_vals) if len(window_vals) > 2 else 0
            kurtosis = stats.kurtosis(window_vals) if len(window_vals) > 2 else 0
            
            features[i, feature_idx] = mean
            features[i, feature_idx + 1] = median
            features[i, feature_idx + 2] = std
            features[i, feature_idx + 3] = min_val
            features[i, feature_idx + 4] = max_val
            features[i, feature_idx + 5] = skewness
            features[i, feature_idx + 6] = kurtosis
            feature_idx += 7
    return features

def calculate_threshold_runs(values, threshold=1.5, max_run_length=10):
    """
    Eşik değeri üzerinde/altında ardışık değer sayılarını hesaplar
    """
    n_samples = len(values)
    n_features = 4
    features = np.zeros((n_samples, n_features))
    above_threshold = [1 if x >= threshold else 0 for x in values]
    for i in range(n_samples):
        current_above_run = 0
        current_below_run = 0
        for j in range(i-1, max(0, i-max_run_length)-1, -1):
            if above_threshold[j] == 1:
                current_above_run += 1
                if current_below_run > 0:
                    break
            else:
                current_below_run += 1
                if current_above_run > 0:
                    break
        max_above_run = 0
        max_below_run = 0
        current_run = 1
        for j in range(1, min(i+1, max_run_length)):
            if i-j < 0:
                break
            if above_threshold[i-j] == above_threshold[i-j+1]:
                current_run += 1
            else:
                if above_threshold[i-j+1] == 1:
                    max_above_run = max(max_above_run, current_run)
                else:
                    max_below_run = max(max_below_run, current_run)
                current_run = 1
        if i > 0 and len(above_threshold) > 0:
            if above_threshold[0] == 1:
                max_above_run = max(max_above_run, current_run)
            else:
                max_below_run = max(max_below_run, current_run)
        features[i, 0] = current_above_run
        features[i, 1] = current_below_run
        features[i, 2] = max_above_run
        features[i, 3] = max_below_run
    return features

def calculate_trend_features(values, window_sizes=[10, 20, 50, 100]):
    """
    Trend ve mevsimsellik özellikleri hesaplar. RuntimeWarning'ları önlemek için kontroller eklendi.
    """
    n_samples = len(values)
    n_features = len(window_sizes) * 3
    features = np.zeros((n_samples, n_features))
    
    for i in range(n_samples):
        feature_idx = 0
        for window in window_sizes:
            start = max(0, i - window)
            window_vals = np.array(values[start:i])
            
            if len(window_vals) < 2:
                feature_idx += 3
                continue
            
            time_idx = np.arange(len(window_vals))
            
            try:
                slope, _, _, _, _ = stats.linregress(time_idx, window_vals)
            except ValueError:
                slope = 0
            
            window_std = np.std(window_vals)
            if window_std > 0 and len(window_vals) > 1:
                autocorr = np.corrcoef(window_vals[:-1], window_vals[1:])[0, 1]
            else:
                autocorr = 0
            
            trend_strength = slope / window_std if window_std > 0 else 0
            
            features[i, feature_idx] = slope if not np.isnan(slope) else 0
            features[i, feature_idx + 1] = autocorr if not np.isnan(autocorr) else 0
            features[i, feature_idx + 2] = trend_strength if not np.isnan(trend_strength) else 0
            feature_idx += 3
    return features

def calculate_advanced_stats(values, window_sizes=[10, 20, 50, 100]):
    """
    Volatilite, momentum ve kayan yüzdelikler gibi gelişmiş istatistiksel özellikleri hesaplar.
    """
    n_samples = len(values)
    n_features = len(window_sizes) * 4
    features = np.zeros((n_samples, n_features))
    
    for i in range(n_samples):
        feature_idx = 0
        for window in window_sizes:
            start = max(0, i - window)
            window_vals = np.array(values[start:i])
            
            if len(window_vals) < 2:
                feature_idx += 4
                continue
            
            returns = np.diff(window_vals) / (window_vals[:-1] + 1e-6)
            volatility = np.std(returns) if len(returns) > 0 else 0
            momentum = (window_vals[-1] - window_vals[0]) / (window_vals[0] + 1e-6)
            q25 = np.percentile(window_vals, 25)
            q75 = np.percentile(window_vals, 75)
            
            features[i, feature_idx] = volatility
            features[i, feature_idx + 1] = momentum
            features[i, feature_idx + 2] = q25
            features[i, feature_idx + 3] = q75
            feature_idx += 4
    return features

def extract_statistical_features(values):
    """
    Tüm istatistiksel özellikleri (temel, trend ve gelişmiş) çıkarır.
    """
    print("İstatistiksel özellikler çıkarılıyor (temel, eşik, trend ve gelişmiş)...")
    basic_stats = calculate_basic_stats(values)
    threshold_runs = calculate_threshold_runs(values)
    trend_features = calculate_trend_features(values)
    advanced_stats = calculate_advanced_stats(values)
    return np.hstack([basic_stats, threshold_runs, trend_features, advanced_stats])