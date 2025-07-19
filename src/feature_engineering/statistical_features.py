# statistical_features.py DOSYASININ OLMASI GEREKEN TAM VE DOĞRU HALİ

import numpy as np
from scipy import stats
import pandas as pd
from typing import List

def calculate_basic_stats(values: pd.Series, window_sizes: List[int]):
    """
    Temel istatistiksel özellikleri (ortalama ve std olmadan) hesaplar
    """
    n_samples = len(values)
    n_features = len(window_sizes) * 5  # 7'den 5'e düşürüldü
    features = np.zeros((n_samples, n_features))
    
    for i in range(n_samples):
        feature_idx = 0
        for window in window_sizes:
            start = max(0, i - window)
            window_vals = values[start:i]
            
            if len(window_vals) == 0:
                feature_idx += 5  # 7'den 5'e düşürüldü
                continue
            
            # mean ve std kaldırıldı
            median = np.median(window_vals)
            min_val = np.min(window_vals)
            max_val = np.max(window_vals)
            skewness = stats.skew(window_vals) if len(window_vals) > 2 else 0
            kurtosis = stats.kurtosis(window_vals) if len(window_vals) > 2 else 0
            
            # İndeksleme güncellendi
            features[i, feature_idx] = median
            features[i, feature_idx + 1] = min_val
            features[i, feature_idx + 2] = max_val
            features[i, feature_idx + 3] = skewness
            features[i, feature_idx + 4] = kurtosis
            feature_idx += 5  # 7'den 5'e düşürüldü
    return features

def calculate_threshold_runs(values: pd.Series, threshold=1.5, max_run_length=10):
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

def calculate_trend_features(values: pd.Series, window_sizes: List[int]):
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

def calculate_advanced_stats(values: pd.Series, window_sizes: List[int]):
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

def calculate_lag_and_rolling_features(values, lags: List[int], windows: List[int]):
    """
    Gecikme (lag) ve yuvarlanan (rolling) istatistik özelliklerini hesaplar.
    """
    df = pd.DataFrame(values, columns=['value'])
    
    # Lag features
    for lag in lags:
        df[f'lag_{lag}'] = df['value'].shift(lag)
        
    # Rolling features
    for window in windows:
        df[f'rolling_mean_{window}'] = df['value'].shift(1).rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['value'].shift(1).rolling(window=window).std()

    df.fillna(0, inplace=True)
    
    # Drop the original 'value' column
    return df.drop('value', axis=1).values

def extract_statistical_features(values: List[float], feature_windows: List[int], lag_windows: List[int], lags: List[int]):
    """
    Tüm istatistiksel özellikleri (temel, trend, gelişmiş, gecikme ve yuvarlanan) çıkarır.
    """
    print("İstatistiksel özellikler çıkarılıyor (temel, eşik, trend, gelişmiş, gecikme ve yuvarlanan)...")
    
    # Convert to pandas Series for easier manipulation
    series_values = pd.Series(values)
    
    # Calculate all feature sets using the provided window sizes
    basic_stats = calculate_basic_stats(series_values, window_sizes=feature_windows)
    threshold_runs = calculate_threshold_runs(series_values) # This one doesn't depend on windows
    trend_features = calculate_trend_features(series_values, window_sizes=feature_windows)
    advanced_stats = calculate_advanced_stats(series_values, window_sizes=feature_windows)
    lag_rolling_features = calculate_lag_and_rolling_features(values, lags=lags, windows=lag_windows)
    
    # Ensure all numpy arrays have the same number of rows
    min_len = min(len(basic_stats), len(threshold_runs), len(trend_features), len(advanced_stats), len(lag_rolling_features))

    return np.hstack([
        basic_stats[:min_len],
        threshold_runs[:min_len],
        trend_features[:min_len],
        advanced_stats[:min_len],
        lag_rolling_features[:min_len]
    ])