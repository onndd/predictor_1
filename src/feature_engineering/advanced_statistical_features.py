"""
Advanced Statistical Features Module
====================================
Bu modül sofistike finansal zaman serisi analizi için gelişmiş istatistiksel özellikler sağlar:
- Hurst Exponent (uzun bellek tespiti)
- Fractal Dimension Analysis (karmaşıklık ölçümü)
- Recurrence Quantification Analysis (RQA)
- Entropy tabanlı ölçümler
- Rejim değişiklik göstergeleri
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from scipy import stats
from scipy.signal import detrend
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

try:
    import nolds
    NOLDS_AVAILABLE = True
except ImportError:
    NOLDS_AVAILABLE = False
    print("Warning: nolds not available. Some fractal features will be disabled.")

try:
    import antropy as ant
    ANTROPY_AVAILABLE = True
except ImportError:
    ANTROPY_AVAILABLE = False
    print("Warning: antropy not available. Some entropy features will be disabled.")

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    print("Warning: ruptures not available. Change point detection will be disabled.")

try:
    from hurst import compute_Hc
    HURST_AVAILABLE = True
except ImportError:
    HURST_AVAILABLE = False
    print("Warning: hurst not available. Using fallback implementation.")


class AdvancedStatisticalFeatures:
    """Gelişmiş istatistiksel özellik çıkarıcı sınıf"""
    
    def __init__(self, window_sizes: List[int] = [10, 20, 50]):
        self.window_sizes = window_sizes
        
    def compute_hurst_exponent(self, series: np.ndarray) -> float:
        """
        Hurst exponent hesaplar (uzun bellek tespiti)
        H > 0.5: Persistent (trending)
        H < 0.5: Anti-persistent (mean reverting)  
        H = 0.5: Random walk
        """
        if len(series) < 10:
            return 0.5
            
        try:
            if HURST_AVAILABLE:
                H, _, _ = compute_Hc(series, kind='price', simplified=True)
                return H
            else:
                # R/S Analysis fallback implementation
                return self._rs_hurst(series)
        except:
            return 0.5
    
    def _rs_hurst(self, series: np.ndarray) -> float:
        """R/S Analysis ile Hurst exponent hesaplar"""
        try:
            lags = range(2, min(len(series)//2, 100))
            rs_values = []
            
            for lag in lags:
                # Reshape series into chunks
                chunks = [series[i:i+lag] for i in range(0, len(series)-lag+1, lag)]
                if len(chunks) < 2:
                    continue
                    
                rs_chunk = []
                for chunk in chunks:
                    if len(chunk) < 2:
                        continue
                    mean_chunk = np.mean(chunk)
                    cumdev = np.cumsum(chunk - mean_chunk)
                    R = np.max(cumdev) - np.min(cumdev)
                    S = np.std(chunk)
                    if S > 0:
                        rs_chunk.append(R/S)
                
                if rs_chunk:
                    rs_values.append(np.mean(rs_chunk))
            
            if len(rs_values) < 2:
                return 0.5
                
            # Log-log regression
            log_lags = np.log(lags[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
            return max(0.0, min(1.0, slope))
        except:
            return 0.5
    
    def compute_fractal_dimension(self, series: np.ndarray) -> Dict[str, float]:
        """
        Fractal dimension hesaplar (karmaşıklık ölçümü)
        Düşük değer = düzenli, yüksek değer = kaotik
        """
        results = {}
        
        try:
            # Higuchi Fractal Dimension
            if NOLDS_AVAILABLE and len(series) >= 10:
                results['higuchi_fd'] = nolds.higuchi_fd(series, kmax=min(10, len(series)//2))
            else:
                results['higuchi_fd'] = self._higuchi_fd_fallback(series)
                
            # Katz Fractal Dimension  
            results['katz_fd'] = self._katz_fractal_dimension(series)
            
            # Petrosian Fractal Dimension
            results['petrosian_fd'] = self._petrosian_fractal_dimension(series)
            
        except:
            results = {'higuchi_fd': 1.5, 'katz_fd': 1.5, 'petrosian_fd': 1.5}
            
        return results
    
    def _higuchi_fd_fallback(self, series: np.ndarray) -> float:
        """Higuchi FD fallback implementation"""
        try:
            N = len(series)
            if N < 10:
                return 1.5
                
            kmax = min(8, N//4)
            lk = []
            
            for k in range(1, kmax + 1):
                Lk = 0
                for m in range(k):
                    Lmk = 0
                    for i in range(1, int((N-m)/k)):
                        Lmk += abs(series[m + i*k] - series[m + (i-1)*k])
                    Lmk = Lmk * (N - 1) / (k * int((N-m)/k) * k)
                    Lk += Lmk
                lk.append(Lk/k)
            
            if len(lk) < 2:
                return 1.5
                
            lk = np.array(lk)
            k_values = np.arange(1, len(lk) + 1)
            
            # Log-log regression
            log_k = np.log(k_values)
            log_lk = np.log(lk)
            
            slope, _, _, _, _ = stats.linregress(log_k, log_lk)
            return max(1.0, min(2.0, -slope))
        except:
            return 1.5
    
    def _katz_fractal_dimension(self, series: np.ndarray) -> float:
        """Katz Fractal Dimension"""
        try:
            if len(series) < 3:
                return 1.5
                
            diffs = np.diff(series)
            L = np.sum(np.abs(diffs))  # Total length
            if L == 0:
                return 1.0
                
            d = np.max(np.abs(series - series[0]))  # Diameter
            if d == 0:
                return 1.0
                
            n = len(series) - 1
            fd = np.log10(n) / (np.log10(d/L) + np.log10(n))
            return max(1.0, min(2.0, fd))
        except:
            return 1.5
    
    def _petrosian_fractal_dimension(self, series: np.ndarray) -> float:
        """Petrosian Fractal Dimension"""
        try:
            if len(series) < 3:
                return 1.5
                
            # Count relative maxima
            diff = np.diff(series)
            N_delta = np.sum(diff[1:] * diff[:-1] < 0)
            
            n = len(series)
            if n <= 1:
                return 1.0
                
            fd = np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * N_delta)))
            return max(1.0, min(2.0, fd))
        except:
            return 1.5
    
    def compute_rqa_features(self, series: np.ndarray, dimension: int = 3, 
                           delay: int = 1, threshold: float = 0.1) -> Dict[str, float]:
        """
        Recurrence Quantification Analysis özellikleri
        """
        try:
            if len(series) < dimension + delay:
                return {
                    'rqa_recurrence_rate': 0.0,
                    'rqa_determinism': 0.0,
                    'rqa_entropy': 0.0,
                    'rqa_laminarity': 0.0
                }
            
            # Phase space reconstruction
            embedded = self._embed_series(series, dimension, delay)
            
            # Recurrence matrix
            distances = squareform(pdist(embedded))
            recurrence_matrix = (distances < threshold * np.std(distances)).astype(int)
            
            # RQA measures
            rr = np.sum(recurrence_matrix) / (len(recurrence_matrix) ** 2)
            det = self._compute_determinism(recurrence_matrix)
            ent = self._compute_rqa_entropy(recurrence_matrix)
            lam = self._compute_laminarity(recurrence_matrix)
            
            return {
                'rqa_recurrence_rate': rr,
                'rqa_determinism': det,
                'rqa_entropy': ent,
                'rqa_laminarity': lam
            }
        except:
            return {
                'rqa_recurrence_rate': 0.0,
                'rqa_determinism': 0.0,
                'rqa_entropy': 0.0,
                'rqa_laminarity': 0.0
            }
    
    def _embed_series(self, series: np.ndarray, dimension: int, delay: int) -> np.ndarray:
        """Time delay embedding for phase space reconstruction"""
        n = len(series) - (dimension - 1) * delay
        embedded = np.zeros((n, dimension))
        for i in range(dimension):
            embedded[:, i] = series[i * delay:i * delay + n]
        return embedded
    
    def _compute_determinism(self, recurrence_matrix: np.ndarray, min_length: int = 2) -> float:
        """Determinism measure from recurrence matrix"""
        try:
            diagonals = []
            n = len(recurrence_matrix)
            
            # Extract diagonal lines
            for offset in range(-(n-1), n):
                diagonal = np.diagonal(recurrence_matrix, offset=offset)
                if len(diagonal) >= min_length:
                    # Find consecutive 1s
                    consecutive = self._find_consecutive_ones(diagonal)
                    diagonals.extend(consecutive)
            
            diagonals = [d for d in diagonals if d >= min_length]
            
            if not diagonals:
                return 0.0
                
            total_recurrent_points = np.sum(recurrence_matrix)
            if total_recurrent_points == 0:
                return 0.0
                
            return sum(diagonals) / total_recurrent_points
        except:
            return 0.0
    
    def _find_consecutive_ones(self, binary_array: np.ndarray) -> List[int]:
        """Find lengths of consecutive 1s in binary array"""
        consecutive_lengths = []
        current_length = 0
        
        for val in binary_array:
            if val == 1:
                current_length += 1
            else:
                if current_length > 0:
                    consecutive_lengths.append(current_length)
                current_length = 0
        
        if current_length > 0:
            consecutive_lengths.append(current_length)
            
        return consecutive_lengths
    
    def _compute_rqa_entropy(self, recurrence_matrix: np.ndarray) -> float:
        """RQA entropy measure"""
        try:
            diagonals = []
            n = len(recurrence_matrix)
            
            for offset in range(-(n-1), n):
                diagonal = np.diagonal(recurrence_matrix, offset=offset)
                consecutive = self._find_consecutive_ones(diagonal)
                diagonals.extend(consecutive)
            
            if not diagonals:
                return 0.0
                
            # Compute entropy of diagonal length distribution
            unique, counts = np.unique(diagonals, return_counts=True)
            probabilities = counts / len(diagonals)
            return -np.sum(probabilities * np.log2(probabilities + 1e-10))
        except:
            return 0.0
    
    def _compute_laminarity(self, recurrence_matrix: np.ndarray, min_length: int = 2) -> float:
        """Laminarity measure (vertical line structures)"""
        try:
            vertical_lines = []
            n = len(recurrence_matrix)
            
            # Check vertical lines in each column
            for col in range(n):
                column = recurrence_matrix[:, col]
                consecutive = self._find_consecutive_ones(column)
                vertical_lines.extend(consecutive)
            
            vertical_lines = [v for v in vertical_lines if v >= min_length]
            
            if not vertical_lines:
                return 0.0
                
            total_recurrent_points = np.sum(recurrence_matrix)
            if total_recurrent_points == 0:
                return 0.0
                
            return sum(vertical_lines) / total_recurrent_points
        except:
            return 0.0
    
    def compute_entropy_features(self, series: np.ndarray) -> Dict[str, float]:
        """
        Çeşitli entropy ölçümlerini hesaplar
        """
        results = {}
        
        try:
            # Shannon Entropy
            results['shannon_entropy'] = self._shannon_entropy(series)
            
            # Approximate Entropy
            if ANTROPY_AVAILABLE and len(series) >= 10:
                results['approximate_entropy'] = ant.app_entropy(series, m=2, r=0.2*np.std(series))
                results['sample_entropy'] = ant.sample_entropy(series, m=2, r=0.2*np.std(series))
                results['permutation_entropy'] = ant.perm_entropy(series, order=3, normalize=True)
            else:
                results['approximate_entropy'] = self._approximate_entropy_fallback(series)
                results['sample_entropy'] = self._sample_entropy_fallback(series)
                results['permutation_entropy'] = self._permutation_entropy_fallback(series)
            
            # Spectral Entropy
            results['spectral_entropy'] = self._spectral_entropy(series)
            
        except:
            results = {
                'shannon_entropy': 0.0,
                'approximate_entropy': 0.0, 
                'sample_entropy': 0.0,
                'permutation_entropy': 0.0,
                'spectral_entropy': 0.0
            }
            
        return results
    
    def _shannon_entropy(self, series: np.ndarray, bins: int = 10) -> float:
        """Shannon entropy hesaplar"""
        try:
            if len(series) < 2:
                return 0.0
                
            hist, _ = np.histogram(series, bins=bins)
            hist = hist[hist > 0]  # Remove zero counts
            probabilities = hist / len(series)
            return -np.sum(probabilities * np.log2(probabilities))
        except:
            return 0.0
    
    def _approximate_entropy_fallback(self, series: np.ndarray, m: int = 2, r: float = None) -> float:
        """Approximate entropy fallback implementation"""
        try:
            if len(series) < m + 1:
                return 0.0
                
            if r is None:
                r = 0.2 * np.std(series)
                
            if r == 0:
                return 0.0
                
            N = len(series)
            
            def _maxdist(xi, xj, N, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([series[i:i + m] for i in range(N - m + 1)])
                C = np.zeros(N - m + 1)
                
                for i in range(N - m + 1):
                    template_i = patterns[i]
                    for j in range(N - m + 1):
                        if _maxdist(template_i, patterns[j], N, m) <= r:
                            C[i] += 1.0
                
                C[C == 0] = 1  # Avoid log(0)
                phi = np.mean(np.log(C / (N - m + 1.0)))
                return phi
            
            return _phi(m) - _phi(m + 1)
        except:
            return 0.0
    
    def _sample_entropy_fallback(self, series: np.ndarray, m: int = 2, r: float = None) -> float:
        """Sample entropy fallback implementation"""
        try:
            if len(series) < m + 1:
                return 0.0
                
            if r is None:
                r = 0.2 * np.std(series)
                
            if r == 0:
                return 0.0
                
            N = len(series)
            
            def _maxdist(xi, xj):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            patterns_m = [series[i:i + m] for i in range(N - m)]
            patterns_m1 = [series[i:i + m + 1] for i in range(N - m)]
            
            A = 0.0  # Number of matches for m+1
            B = 0.0  # Number of matches for m
            
            for i in range(N - m):
                for j in range(N - m):
                    if i != j:
                        if _maxdist(patterns_m[i], patterns_m[j]) <= r:
                            B += 1
                            if _maxdist(patterns_m1[i], patterns_m1[j]) <= r:
                                A += 1
            
            if B == 0:
                return 0.0
                
            return -np.log(A / B)
        except:
            return 0.0
    
    def _permutation_entropy_fallback(self, series: np.ndarray, order: int = 3) -> float:
        """Permutation entropy fallback implementation"""
        try:
            if len(series) < order:
                return 0.0
                
            # Extract ordinal patterns
            patterns = []
            for i in range(len(series) - order + 1):
                subseries = series[i:i + order]
                pattern = tuple(np.argsort(np.argsort(subseries)))
                patterns.append(pattern)
            
            # Count pattern frequencies
            unique_patterns, counts = np.unique(patterns, return_counts=True, axis=0)
            probabilities = counts / len(patterns)
            
            # Calculate entropy
            return -np.sum(probabilities * np.log2(probabilities))
        except:
            return 0.0
    
    def _spectral_entropy(self, series: np.ndarray) -> float:
        """Spectral entropy hesaplar"""
        try:
            if len(series) < 4:
                return 0.0
                
            # Power spectral density
            freqs, psd = self._welch_psd(series)
            psd = psd[psd > 0]  # Remove zero power
            
            if len(psd) == 0:
                return 0.0
                
            # Normalize to get probabilities
            psd_norm = psd / np.sum(psd)
            
            # Calculate entropy
            return -np.sum(psd_norm * np.log2(psd_norm))
        except:
            return 0.0
    
    def _welch_psd(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Welch method for power spectral density estimation"""
        try:
            from scipy.signal import welch
            freqs, psd = welch(series, nperseg=min(len(series)//4, 256))
            return freqs, psd
        except:
            # Simple periodogram fallback
            fft = np.fft.fft(series)
            psd = np.abs(fft) ** 2
            freqs = np.fft.fftfreq(len(series))
            return freqs[:len(freqs)//2], psd[:len(psd)//2]
    
    def compute_regime_change_indicators(self, series: np.ndarray) -> Dict[str, float]:
        """
        Rejim değişiklik göstergelerini hesaplar
        """
        results = {}
        
        try:
            # CUSUM change point detection
            results.update(self._cusum_indicators(series))
            
            # Variance change detection
            results.update(self._variance_change_indicators(series))
            
            # Trend change indicators
            results.update(self._trend_change_indicators(series))
            
            # Advanced change point detection with ruptures
            if RUPTURES_AVAILABLE and len(series) >= 10:
                results.update(self._ruptures_change_detection(series))
            
        except:
            results = {
                'cusum_positive': 0.0,
                'cusum_negative': 0.0,
                'variance_ratio': 1.0,
                'trend_change_strength': 0.0,
                'recent_change_prob': 0.0
            }
            
        return results
    
    def _cusum_indicators(self, series: np.ndarray) -> Dict[str, float]:
        """CUSUM tabanlı değişiklik göstergeleri"""
        try:
            if len(series) < 5:
                return {'cusum_positive': 0.0, 'cusum_negative': 0.0}
                
            mean_series = np.mean(series)
            std_series = np.std(series)
            
            if std_series == 0:
                return {'cusum_positive': 0.0, 'cusum_negative': 0.0}
            
            # Standardize series
            standardized = (series - mean_series) / std_series
            
            # CUSUM calculation
            h = 1.0  # Threshold
            cusum_pos = 0.0
            cusum_neg = 0.0
            max_cusum_pos = 0.0
            max_cusum_neg = 0.0
            
            for val in standardized:
                cusum_pos = max(0, cusum_pos + val - h)
                cusum_neg = max(0, cusum_neg - val - h)
                max_cusum_pos = max(max_cusum_pos, cusum_pos)
                max_cusum_neg = max(max_cusum_neg, cusum_neg)
            
            return {
                'cusum_positive': max_cusum_pos,
                'cusum_negative': max_cusum_neg
            }
        except:
            return {'cusum_positive': 0.0, 'cusum_negative': 0.0}
    
    def _variance_change_indicators(self, series: np.ndarray) -> Dict[str, float]:
        """Varyans değişiklik göstergeleri"""
        try:
            if len(series) < 10:
                return {'variance_ratio': 1.0}
                
            # Split series in half
            mid = len(series) // 2
            first_half = series[:mid]
            second_half = series[mid:]
            
            var1 = np.var(first_half)
            var2 = np.var(second_half)
            
            if var1 == 0:
                ratio = 1.0 if var2 == 0 else 10.0
            else:
                ratio = var2 / var1
                
            return {'variance_ratio': ratio}
        except:
            return {'variance_ratio': 1.0}
    
    def _trend_change_indicators(self, series: np.ndarray) -> Dict[str, float]:
        """Trend değişiklik göstergeleri"""
        try:
            if len(series) < 6:
                return {'trend_change_strength': 0.0}
                
            # Calculate trends in different parts
            mid = len(series) // 2
            first_half = series[:mid]
            second_half = series[mid:]
            
            # Linear regression slopes
            x1 = np.arange(len(first_half))
            x2 = np.arange(len(second_half))
            
            slope1, _, _, _, _ = stats.linregress(x1, first_half)
            slope2, _, _, _, _ = stats.linregress(x2, second_half)
            
            # Trend change strength
            trend_change = abs(slope2 - slope1) / (np.std(series) + 1e-6)
            
            return {'trend_change_strength': trend_change}
        except:
            return {'trend_change_strength': 0.0}
    
    def _ruptures_change_detection(self, series: np.ndarray) -> Dict[str, float]:
        """Ruptures library ile gelişmiş değişiklik noktası tespiti"""
        try:
            # Use Pelt algorithm for change point detection
            algo = rpt.Pelt(model="rbf").fit(series.reshape(-1, 1))
            change_points = algo.predict(pen=10)
            
            # Calculate probability of recent change
            recent_window = min(20, len(series) // 4)
            recent_change = any(cp > len(series) - recent_window for cp in change_points[:-1])
            
            return {
                'recent_change_prob': 1.0 if recent_change else 0.0,
                'total_change_points': len(change_points) - 1  # Exclude end point
            }
        except:
            return {
                'recent_change_prob': 0.0,
                'total_change_points': 0
            }
    
    def extract_all_advanced_features(self, series: np.ndarray, 
                                    window_sizes: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Tüm gelişmiş istatistiksel özellikleri çıkarır
        """
        if window_sizes is None:
            window_sizes = self.window_sizes
            
        all_features = {}
        
        # Her pencere boyutu için özellikleri hesapla
        for window in window_sizes:
            if len(series) < window:
                continue
                
            # Son window kadar veriyi al
            window_series = series[-window:]
            
            # Hurst exponent
            hurst = self.compute_hurst_exponent(window_series)
            all_features[f'hurst_exponent_w{window}'] = hurst
            
            # Fractal dimensions
            fractal_features = self.compute_fractal_dimension(window_series)
            for key, value in fractal_features.items():
                all_features[f'{key}_w{window}'] = value
            
            # RQA features (only for larger windows)
            if window >= 20:
                rqa_features = self.compute_rqa_features(window_series)
                for key, value in rqa_features.items():
                    all_features[f'{key}_w{window}'] = value
            
            # Entropy features
            entropy_features = self.compute_entropy_features(window_series)
            for key, value in entropy_features.items():
                all_features[f'{key}_w{window}'] = value
            
            # Regime change indicators (only for larger windows)
            if window >= 10:
                regime_features = self.compute_regime_change_indicators(window_series)
                for key, value in regime_features.items():
                    all_features[f'{key}_w{window}'] = value
        
        # Clean any NaN or inf values
        for key, value in all_features.items():
            if np.isnan(value) or np.isinf(value):
                all_features[key] = 0.0
                
        return all_features


def extract_advanced_statistical_features(values: List[float], 
                                         window_sizes: List[int] = [10, 20, 50]) -> np.ndarray:
    """
    Ana fonksiyon: Gelişmiş istatistiksel özellikleri çıkarır
    
    Args:
        values: Zaman serisi değerleri
        window_sizes: Analiz pencere boyutları
        
    Returns:
        Feature matrix (n_samples, n_features)
    """
    print("Gelişmiş istatistiksel özellikler çıkarılıyor...")
    
    extractor = AdvancedStatisticalFeatures(window_sizes)
    series = np.array(values)
    n_samples = len(series)
    
    # Her zaman noktası için özellikleri hesapla
    all_sample_features = []
    
    for i in range(n_samples):
        if i < min(window_sizes):
            # İlk birkaç sample için sıfır features
            sample_features = {}
            for window in window_sizes:
                sample_features.update({
                    f'hurst_exponent_w{window}': 0.5,
                    f'higuchi_fd_w{window}': 1.5,
                    f'katz_fd_w{window}': 1.5,
                    f'petrosian_fd_w{window}': 1.5,
                    f'shannon_entropy_w{window}': 0.0,
                    f'approximate_entropy_w{window}': 0.0,
                    f'sample_entropy_w{window}': 0.0,
                    f'permutation_entropy_w{window}': 0.0,
                    f'spectral_entropy_w{window}': 0.0,
                })
                if window >= 20:
                    sample_features.update({
                        f'rqa_recurrence_rate_w{window}': 0.0,
                        f'rqa_determinism_w{window}': 0.0,
                        f'rqa_entropy_w{window}': 0.0,
                        f'rqa_laminarity_w{window}': 0.0,
                    })
                if window >= 10:
                    sample_features.update({
                        f'cusum_positive_w{window}': 0.0,
                        f'cusum_negative_w{window}': 0.0,
                        f'variance_ratio_w{window}': 1.0,
                        f'trend_change_strength_w{window}': 0.0,
                        f'recent_change_prob_w{window}': 0.0,
                    })
        else:
            # Mevcut pozisyona kadar olan veriyi al
            current_series = series[:i+1]
            sample_features = extractor.extract_all_advanced_features(current_series, window_sizes)
        
        all_sample_features.append(sample_features)
    
    # Feature matrix'e dönüştür
    if all_sample_features:
        # Tüm feature isimlerini topla
        all_feature_names = set()
        for features in all_sample_features:
            all_feature_names.update(features.keys())
        
        feature_names = sorted(list(all_feature_names))
        n_features = len(feature_names)
        
        # Feature matrix oluştur
        feature_matrix = np.zeros((n_samples, n_features))
        
        for i, features in enumerate(all_sample_features):
            for j, feature_name in enumerate(feature_names):
                feature_matrix[i, j] = features.get(feature_name, 0.0)
        
        print(f"✅ Gelişmiş istatistiksel özellikler çıkarıldı: {n_features} özellik")
        return feature_matrix
    else:
        print("❌ Gelişmiş özellik çıkarımında hata!")
        return np.zeros((n_samples, 1))
