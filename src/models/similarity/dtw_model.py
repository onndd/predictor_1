import numpy as np
from collections import defaultdict
from scipy.spatial.distance import euclidean

try:
    from fastdtw import fastdtw
    FASTDTW_AVAILABLE = True
except ImportError:
    FASTDTW_AVAILABLE = False
    print("Warning: fastdtw not available. Using basic DTW implementation.")

class DTWModel:
    def __init__(self, window_size=10, max_neighbors=5, threshold=1.5, max_distance=0.5):
        """
        Dynamic Time Warping modeli
        
        Args:
            window_size: Karşılaştırılacak dizi uzunluğu
            max_neighbors: Maksimum komşu sayısı
            threshold: Eşik değeri
            max_distance: Maksimum DTW mesafesi
        """
        self.window_size = window_size
        self.max_neighbors = max_neighbors
        self.threshold = threshold
        self.max_distance = max_distance
        self.sequences = []
        self.next_values = []
        
    def fit(self, values):
        """
        Modeli eğitir
        
        Args:
            values: Değerler dizisi
        """
        if len(values) <= self.window_size:
            return
            
        # Diziler oluştur
        self.sequences = []
        self.next_values = []
        
        for i in range(len(values) - self.window_size):
            seq = values[i:i+self.window_size]
            next_val = values[i+self.window_size]
            
            self.sequences.append(seq)
            self.next_values.append(next_val)
    
    def _basic_dtw_distance(self, seq1, seq2):
        """
        Basic DTW distance implementation as fallback
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            DTW distance
        """
        n, m = len(seq1), len(seq2)
        
        # Initialize DTW matrix
        dtw_matrix = np.full((n + 1, m + 1), float('inf'))
        dtw_matrix[0, 0] = 0
        
        # Fill the matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(seq1[i-1] - seq2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # insertion
                    dtw_matrix[i, j-1],      # deletion
                    dtw_matrix[i-1, j-1]     # match
                )
        
        return dtw_matrix[n, m]
    
    def _get_nearest_sequences(self, query_sequence, top_n=None):
        """
        En yakın dizileri bulur
        
        Args:
            query_sequence: Sorgu dizisi
            top_n: Döndürülecek maksimum sonuç sayısı
            
        Returns:
            list: (mesafe, indeks, sonraki değer) üçlülerinin listesi
        """
        if not self.sequences:
            return []
            
        if top_n is None:
            top_n = self.max_neighbors
            
        # DTW mesafelerini hesapla
        distances = []
        
        for i, seq in enumerate(self.sequences):
            try:
                if FASTDTW_AVAILABLE:
                    # FastDTW mesafesini hesapla
                    distance, _ = fastdtw(query_sequence, seq, dist=euclidean)
                else:
                    # Basic DTW implementation
                    distance = self._basic_dtw_distance(query_sequence, seq)
                
                # Normalize et - sequence length'e göre
                distance /= max(len(query_sequence), len(seq))
                
                if distance <= self.max_distance:
                    distances.append((distance, i, self.next_values[i]))
            except Exception as e:
                # Hata durumunda euclidean distance kullan
                if len(query_sequence) == len(seq):
                    distance = euclidean(query_sequence, seq)
                    distance /= len(query_sequence)
                    if distance <= self.max_distance:
                        distances.append((distance, i, self.next_values[i]))
        
        # Mesafeye göre sırala
        distances.sort(key=lambda x: x[0])
        
        # En yakın top_n sonucu döndür
        return distances[:top_n]
    
    def predict_next_value(self, sequence):
        """
        Bir sonraki değeri tahmin eder
        
        Args:
            sequence: Değerler dizisi
            
        Returns:
            tuple: (tahmini değer, eşik üstü olasılığı)
        """
        if not self.sequences or len(sequence) < self.window_size:
            return None, 0.5
            
        # Son window_size değeri al
        query = sequence[-self.window_size:]
        
        # En yakın dizileri bul
        nearest = self._get_nearest_sequences(query)
        
        if not nearest:
            return None, 0.5
            
        # Ağırlıklı tahmin
        total_weight = 0
        weighted_sum = 0
        above_weight = 0
        
        for distance, _, next_value in nearest:
            # Uzaklık tersini ağırlık olarak kullan
            weight = 1.0 / (distance + 1e-5)
            total_weight += weight
            weighted_sum += next_value * weight
            
            # Eşik üstü ağırlık
            if next_value >= self.threshold:
                above_weight += weight
        
        if total_weight == 0:
            return None, 0.5
            
        # Ağırlıklı ortalama
        prediction = weighted_sum / total_weight
        
        # Eşik üstü olasılığı
        above_prob = above_weight / total_weight
        
        return prediction, above_prob
