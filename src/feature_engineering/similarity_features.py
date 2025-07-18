"""
Similarity-Based Feature Extractor for JetX Prediction System

This module calculates features based on the similarity of the current sequence
to historical patterns. It provides valuable context to ML models about how
similar past situations have resolved.
"""

import numpy as np
from typing import List, Tuple

class SimilarityFeatureEncoder:
    """
    Encodes similarity features by comparing a sequence to historical data.
    """
    def __init__(self, tolerance: float = 0.1, min_similarity_threshold: float = 0.8, max_neighbors: int = 10):
        """
        Initializes the similarity feature encoder.

        Args:
            tolerance (float): The tolerance for considering values in patterns similar.
            min_similarity_threshold (float): The minimum similarity score for a pattern to be considered a match.
            max_neighbors (int): The maximum number of similar historical patterns to consider.
        """
        self.tolerance = tolerance
        self.min_similarity_threshold = min_similarity_threshold
        self.max_neighbors = max_neighbors
        self.historical_sequences: List[np.ndarray] = []
        self.historical_next_values: List[float] = []
        self.is_fitted = False

    def fit(self, values: List[float], sequence_length: int = 100):
        """
        Prepares the historical data for similarity search.

        Args:
            values (List[float]): The entire historical dataset.
            sequence_length (int): The length of patterns to compare.
        """
        print("Fitting SimilarityFeatureEncoder: Preparing historical patterns...")
        self.historical_sequences = []
        self.historical_next_values = []

        if len(values) <= sequence_length:
            print("⚠️ Not enough historical data to fit SimilarityFeatureEncoder.")
            return

        for i in range(len(values) - sequence_length):
            self.historical_sequences.append(np.array(values[i:i + sequence_length]))
            self.historical_next_values.append(values[i + sequence_length])
        
        self.is_fitted = True
        print(f"✅ SimilarityFeatureEncoder fitted with {len(self.historical_sequences)} historical patterns.")

    def transform(self, values: List[float], sequence_length: int = 100) -> np.ndarray:
        """
        Transforms a list of values into similarity features.

        Args:
            values (List[float]): The input values to generate features for.
            sequence_length (int): The length of patterns to use for comparison.

        Returns:
            np.ndarray: A matrix of similarity features.
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder must be fitted before transforming data.")

        num_samples = len(values)
        # Each sequence will have a set of features derived from its similarity to past data
        num_features = 5 # avg_next_val, std_next_val, avg_similarity, num_neighbors, prob_above_threshold
        features = np.zeros((num_samples, num_features))

        for i in range(num_samples):
            if i < sequence_length:
                continue # Not enough data to form a full sequence

            current_sequence = np.array(values[i - sequence_length:i])
            similar_patterns = self._find_similar_patterns(current_sequence)

            if not similar_patterns:
                continue

            next_values = [p[1] for p in similar_patterns]
            similarities = [p[2] for p in similar_patterns]

            features[i, 0] = np.mean(next_values)
            features[i, 1] = np.std(next_values) if len(next_values) > 1 else 0
            features[i, 2] = np.mean(similarities)
            features[i, 3] = len(similar_patterns)
            features[i, 4] = np.mean([1 if v >= 1.5 else 0 for v in next_values])

        return features

    def _find_similar_patterns(self, current_sequence: np.ndarray) -> List[Tuple[int, float, float]]:
        """
        Finds the most similar historical patterns to the current sequence.

        Args:
            current_sequence (np.ndarray): The sequence to find matches for.

        Returns:
            List[Tuple[int, float, float]]: A list of tuples containing (index, next_value, similarity_score).
        """
        similarities = []
        for i, historical_seq in enumerate(self.historical_sequences):
            similarity_score = self._calculate_similarity(current_sequence, historical_seq)
            if similarity_score >= self.min_similarity_threshold:
                similarities.append((i, self.historical_next_values[i], similarity_score))

        # Sort by similarity and take the top N neighbors
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:self.max_neighbors]

    def _calculate_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """
        Calculates the similarity between two patterns.

        Args:
            pattern1 (np.ndarray): The first pattern.
            pattern2 (np.ndarray): The second pattern.

        Returns:
            float: A similarity score between 0 and 1.
        """
        # Using normalized euclidean distance as a similarity metric
        diff = pattern1 - pattern2
        distance = np.sqrt(np.sum(diff**2)) / len(pattern1)
        
        # Convert distance to similarity (0 distance = 1 similarity)
        similarity = max(0, 1 - distance)
        return similarity