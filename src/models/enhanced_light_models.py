import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
import pickle
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

warnings.filterwarnings('ignore')

class HeavyModelKnowledge:
    """
    A knowledge class to store information extracted from heavy models.
    """
    
    def __init__(self):
        self.pattern_weights = {}
        self.feature_importance = {}
        self.decision_boundaries = {}
        self.performance_metrics = {}
        self.extracted_patterns = {}
        self.threshold_adjustments = {}
        self.confidence_calibration = {}
        self.creation_time = datetime.now()
        
    def add_pattern_weight(self, pattern_name: str, weight: float):
        """Adds a pattern weight."""
        self.pattern_weights[pattern_name] = weight
        
    def add_feature_importance(self, feature_name: str, importance: float):
        """Adds feature importance."""
        self.feature_importance[feature_name] = importance
        
    def add_decision_boundary(self, feature_range: Tuple[float, float], decision: str):
        """Adds a decision boundary."""
        self.decision_boundaries[feature_range] = decision
        
    def add_extracted_pattern(self, pattern_type: str, pattern_data: Dict):
        """Adds an extracted pattern."""
        self.extracted_patterns[pattern_type] = pattern_data
        
    def add_threshold_adjustment(self, condition: str, adjustment: float):
        """Adds a threshold adjustment."""
        self.threshold_adjustments[condition] = adjustment
        
    def get_pattern_weight(self, pattern_name: str) -> float:
        """Gets a pattern weight."""
        return self.pattern_weights.get(pattern_name, 1.0)
        
    def get_feature_importance(self, feature_name: str) -> float:
        """Gets feature importance."""
        return self.feature_importance.get(feature_name, 1.0)
        
    def get_threshold_adjustment(self, sequence: List[float]) -> float:
        """Suggests a threshold adjustment for a given sequence."""
        # Simple example - real implementation will be more complex
        recent_avg = np.mean(sequence[-10:])
        
        for condition, adjustment in self.threshold_adjustments.items():
            if condition == "high_volatility" and np.std(sequence[-10:]) > 0.5:
                return adjustment
            elif condition == "low_values" and recent_avg < 1.3:
                return adjustment
            elif condition == "high_values" and recent_avg > 2.0:
                return adjustment
                
        return 0.0
        
    def get_confidence_boost(self, prediction_prob: float) -> float:
        """Calculates a confidence boost for a prediction."""
        # Confidence boost based on heavy model knowledge
        if prediction_prob > 0.7:
            return 0.15  # More boost for high confidence
        elif prediction_prob > 0.6:
            return 0.10
        elif prediction_prob > 0.4:
            return 0.05
        else:
            return 0.0
            
    def save_knowledge(self, filepath: str):
        """Saves the knowledge to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load_knowledge(cls, filepath: str) -> Optional['HeavyModelKnowledge']:
        """Loads knowledge from a file."""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading knowledge: {e}")
        return None
        
    def is_valid(self) -> bool:
        """Checks if the knowledge is valid."""
        return len(self.pattern_weights) > 0 or len(self.feature_importance) > 0
        
    def get_summary(self) -> Dict[str, Any]:
        """Gets a summary of the knowledge."""
        return {
            'pattern_weights_count': len(self.pattern_weights),
            'feature_importance_count': len(self.feature_importance),
            'decision_boundaries_count': len(self.decision_boundaries),
            'extracted_patterns_count': len(self.extracted_patterns),
            'threshold_adjustments_count': len(self.threshold_adjustments),
            'creation_time': self.creation_time,
            'age_hours': (datetime.now() - self.creation_time).total_seconds() / 3600
        }

def _extract_light_model_features(sequence: List[float], threshold: float = 1.5) -> Tuple[List[float], List[str]]:
    """
    Extracts a standardized set of features and their names for light models.
    This centralized function prevents code duplication and enables knowledge transfer.
    """
    seq = np.array(sequence)
    features: List[Any] = []
    feature_names: List[str] = []

    # Basic statistics
    features.extend([
        np.mean(seq), np.std(seq), np.max(seq), np.min(seq),
        np.median(seq), np.percentile(seq, 75), np.percentile(seq, 25)
    ])
    feature_names.extend([
        'mean', 'std', 'max', 'min', 'median', 'p75', 'p25'
    ])

    # Threshold-based features
    features.extend([
        np.mean(seq >= 1.5), np.mean(seq >= 2.0),
        np.mean(seq >= 3.0), np.mean(seq <= 1.3)
    ])
    feature_names.extend([
        'rate_above_1.5', 'rate_above_2.0', 'rate_above_3.0', 'rate_below_1.3'
    ])

    # Trend features
    if len(seq) > 1:
        diff = np.diff(seq)
        features.extend([
            np.mean(diff), np.std(diff),
            np.mean(diff > 0), np.mean(diff < 0)
        ])
    else:
        features.extend([0, 0, 0, 0])
    feature_names.extend([
        'mean_diff', 'std_diff', 'increase_rate', 'decrease_rate'
    ])

    # Rolling statistics
    for window in [5, 10, 20]:
        if len(seq) >= window:
            rolling_seq = seq[-window:]
            features.extend([
                np.mean(rolling_seq), np.std(rolling_seq),
                np.mean(rolling_seq >= threshold)
            ])
        else:
            features.extend([np.mean(seq), np.std(seq), np.mean(seq >= threshold)])
        feature_names.extend([
            f'rolling_mean_{window}', f'rolling_std_{window}', f'rolling_rate_above_thresh_{window}'
        ])

    # Pattern features
    consecutive_above = 0
    consecutive_below = 0
    current_above = 0
    current_below = 0
    for val in seq[-20:]:
        if val >= threshold:
            current_above += 1
            current_below = 0
            consecutive_above = max(consecutive_above, current_above)
        else:
            current_below += 1
            current_above = 0
            consecutive_below = max(consecutive_below, current_below)
    features.extend([consecutive_above, consecutive_below])
    feature_names.extend(['consecutive_above', 'consecutive_below'])

    # Volatility features
    if len(seq) > 2:
        returns = np.diff(seq) / (seq[:-1] + 1e-8)
        features.extend([np.std(returns), np.mean(np.abs(returns))])
    else:
        features.extend([0, 0])
    feature_names.extend(['volatility', 'mean_abs_return'])

    return features, feature_names


class EnhancedRandomForest:
    """
    Enhanced Random Forest - Optimized for JetX.
    Can be boosted with heavy model knowledge.
    """
    
    def __init__(self, threshold=1.5, n_estimators=300, max_depth=20):
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.heavy_knowledge = None
        self.knowledge_boost_enabled = False
        self.feature_names: List[str] = []
        
    def update_with_heavy_knowledge(self, knowledge: HeavyModelKnowledge):
        """Updates the model with heavy model knowledge."""
        if knowledge and knowledge.is_valid():
            self.heavy_knowledge = knowledge
            self.knowledge_boost_enabled = True
            print("✅ Random Forest boosted with heavy model knowledge")
        else:
            print("⚠️ Invalid heavy model knowledge provided")
        
    def fit(self, sequences, labels):
        """Trains the model."""
        print(f"Starting Enhanced Random Forest training... {len(sequences)} sequences")
        
        # Feature extraction using the centralized function
        X_list = []
        feature_names: List[str] = []
        for seq in sequences:
            features, names = _extract_light_model_features(seq, self.threshold)
            X_list.append(features)
            if not feature_names:  # Store names only once
                feature_names = names
        
        self.feature_names = feature_names
        X = np.array(X_list)
        y = np.array(labels)
        
        # Feature scaling
        X_scaled = self.scaler.fit_transform(X)
        
        # Create model
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Training
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        print("Enhanced Random Forest training completed!")
        return self
    
    def predict_next_value(self, sequence):
        """Makes a prediction, boosted with heavy model knowledge if available."""
        if not self.is_fitted or self.model is None:
            return None, 0.5, 0.0
        
        try:
            features, feature_names = _extract_light_model_features(sequence, self.threshold)
            X = np.array(features).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Base probability prediction
            prob = self.model.predict_proba(X_scaled)[0][1]
            
            # Boost with heavy model knowledge
            if self.knowledge_boost_enabled and self.heavy_knowledge:
                # Threshold adjustment
                threshold_adjustment = self.heavy_knowledge.get_threshold_adjustment(sequence)
                adjusted_threshold = self.threshold + threshold_adjustment
                
                # Feature importance-based probability adjustment
                feature_boost = 0.0
                if feature_names and self.heavy_knowledge.feature_importance:
                    # Create a dictionary for quick lookup
                    feature_dict = dict(zip(feature_names, features))
                    
                    # Get top 5 most important features from knowledge
                    top_features = sorted(
                        self.heavy_knowledge.feature_importance.items(),
                        key=lambda item: item[1],
                        reverse=True
                    )[:5]

                    for feature_key, importance_score in top_features:
                        # Extract the base feature name (e.g., 'rolling_mean_5' from 'enhanced_rf_rolling_mean_5')
                        base_feature_name = "_".join(feature_key.split('_')[2:])
                        
                        if base_feature_name in feature_dict:
                            # Apply a boost based on the feature's value and its importance
                            # This is a simple heuristic and can be improved
                            feature_value = feature_dict[base_feature_name]
                            if feature_value > 0.5: # Example condition
                                feature_boost += 0.01 * importance_score
                            else:
                                feature_boost -= 0.01 * importance_score

                # Apply feature boost
                prob = max(0.0, min(1.0, prob + feature_boost))
                
                # Confidence boost from heavy model
                confidence_boost = self.heavy_knowledge.get_confidence_boost(prob)
                base_confidence = abs(prob - 0.5) * 2
                confidence = min(1.0, base_confidence + confidence_boost)
                
                # Value prediction with heavy model insights
                base_value = np.mean(sequence[-10:])
                if prob > 0.6:
                    # High confidence above threshold
                    value_multiplier = 1.3 + (prob - 0.6) * 0.5
                else:
                    # Lower confidence or below threshold
                    value_multiplier = 0.7 + prob * 0.6
                
                indicative_value = base_value * value_multiplier
                
            else:
                # Standard prediction without heavy model knowledge
                confidence = min(1.0, max(0.0, abs(prob - 0.5) * 2))
                indicative_value = np.mean(sequence[-10:]) * (1.2 if prob > 0.5 else 0.8)
            
            return indicative_value, prob, confidence
            
        except Exception as e:
            print(f"Enhanced RF prediction error: {e}")
            return None, 0.5, 0.0


class EnhancedGradientBoosting:
    """
    Enhanced Gradient Boosting - Powerful pattern recognition.
    """
    
    def __init__(self, threshold=1.5, n_estimators=200, learning_rate=0.1):
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names: List[str] = []
        
    def fit(self, sequences, labels):
        """Trains the model."""
        print(f"Starting Enhanced Gradient Boosting training... {len(sequences)} sequences")
        
        # Feature extraction using the centralized function
        X_list = []
        feature_names: List[str] = []
        for seq in sequences:
            features, names = _extract_light_model_features(seq, self.threshold)
            X_list.append(features)
            if not feature_names:
                feature_names = names
        
        self.feature_names = feature_names
        X = np.array(X_list)
        y = np.array(labels)
        
        # Feature scaling
        X_scaled = self.scaler.fit_transform(X)
        
        # Model
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        )
        
        # Training
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        print("Enhanced Gradient Boosting training completed!")
        return self
    
    def predict_next_value(self, sequence):
        """Makes a prediction."""
        if not self.is_fitted or self.model is None:
            return None, 0.5, 0.0
        
        try:
            features, _ = _extract_light_model_features(sequence, self.threshold)
            X = np.array(features).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            prob = self.model.predict_proba(X_scaled)[0][1]
            confidence = min(1.0, max(0.0, abs(prob - 0.5) * 2.5))  # Boosting confidence
            
            # Value prediction
            indicative_value = np.mean(sequence[-5:]) * (1.3 if prob > 0.6 else 0.7)
            
            return indicative_value, prob, confidence
            
        except Exception as e:
            print(f"Enhanced GB prediction error: {e}")
            return None, 0.5, 0.0


class LightModelEnsemble:
    """
    Ensemble of light models.
    """
    
    def __init__(self, threshold=1.5):
        self.threshold = threshold
        self.models = {}
        self.model_weights = {}
        self.is_fitted = False
        self.heavy_knowledge: Optional[HeavyModelKnowledge] = None
        self.knowledge_boost_enabled = False
        
    def add_model(self, name, model, weight=1.0):
        """Adds a model to the ensemble."""
        self.models[name] = model
        self.model_weights[name] = weight
        
    def update_with_heavy_knowledge(self, knowledge: HeavyModelKnowledge):
        """Propagates heavy model knowledge to all child models."""
        if knowledge and knowledge.is_valid():
            self.heavy_knowledge = knowledge
            self.knowledge_boost_enabled = True
            print("✅ Light Model Ensemble boosted with heavy model knowledge")
            for model in self.models.values():
                if hasattr(model, 'update_with_heavy_knowledge'):
                    model.update_with_heavy_knowledge(knowledge)
        else:
            print("⚠️ Invalid heavy model knowledge for ensemble")

    def fit(self, sequences, labels):
        """Trains all models in the ensemble."""
        print(f"Starting Light Model Ensemble training... {len(self.models)} models")
        
        # Train each model
        for name, model in self.models.items():
            print(f"  -> Training {name}...")
            model.fit(sequences, labels)
        
        self.is_fitted = True
        print("Light Model Ensemble training completed!")
        
    def predict_next_value(self, sequence):
        """Makes an ensemble prediction, boosted with heavy model knowledge if available."""
        if not self.is_fitted:
            return None, 0.5, 0.0
        
        predictions = []
        confidences = []
        values = []
        total_weight = 0
        knowledge_boost_count = 0
        
        for name, model in self.models.items():
            try:
                indicative_val, prob, conf = model.predict_next_value(sequence)
                if prob is not None:
                    weight = self.model_weights[name]
                    
                    # Extra weight for models with heavy model knowledge
                    if hasattr(model, 'knowledge_boost_enabled') and model.knowledge_boost_enabled:
                        weight *= 1.2  # 20% extra weight
                        knowledge_boost_count += 1
                    
                    predictions.append(prob * weight)
                    confidences.append(conf * weight)
                    if indicative_val is not None:
                        values.append(indicative_val * weight)
                    total_weight += weight
            except:
                continue
        
        if total_weight == 0:
            return None, 0.5, 0.0
        
        # Weighted average
        avg_prob = sum(predictions) / total_weight
        avg_confidence = sum(confidences) / total_weight if confidences else 0.0
        avg_indicative_value = sum(values) / len(values) if values else np.mean(sequence[-5:])
        
        # Ensemble-level heavy model knowledge boost
        if self.knowledge_boost_enabled and self.heavy_knowledge:
            # Global confidence boost
            ensemble_confidence_boost = self.heavy_knowledge.get_confidence_boost(avg_prob)
            avg_confidence = min(1.0, avg_confidence + ensemble_confidence_boost * 0.5)
            
            # Ensemble-level pattern adjustment
            recent_avg = np.mean(sequence[-10:])
            if recent_avg > 2.0 and avg_prob > 0.7:
                # High values with high confidence - slight reduction for safety
                avg_prob = max(0.5, avg_prob - 0.05)
            elif recent_avg < 1.3 and avg_prob < 0.3:
                # Low values with low confidence - slight increase
                avg_prob = min(0.5, avg_prob + 0.05)
        
        return avg_indicative_value, avg_prob, avg_confidence


def create_enhanced_light_models(threshold=1.5):
    """
    Creates a dictionary of enhanced light models.
    """
    models = {}
    
    # Enhanced Random Forest
    models['enhanced_rf'] = EnhancedRandomForest(
        threshold=threshold,
        n_estimators=300,
        max_depth=20
    )
    
    # Enhanced Gradient Boosting
    models['enhanced_gb'] = EnhancedGradientBoosting(
        threshold=threshold,
        n_estimators=200,
        learning_rate=0.1
    )
    
    # Extra Trees (Random Forest variant)
    class ExtraTreesWrapper:
        def __init__(self, threshold=1.5):
            self.threshold = threshold
            self.model = ExtraTreesClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            self.scaler = StandardScaler()
            self.is_fitted = False
            
        def fit(self, sequences, labels):
            X = [_extract_light_model_features(seq, self.threshold)[0] for seq in sequences]
            X = self.scaler.fit_transform(X)
            self.model.fit(X, labels)
            self.is_fitted = True
            
        def predict_next_value(self, sequence):
            if not self.is_fitted:
                return None, 0.5, 0.0
            features, _ = _extract_light_model_features(sequence, self.threshold)
            X = self.scaler.transform([features])
            prob = self.model.predict_proba(X)[0][1]
            return None, prob, abs(prob - 0.5) * 2
    
    models['extra_trees'] = ExtraTreesWrapper(threshold)
    
    # SVM (Support Vector Machine)
    class SVMWrapper:
        def __init__(self, threshold=1.5):
            self.threshold = threshold
            self.model = SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
            self.scaler = StandardScaler()
            self.is_fitted = False
            
        def fit(self, sequences, labels):
            X = [_extract_light_model_features(seq, self.threshold)[0] for seq in sequences]
            X = self.scaler.fit_transform(X)
            self.model.fit(X, labels)
            self.is_fitted = True
            
        def predict_next_value(self, sequence):
            if not self.is_fitted:
                return None, 0.5, 0.0
            features, _ = _extract_light_model_features(sequence, self.threshold)
            X = self.scaler.transform([features])
            prob = self.model.predict_proba(X)[0][1]
            return None, prob, abs(prob - 0.5) * 2
    
    models['svm'] = SVMWrapper(threshold)
    
    return models
