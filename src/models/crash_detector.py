"""
Advanced Crash Detector for JetX Prediction System
Enhanced with multiple algorithms and sophisticated features
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class AdvancedCrashDetector:
    def __init__(self, crash_threshold=1.5, lookback_period=20,
                 very_low_value_threshold=1.05, low_avg_threshold=1.20,
                 sustained_high_streak_threshold=1.80, streak_length_for_sustained_high=4):
        self.crash_threshold = crash_threshold
        self.lookback_period = lookback_period
        self.very_low_value_threshold = very_low_value_threshold
        self.low_avg_threshold = low_avg_threshold
        self.sustained_high_streak_threshold = sustained_high_streak_threshold
        self.streak_length_for_sustained_high = streak_length_for_sustained_high
        
        # Advanced feature names
        self.feature_names = [
            # Basic features
            'avg_last_n', 'count_very_low', 'all_above_sustained_high', 'is_decreasing_sharply',
            # Statistical features
            'std_last_n', 'max_last_n', 'min_last_n', 'median_last_n', 'range_last_n',
            # Pattern features
            'consecutive_high_count', 'consecutive_low_count', 'oscillation_count',
            # Trend features
            'trend_slope', 'trend_acceleration', 'momentum_5', 'momentum_10',
            # Volatility features
            'volatility_5', 'volatility_10', 'volatility_ratio', 'price_change_velocity',
            # Threshold features
            'above_15_ratio', 'above_20_ratio', 'above_25_ratio', 'above_30_ratio',
            # Advanced pattern features
            'pump_and_dump_score', 'crash_pattern_score', 'recovery_pattern_score'
        ]
        
        self.X_train = []
        self.y_train = []
        self.scaler = StandardScaler()
        
        # Multiple models for ensemble
        self.models = {
            'logistic_regression': LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            'svm': SVC(probability=True, class_weight='balanced', random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        }
        
        self.is_ml_model_fitted = False
        self.ensemble_weights = {}
        print(f"AdvancedCrashDetector başlatıldı: Geriye bakış={self.lookback_period}, Özellik sayısı: {len(self.feature_names)}")

    def _extract_features(self, sequence):
        """Advanced feature extraction for crash detection"""
        if len(sequence) < self.lookback_period:
            # Return default features
            return np.array([1.5] + [0.0] * (len(self.feature_names) - 1))
        
        relevant_sequence = np.array(sequence[-self.lookback_period:])
        features = {}
        
        # Basic features
        features['avg_last_n'] = np.mean(relevant_sequence)
        features['count_very_low'] = np.sum(relevant_sequence < self.very_low_value_threshold)
        features['all_above_sustained_high'] = 1 if np.all(relevant_sequence > self.sustained_high_streak_threshold) else 0
        features['is_decreasing_sharply'] = self._calculate_sharp_decrease(relevant_sequence)
        
        # Statistical features
        features['std_last_n'] = np.std(relevant_sequence)
        features['max_last_n'] = np.max(relevant_sequence)
        features['min_last_n'] = np.min(relevant_sequence)
        features['median_last_n'] = np.median(relevant_sequence)
        features['range_last_n'] = np.max(relevant_sequence) - np.min(relevant_sequence)
        
        # Pattern features
        features['consecutive_high_count'] = self._count_consecutive_high(relevant_sequence)
        features['consecutive_low_count'] = self._count_consecutive_low(relevant_sequence)
        features['oscillation_count'] = self._count_oscillations(relevant_sequence)
        
        # Trend features
        features['trend_slope'] = self._calculate_trend_slope(relevant_sequence)
        features['trend_acceleration'] = self._calculate_trend_acceleration(relevant_sequence)
        features['momentum_5'] = self._calculate_momentum(relevant_sequence, 5)
        features['momentum_10'] = self._calculate_momentum(relevant_sequence, 10)
        
        # Volatility features
        features['volatility_5'] = self._calculate_volatility(relevant_sequence, 5)
        features['volatility_10'] = self._calculate_volatility(relevant_sequence, 10)
        features['volatility_ratio'] = features['volatility_5'] / (features['volatility_10'] + 1e-8)
        features['price_change_velocity'] = self._calculate_price_change_velocity(relevant_sequence)
        
        # Threshold features
        features['above_15_ratio'] = np.mean(relevant_sequence >= 1.5)
        features['above_20_ratio'] = np.mean(relevant_sequence >= 2.0)
        features['above_25_ratio'] = np.mean(relevant_sequence >= 2.5)
        features['above_30_ratio'] = np.mean(relevant_sequence >= 3.0)
        
        # Advanced pattern features
        features['pump_and_dump_score'] = self._calculate_pump_and_dump_score(relevant_sequence)
        features['crash_pattern_score'] = self._calculate_crash_pattern_score(relevant_sequence)
        features['recovery_pattern_score'] = self._calculate_recovery_pattern_score(relevant_sequence)
        
        # Convert to array
        features_array = np.array([features.get(name, 0.0) for name in self.feature_names])
        
        # Handle NaN values
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features_array
    
    def _calculate_sharp_decrease(self, sequence):
        """Calculate if there's a sharp decrease"""
        if len(sequence) < 3:
            return 0
        
        # Check for sharp decrease in last 3 values
        if (sequence[-1] < sequence[-2] < sequence[-3] and
            (sequence[-3] - sequence[-1]) > 0.3):
            return 1
        
        # Check for any sharp decrease in the sequence
        for i in range(2, len(sequence)):
            if (sequence[i] < sequence[i-1] < sequence[i-2] and
                (sequence[i-2] - sequence[i]) > 0.5):
                return 1
        
        return 0
    
    def _count_consecutive_high(self, sequence):
        """Count consecutive high values"""
        count = 0
        max_count = 0
        for val in sequence:
            if val >= self.sustained_high_streak_threshold:
                count += 1
                max_count = max(max_count, count)
            else:
                count = 0
        return max_count
    
    def _count_consecutive_low(self, sequence):
        """Count consecutive low values"""
        count = 0
        max_count = 0
        for val in sequence:
            if val < self.crash_threshold:
                count += 1
                max_count = max(max_count, count)
            else:
                count = 0
        return max_count
    
    def _count_oscillations(self, sequence):
        """Count oscillations in the sequence"""
        if len(sequence) < 3:
            return 0
        
        oscillations = 0
        for i in range(1, len(sequence) - 1):
            if ((sequence[i] > sequence[i-1] and sequence[i] > sequence[i+1]) or
                (sequence[i] < sequence[i-1] and sequence[i] < sequence[i+1])):
                oscillations += 1
        
        return oscillations
    
    def _calculate_trend_slope(self, sequence):
        """Calculate trend slope using linear regression"""
        if len(sequence) < 2:
            return 0
        
        x = np.arange(len(sequence))
        y = sequence
        
        # Simple linear regression
        n = len(x)
        if n * np.sum(x**2) - np.sum(x)**2 == 0:
            return 0
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        
        return slope
    
    def _calculate_trend_acceleration(self, sequence):
        """Calculate trend acceleration (second derivative)"""
        if len(sequence) < 3:
            return 0
        
        # Calculate differences
        first_diff = np.diff(sequence)
        second_diff = np.diff(first_diff)
        
        return np.mean(second_diff)
    
    def _calculate_momentum(self, sequence, window):
        """Calculate momentum over a window"""
        if len(sequence) < window:
            return 0
        
        recent = sequence[-window:]
        return (recent[-1] - recent[0]) / recent[0] if recent[0] != 0 else 0
    
    def _calculate_volatility(self, sequence, window):
        """Calculate volatility over a window"""
        if len(sequence) < window:
            return np.std(sequence)
        
        recent = sequence[-window:]
        returns = np.diff(recent) / recent[:-1]
        returns = returns[~np.isnan(returns)]  # Remove NaN values
        
        return np.std(returns) if len(returns) > 0 else 0
    
    def _calculate_price_change_velocity(self, sequence):
        """Calculate velocity of price changes"""
        if len(sequence) < 2:
            return 0
        
        changes = np.abs(np.diff(sequence))
        return np.mean(changes)
    
    def _calculate_pump_and_dump_score(self, sequence):
        """Calculate pump and dump pattern score"""
        if len(sequence) < 10:
            return 0
        
        # Look for rapid increase followed by rapid decrease
        max_idx = np.argmax(sequence)
        
        if max_idx < 3 or max_idx > len(sequence) - 3:
            return 0
        
        # Check for rapid increase to max
        before_max = sequence[:max_idx]
        if len(before_max) >= 3:
            increase_rate = (sequence[max_idx] - before_max[-3]) / 3
        else:
            increase_rate = 0
        
        # Check for rapid decrease after max
        after_max = sequence[max_idx:]
        if len(after_max) >= 3:
            decrease_rate = (after_max[0] - after_max[2]) / 2
        else:
            decrease_rate = 0
        
        # Score based on both increase and decrease rates
        if increase_rate > 0.3 and decrease_rate > 0.3:
            return min(1.0, (increase_rate + decrease_rate) / 2)
        
        return 0
    
    def _calculate_crash_pattern_score(self, sequence):
        """Calculate crash pattern score"""
        if len(sequence) < 5:
            return 0
        
        # Look for high values followed by crash
        recent = sequence[-5:]
        max_recent = np.max(recent)
        min_recent = np.min(recent)
        
        if max_recent > 2.0 and min_recent < self.crash_threshold:
            crash_magnitude = max_recent - min_recent
            return min(1.0, crash_magnitude / 3.0)
        
        return 0
    
    def _calculate_recovery_pattern_score(self, sequence):
        """Calculate recovery pattern score"""
        if len(sequence) < 5:
            return 0
        
        # Look for crash followed by recovery
        recent = sequence[-5:]
        min_idx = np.argmin(recent)
        
        if min_idx < len(recent) - 2:  # Not at the end
            recovery = recent[-1] - recent[min_idx]
            if recovery > 0.2:
                return min(1.0, recovery / 0.5)
        
        return 0

    def fit(self, historical_values):
        """Train the ensemble of models"""
        print(f"AdvancedCrashDetector, {len(historical_values)} adet geçmiş değer ile eğitime hazırlanıyor...")
        self.X_train = []
        self.y_train = []
        self.is_ml_model_fitted = False
        
        if len(historical_values) <= self.lookback_period:
            print("AdvancedCrashDetector 'fit' metodu için yeterli geçmiş veri yok.")
            return
        
        # Extract features and labels
        for i in range(len(historical_values) - self.lookback_period):
            current_sequence_for_features = historical_values[i : i + self.lookback_period]
            features = self._extract_features(current_sequence_for_features)
            actual_next_value = historical_values[i + self.lookback_period]
            label = 1 if actual_next_value < self.crash_threshold else 0
            self.X_train.append(features)
            self.y_train.append(label)
        
        if not self.X_train:
            print("AdvancedCrashDetector için hiç özellik örneği oluşturulamadı.")
            return
        
        X_np = np.array(self.X_train)
        y_np = np.array(self.y_train)
        X_shuffled, y_shuffled = shuffle(X_np, y_np, random_state=42)
        
        min_samples_for_training = 50
        if len(X_shuffled) < min_samples_for_training:
            print(f"AdvancedCrashDetector eğitimi için yeterli örnek yok ({len(X_shuffled)}/{min_samples_for_training}).")
            return
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_shuffled)
        
        # Train each model and calculate weights
        total_accuracy = 0
        model_accuracies = {}
        
        # Split for validation
        split_idx = int(len(X_scaled) * 0.8)
        X_train_split = X_scaled[:split_idx]
        X_val_split = X_scaled[split_idx:]
        y_train_split = y_shuffled[:split_idx]
        y_val_split = y_shuffled[split_idx:]
        
        print(f"  -> {len(self.models)} model ile ensemble eğitimi başlıyor...")
        
        for model_name, model in self.models.items():
            try:
                print(f"    -> {model_name} eğitiliyor...")
                model.fit(X_train_split, y_train_split)
                
                # Calculate validation accuracy
                y_pred = model.predict(X_val_split)
                accuracy = accuracy_score(y_val_split, y_pred)
                model_accuracies[model_name] = accuracy
                total_accuracy += accuracy
                
                print(f"      -> {model_name} doğruluk: {accuracy:.3f}")
                
            except Exception as e:
                print(f"      -> {model_name} eğitimi başarısız: {e}")
                model_accuracies[model_name] = 0.0
        
        # Calculate ensemble weights based on accuracy
        if total_accuracy > 0:
            for model_name in self.models:
                self.ensemble_weights[model_name] = model_accuracies[model_name] / total_accuracy
        else:
            # Equal weights as fallback
            weight = 1.0 / len(self.models)
            for model_name in self.models:
                self.ensemble_weights[model_name] = weight
        
        self.is_ml_model_fitted = True
        print(f"  -> AdvancedCrashDetector ensemble eğitimi tamamlandı!")
        print(f"  -> Ensemble ağırlıkları: {self.ensemble_weights}")

    def predict_crash_risk(self, current_sequence):
        """Predict crash risk using ensemble of models"""
        if not self.is_ml_model_fitted:
            return 0.1
        
        if len(current_sequence) < self.lookback_period:
            return 0.1
        
        features_for_prediction = self._extract_features(current_sequence)
        if features_for_prediction.ndim == 1:
            features_for_prediction = features_for_prediction.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features_for_prediction)
        
        try:
            # Get predictions from all models
            ensemble_prediction = 0.0
            total_weight = 0.0
            
            for model_name, model in self.models.items():
                try:
                    probabilities = model.predict_proba(features_scaled)
                    crash_probability = probabilities[0, 1]
                    weight = self.ensemble_weights.get(model_name, 0.0)
                    
                    ensemble_prediction += crash_probability * weight
                    total_weight += weight
                    
                except Exception as e:
                    print(f"Model {model_name} prediction failed: {e}")
                    continue
            
            # Normalize if needed
            if total_weight > 0:
                ensemble_prediction = ensemble_prediction / total_weight
            else:
                ensemble_prediction = 0.1
            
            return min(1.0, max(0.0, ensemble_prediction))
            
        except Exception as e:
            print(f"AdvancedCrashDetector ensemble prediction error: {e}")
            return 0.1

    def get_feature_importance(self):
        """Get feature importance from random forest model"""
        if not self.is_ml_model_fitted:
            return {}
        
        try:
            rf_model = self.models['random_forest']
            if hasattr(rf_model, 'feature_importances_'):
                importances = rf_model.feature_importances_
                return dict(zip(self.feature_names, importances))
        except Exception as e:
            print(f"Feature importance extraction failed: {e}")
        
        return {}

    def get_model_performance(self):
        """Get model performance metrics"""
        return {
            'ensemble_weights': self.ensemble_weights,
            'feature_count': len(self.feature_names),
            'is_fitted': self.is_ml_model_fitted
        }


# Backward compatibility
class CrashDetector(AdvancedCrashDetector):
    """Backward compatible crash detector"""
    def __init__(self, crash_threshold=1.5, lookback_period=5,
                 very_low_value_threshold=1.05, low_avg_threshold=1.20,
                 sustained_high_streak_threshold=1.80, streak_length_for_sustained_high=4):
        # Use minimum of 20 for advanced features
        super().__init__(crash_threshold, max(lookback_period, 20), very_low_value_threshold, 
                        low_avg_threshold, sustained_high_streak_threshold, streak_length_for_sustained_high)
