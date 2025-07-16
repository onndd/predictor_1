import numpy as np
import math
from collections import deque
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimplifiedConfidenceEstimator:
    """
    Simplified Confidence Estimator for JetX Prediction
    
    Focus on 3 key factors:
    1. Model Performance (how accurate models are)
    2. Prediction Certainty (how confident models are)
    3. Model Agreement (how much models agree)
    
    This is much simpler than the previous 7-factor system
    """
    
    def __init__(self, history_window=200):
        """
        Initialize simplified confidence estimator
        
        Args:
            history_window: Window size for performance tracking
        """
        self.history_window = history_window
        
        # Core tracking data
        self.prediction_history = deque(maxlen=history_window)
        self.actual_history = deque(maxlen=history_window)
        self.accuracy_history = deque(maxlen=history_window)
        
        # Performance metrics
        self.total_predictions = 0
        self.correct_predictions = 0
        self.running_accuracy = 0.5
        
        # Model agreement tracking
        self.agreement_history = deque(maxlen=history_window)
        
        # Confidence calibration
        self.confidence_bins = {
            'low': {'correct': 0, 'total': 0},      # 0.0-0.5
            'medium': {'correct': 0, 'total': 0},   # 0.5-0.75
            'high': {'correct': 0, 'total': 0}      # 0.75-1.0
        }
        
        print("SimplifiedConfidenceEstimator initialized")

    def add_prediction_result(self, prediction, actual_value, model_predictions=None):
        """
        Add a prediction result for confidence calibration
        
        Args:
            prediction: Predicted value or probability
            actual_value: Actual JetX value
            model_predictions: Dictionary of individual model predictions
        """
        # Store prediction and actual
        self.prediction_history.append(prediction)
        self.actual_history.append(actual_value)
        
        # Calculate if prediction was correct (threshold-based)
        threshold = 1.5
        predicted_above = prediction >= threshold if isinstance(prediction, (int, float)) else prediction > 0.5
        actual_above = actual_value >= threshold
        is_correct = predicted_above == actual_above
        
        # Update accuracy tracking
        self.accuracy_history.append(1 if is_correct else 0)
        self.total_predictions += 1
        
        if is_correct:
            self.correct_predictions += 1
        
        # Update running accuracy with exponential smoothing
        if self.total_predictions == 1:
            self.running_accuracy = 1.0 if is_correct else 0.0
        else:
            alpha = 0.1  # Smoothing factor
            self.running_accuracy = alpha * (1 if is_correct else 0) + (1 - alpha) * self.running_accuracy
        
        # Track model agreement if available
        if model_predictions and len(model_predictions) > 1:
            self._track_model_agreement(model_predictions)

    def _track_model_agreement(self, model_predictions):
        """Track how much models agree with each other"""
        if not model_predictions:
            return
        
        # Get probabilities from all models
        probabilities = []
        for model_name, pred_info in model_predictions.items():
            if isinstance(pred_info, dict) and 'probability' in pred_info:
                probabilities.append(pred_info['probability'])
            elif isinstance(pred_info, (int, float)):
                probabilities.append(pred_info)
        
        if len(probabilities) < 2:
            return
        
        # Calculate agreement as inverse of standard deviation
        prob_std = np.std(probabilities)
        agreement = max(0.0, 1.0 - prob_std * 2)  # Higher std = lower agreement
        
        self.agreement_history.append(agreement)

    def estimate_confidence(self, model_predictions, ensemble_probability):
        """
        Estimate confidence based on simplified 3-factor system
        
        Args:
            model_predictions: Dictionary of individual model predictions
            ensemble_probability: Final ensemble probability
            
        Returns:
            dict: Confidence analysis
        """
        # Factor 1: Model Performance (40% weight)
        performance_factor = self._calculate_performance_factor()
        
        # Factor 2: Prediction Certainty (35% weight)
        certainty_factor = self._calculate_certainty_factor(ensemble_probability)
        
        # Factor 3: Model Agreement (25% weight)
        agreement_factor = self._calculate_agreement_factor(model_predictions)
        
        # Calculate weighted confidence
        confidence_score = (
            performance_factor * 0.40 +
            certainty_factor * 0.35 +
            agreement_factor * 0.25
        )
        
        # Ensure confidence is in valid range
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        # Determine confidence level
        confidence_level = self._get_confidence_level(confidence_score)
        
        # Update confidence calibration
        self._update_confidence_calibration(confidence_score, ensemble_probability)
        
        return {
            'confidence_score': confidence_score,
            'confidence_level': confidence_level,
            'factors': {
                'performance': performance_factor,
                'certainty': certainty_factor,
                'agreement': agreement_factor
            },
            'factor_weights': {
                'performance': 0.40,
                'certainty': 0.35,
                'agreement': 0.25
            },
            'recommendation': self._get_recommendation(confidence_score),
            'calibration_info': self._get_calibration_info()
        }

    def _calculate_performance_factor(self):
        """Calculate performance factor based on recent accuracy"""
        if len(self.accuracy_history) < 10:
            return 0.5  # Default when insufficient data
        
        # Recent accuracy (last 50 predictions)
        recent_window = min(50, len(self.accuracy_history))
        recent_accuracy = np.mean(list(self.accuracy_history)[-recent_window:])
        
        # Overall accuracy with more weight on recent
        overall_accuracy = self.running_accuracy
        
        # Combine recent and overall with more weight on recent
        performance_factor = 0.7 * recent_accuracy + 0.3 * overall_accuracy
        
        return max(0.0, min(1.0, performance_factor))

    def _calculate_certainty_factor(self, ensemble_probability):
        """Calculate certainty factor based on how far prediction is from 0.5"""
        if ensemble_probability is None:
            return 0.0
        
        # Distance from 0.5 (uncertainty point)
        distance_from_uncertainty = abs(ensemble_probability - 0.5)
        
        # Convert to certainty (0.5 distance = 1.0 certainty)
        certainty_factor = distance_from_uncertainty * 2.0
        
        return max(0.0, min(1.0, certainty_factor))

    def _calculate_agreement_factor(self, model_predictions):
        """Calculate agreement factor based on model consensus"""
        if not model_predictions or len(model_predictions) < 2:
            return 0.5  # Default when insufficient models
        
        # Get current agreement from model predictions
        probabilities = []
        for model_name, pred_info in model_predictions.items():
            if isinstance(pred_info, dict) and 'probability' in pred_info:
                probabilities.append(pred_info['probability'])
            elif isinstance(pred_info, (int, float)):
                probabilities.append(pred_info)
        
        if len(probabilities) < 2:
            return 0.5
        
        # Calculate current agreement
        prob_std = np.std(probabilities)
        current_agreement = max(0.0, 1.0 - prob_std * 2)
        
        # Historical agreement
        if len(self.agreement_history) > 0:
            historical_agreement = np.mean(self.agreement_history)
            # Combine current and historical
            agreement_factor = 0.6 * current_agreement + 0.4 * historical_agreement
        else:
            agreement_factor = current_agreement
        
        return max(0.0, min(1.0, agreement_factor))

    def _get_confidence_level(self, confidence_score):
        """Convert confidence score to descriptive level"""
        if confidence_score >= 0.8:
            return "Very High"
        elif confidence_score >= 0.65:
            return "High"
        elif confidence_score >= 0.5:
            return "Medium"
        elif confidence_score >= 0.35:
            return "Low"
        else:
            return "Very Low"

    def _get_recommendation(self, confidence_score):
        """Get recommendation based on confidence score"""
        if confidence_score >= 0.8:
            return "✅ High confidence - Safe to use"
        elif confidence_score >= 0.65:
            return "⚠️ Good confidence - Use with caution"
        elif confidence_score >= 0.5:
            return "⚠️ Medium confidence - Consider carefully"
        elif confidence_score >= 0.35:
            return "❌ Low confidence - Use with extreme caution"
        else:
            return "❌ Very low confidence - Not recommended"

    def _update_confidence_calibration(self, confidence_score, ensemble_probability):
        """Update confidence calibration bins"""
        if confidence_score < 0.5:
            bin_name = 'low'
        elif confidence_score < 0.75:
            bin_name = 'medium'
        else:
            bin_name = 'high'
        
        self.confidence_bins[bin_name]['total'] += 1
        
        # This will be updated when we get the actual result
        # For now, we just track the total

    def update_calibration_result(self, confidence_score, was_correct):
        """Update calibration with actual result"""
        if confidence_score < 0.5:
            bin_name = 'low'
        elif confidence_score < 0.75:
            bin_name = 'medium'
        else:
            bin_name = 'high'
        
        if was_correct:
            self.confidence_bins[bin_name]['correct'] += 1

    def _get_calibration_info(self):
        """Get calibration information"""
        calibration_info = {}
        
        for bin_name, bin_data in self.confidence_bins.items():
            if bin_data['total'] > 0:
                accuracy = bin_data['correct'] / bin_data['total']
                calibration_info[bin_name] = {
                    'accuracy': accuracy,
                    'total': bin_data['total'],
                    'correct': bin_data['correct']
                }
            else:
                calibration_info[bin_name] = {
                    'accuracy': 0.0,
                    'total': 0,
                    'correct': 0
                }
        
        return calibration_info

    def get_performance_summary(self):
        """Get performance summary"""
        if self.total_predictions == 0:
            return {
                'total_predictions': 0,
                'accuracy': 0.0,
                'recent_accuracy': 0.0,
                'average_agreement': 0.0
            }
        
        overall_accuracy = self.correct_predictions / self.total_predictions
        recent_accuracy = np.mean(list(self.accuracy_history)[-50:]) if len(self.accuracy_history) >= 50 else overall_accuracy
        average_agreement = np.mean(self.agreement_history) if self.agreement_history else 0.0
        
        return {
            'total_predictions': self.total_predictions,
            'accuracy': overall_accuracy,
            'recent_accuracy': recent_accuracy,
            'running_accuracy': self.running_accuracy,
            'average_agreement': average_agreement,
            'calibration': self._get_calibration_info()
        }

    def reset_performance(self):
        """Reset all performance tracking"""
        self.prediction_history.clear()
        self.actual_history.clear()
        self.accuracy_history.clear()
        self.agreement_history.clear()
        
        self.total_predictions = 0
        self.correct_predictions = 0
        self.running_accuracy = 0.5
        
        # Reset calibration
        for bin_name in self.confidence_bins:
            self.confidence_bins[bin_name] = {'correct': 0, 'total': 0}

    def save_confidence_state(self, filepath):
        """Save confidence state to file"""
        import pickle
        
        state = {
            'prediction_history': list(self.prediction_history),
            'actual_history': list(self.actual_history),
            'accuracy_history': list(self.accuracy_history),
            'agreement_history': list(self.agreement_history),
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'running_accuracy': self.running_accuracy,
            'confidence_bins': self.confidence_bins
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_confidence_state(self, filepath):
        """Load confidence state from file"""
        import pickle
        import os
        
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.prediction_history = deque(state['prediction_history'], maxlen=self.history_window)
            self.actual_history = deque(state['actual_history'], maxlen=self.history_window)
            self.accuracy_history = deque(state['accuracy_history'], maxlen=self.history_window)
            self.agreement_history = deque(state['agreement_history'], maxlen=self.history_window)
            self.total_predictions = state['total_predictions']
            self.correct_predictions = state['correct_predictions']
            self.running_accuracy = state['running_accuracy']
            self.confidence_bins = state['confidence_bins']
            
            return True
            
        except Exception as e:
            print(f"Error loading confidence state: {e}")
            return False

    def get_confidence_trend(self, window=20):
        """Get confidence trend over recent predictions"""
        if len(self.accuracy_history) < window:
            return "Insufficient data"
        
        recent_accuracy = list(self.accuracy_history)[-window:]
        
        # Calculate trend
        x = np.arange(len(recent_accuracy))
        y = np.array(recent_accuracy)
        
        # Simple linear regression
        slope, _ = np.polyfit(x, y, 1)
        
        if slope > 0.01:
            return "Improving"
        elif slope < -0.01:
            return "Declining"
        else:
            return "Stable"

    def is_prediction_reliable(self, confidence_score, min_confidence=0.5):
        """
        Check if prediction is reliable based on confidence score
        
        Args:
            confidence_score: Confidence score from estimate_confidence
            min_confidence: Minimum acceptable confidence
            
        Returns:
            bool: True if prediction is reliable
        """
        # Basic confidence check
        if confidence_score < min_confidence:
            return False
        
        # Check if we have enough data
        if self.total_predictions < 10:
            return False
        
        # Check if recent performance is acceptable
        recent_accuracy = self._calculate_performance_factor()
        if recent_accuracy < 0.45:
            return False
        
        return True

    def get_dynamic_threshold(self):
        """
        Get dynamic confidence threshold based on recent performance
        
        Returns:
            float: Recommended confidence threshold
        """
        if self.total_predictions < 20:
            return 0.6  # Conservative threshold for new system
        
        recent_accuracy = self._calculate_performance_factor()
        
        if recent_accuracy >= 0.7:
            return 0.4  # Lower threshold for good performance
        elif recent_accuracy >= 0.6:
            return 0.5  # Normal threshold
        elif recent_accuracy >= 0.5:
            return 0.6  # Higher threshold for poor performance
        else:
            return 0.7  # Very high threshold for very poor performance


# Backward compatibility
ConfidenceEstimator = SimplifiedConfidenceEstimator
