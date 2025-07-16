import numpy as np
import math
from collections import defaultdict, deque
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class OptimizedEnsemble:
    """
    Optimized Ensemble System for JetX Prediction
    
    Improvements:
    - Simplified weight update mechanism
    - Better confidence estimation
    - Performance-based model selection
    - Memory efficient tracking
    - Robust error handling
    """
    
    def __init__(self, models=None, threshold=1.5, performance_window=100, max_model_errors: int = 5):
        """
        Initialize optimized ensemble
        
        Args:
            models: Dictionary of model instances
            threshold: Decision threshold
            performance_window: Window size for performance tracking
            max_model_errors: Number of errors before deactivating a model
        """
        self.models = models or {}
        self.threshold = threshold
        self.performance_window = performance_window
        self.max_model_errors = max_model_errors
        
        # Optimized weight system
        self.weights = {}
        self.performance_tracker = {}
        
        # Efficient data structures
        self.prediction_history = deque(maxlen=performance_window)
        self.accuracy_history = deque(maxlen=performance_window)
        self.confidence_history = deque(maxlen=performance_window)
        
        # Model metadata
        self.model_stats = {}
        self.last_update = datetime.now()
        
        # Performance thresholds
        self.min_accuracy = 0.45  # Minimum acceptable accuracy
        self.confidence_threshold = 0.4  # Minimum confidence for prediction
        
        # Initialize models
        self._initialize_models()
        
        print(f"OptimizedEnsemble initialized with {len(self.models)} models")

    def _initialize_models(self):
        """Initialize model weights and performance tracking"""
        for model_name in self.models:
            # Equal initial weights
            self.weights[model_name] = 1.0 / len(self.models) if self.models else 1.0
            
            # Performance tracking
            self.performance_tracker[model_name] = {
                'correct': 0,
                'total': 0,
                'accuracy': 0.5,
                'confidence_sum': 0.0,
                'avg_confidence': 0.5,
                'recent_correct': deque(maxlen=50),
                'last_updated': datetime.now()
            }
            
            # Model stats
            self.model_stats[model_name] = {
                'active': True,
                'error_count': 0,
                'last_error': None,
                'prediction_count': 0
            }

    def add_model(self, name, model, initial_weight=None):
        """
        Add a new model to the ensemble
        
        Args:
            name: Model name
            model: Model instance
            initial_weight: Initial weight (optional)
        """
        self.models[name] = model
        
        # Calculate initial weight
        if initial_weight is None:
            initial_weight = 1.0 / len(self.models)
        
        self.weights[name] = initial_weight
        
        # Initialize tracking
        self.performance_tracker[name] = {
            'correct': 0,
            'total': 0,
            'accuracy': 0.5,
            'confidence_sum': 0.0,
            'avg_confidence': 0.5,
            'recent_correct': deque(maxlen=50),
            'last_updated': datetime.now()
        }
        
        self.model_stats[name] = {
            'active': True,
            'error_count': 0,
            'last_error': None,
            'prediction_count': 0
        }
        
        # Renormalize weights
        self._normalize_weights()

    def predict_next_value(self, sequence):
        """
        Make ensemble prediction with improved logic
        
        Args:
            sequence: Input sequence
            
        Returns:
            tuple: (predicted_value, above_threshold_probability, confidence_score)
        """
        if not self.models:
            return None, 0.5, 0.0
        
        # Get predictions from all active models
        model_predictions = self._get_model_predictions(sequence)
        
        if not model_predictions:
            return None, 0.5, 0.0
        
        # Calculate ensemble results
        ensemble_result = self._calculate_ensemble_result(model_predictions)
        
        # Store prediction for tracking
        self._store_prediction_info(model_predictions, ensemble_result)
        
        return ensemble_result

    def _get_model_predictions(self, sequence):
        """Get predictions from all active models"""
        model_predictions = {}
        
        for model_name, model in self.models.items():
            if not self.model_stats[model_name]['active']:
                continue
                
            try:
                # Get prediction from model
                result = self._safe_model_predict(model, sequence)
                
                if result is not None:
                    indicative_value, above_prob, confidence = result
                    
                    # Validate prediction
                    if self._validate_prediction(indicative_value, above_prob, confidence):
                        model_predictions[model_name] = {
                            'value': indicative_value,
                            'probability': above_prob,
                            'confidence': confidence,
                            'weight': self.weights[model_name]
                        }
                        
                        # Update model stats
                        self.model_stats[model_name]['prediction_count'] += 1
                    
            except Exception as e:
                self._handle_model_error(model_name, e)
        
        return model_predictions

    def _safe_model_predict(self, model, sequence):
        """Safely get prediction from a model"""
        try:
            if hasattr(model, 'predict_next_value'):
                result = model.predict_next_value(sequence)
                
                # Standardize result format
                if isinstance(result, tuple):
                    if len(result) == 2:
                        return result + (0.5,)  # Add default confidence
                    elif len(result) == 3:
                        return result
                    else:
                        return None
                        
            elif hasattr(model, 'predict'):
                # For sklearn-like models
                result = model.predict(sequence)
                return result, 0.5, 0.5  # Default values
                
        except Exception as e:
            print(f"Model prediction error for {model.__class__.__name__}: {e}")
            return None

    def _validate_prediction(self, indicative_value, probability, confidence):
        """Validate prediction values"""
        # Check for NaN or inf
        if any(math.isnan(x) or math.isinf(x) for x in [indicative_value or 0, probability, confidence]):
            return False
            
        # Check probability bounds
        if not (0 <= probability <= 1):
            return False
            
        # Check confidence bounds
        if not (0 <= confidence <= 1):
            return False
            
        # Check value reasonableness
        if indicative_value is not None and (indicative_value < 0 or indicative_value > 1000):
            return False
            
        return True

    def _calculate_ensemble_result(self, model_predictions):
        """Calculate ensemble result with improved logic"""
        if not model_predictions:
            return None, 0.5, 0.0
        
        # Separate values and probabilities
        indicative_values = []
        probabilities = []
        confidences = []
        weights = []
        
        for model_name, pred in model_predictions.items():
            if pred['value'] is not None:
                indicative_values.append(pred['value'])
                probabilities.append(pred['probability'])
                confidences.append(pred['confidence'])
                
                # Weight adjusted by confidence and model performance
                model_perf = self.performance_tracker[model_name]['accuracy']
                adjusted_weight = pred['weight'] * pred['confidence'] * model_perf
                weights.append(adjusted_weight)
        
        if not indicative_values:
            return None, 0.5, 0.0
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(indicative_values)] * len(indicative_values)
        
        # Calculate weighted predictions
        ensemble_indicative_value = sum(v * w for v, w in zip(indicative_values, weights))
        above_prob = sum(p * w for p, w in zip(probabilities, weights))
        
        # Calculate ensemble confidence
        ensemble_confidence = self._calculate_ensemble_confidence(
            model_predictions, weights, above_prob
        )
        
        return ensemble_indicative_value, above_prob, ensemble_confidence

    def _calculate_ensemble_confidence(self, model_predictions, weights, above_prob):
        """Calculate ensemble confidence score"""
        if not model_predictions:
            return 0.0
        
        # Agreement factor (how much models agree)
        probabilities = [pred['probability'] for pred in model_predictions.values()]
        prob_std = np.std(probabilities) if len(probabilities) > 1 else 0.0
        agreement_factor = max(0.0, float(1.0 - prob_std * 2))
        
        # Confidence factor (average model confidence)
        confidences = [pred['confidence'] for pred in model_predictions.values()]
        confidence_factor = sum(c * w for c, w in zip(confidences, weights))
        
        # Certainty factor (how far from 0.5 is the probability)
        certainty_factor = 2 * abs(above_prob - 0.5)
        
        # Model count factor (more models = more confidence)
        model_count_factor = min(1.0, len(model_predictions) / 3.0)
        
        # Performance factor (how well models perform recently)
        performance_factor = np.mean([
            self.performance_tracker[name]['accuracy'] 
            for name in model_predictions.keys()
        ])
        
        # Combined confidence
        ensemble_confidence = (
            agreement_factor * 0.3 +
            confidence_factor * 0.25 +
            certainty_factor * 0.2 +
            model_count_factor * 0.15 +
            performance_factor * 0.1
        )
        
        return max(0.0, min(1.0, ensemble_confidence))

    def update_performance(self, actual_value, prediction_id=None):
        """
        Update model performance with actual result
        
        Args:
            actual_value: Actual JetX value
            prediction_id: Optional prediction ID for tracking
        """
        if not self.prediction_history:
            return
        
        # Get last prediction
        last_prediction = self.prediction_history[-1]
        
        # Update performance for each model
        for model_name in last_prediction['models']:
            model_pred = last_prediction['models'][model_name]
            
            # Check if prediction was correct
            predicted_above = model_pred['probability'] > 0.5
            actual_above = actual_value >= self.threshold
            is_correct = predicted_above == actual_above
            
            # Update performance tracker
            tracker = self.performance_tracker[model_name]
            tracker['total'] += 1
            
            if is_correct:
                tracker['correct'] += 1
                tracker['recent_correct'].append(1)
            else:
                tracker['recent_correct'].append(0)
            
            # Update accuracy
            tracker['accuracy'] = tracker['correct'] / tracker['total']
            
            # Update confidence tracking
            tracker['confidence_sum'] += model_pred['confidence']
            tracker['avg_confidence'] = tracker['confidence_sum'] / tracker['total']
            
            # Update last update time
            tracker['last_updated'] = datetime.now()
        
        # Update ensemble performance
        ensemble_pred = last_prediction['ensemble']
        ensemble_above = ensemble_pred['probability'] > 0.5
        actual_above = actual_value >= self.threshold
        ensemble_correct = ensemble_above == actual_above
        
        self.accuracy_history.append(1 if ensemble_correct else 0)
        
        # Update model weights
        self._update_model_weights(actual_value, last_prediction)
        
        # Deactivate poorly performing models
        self._manage_model_activity()

    def _update_model_weights(self, actual_value, last_prediction):
        """Update model weights based on performance"""
        actual_above = actual_value >= self.threshold
        
        for model_name in last_prediction['models']:
            model_pred = last_prediction['models'][model_name]
            predicted_above = model_pred['probability'] > 0.5
            is_correct = predicted_above == actual_above
            
            # Get current performance
            current_accuracy = self.performance_tracker[model_name]['accuracy']
            
            # Adaptive learning rate
            if is_correct:
                # Lower learning rate for already good models to avoid overfitting weights
                adaptive_lr = 0.05 * (1.0 - current_accuracy)
            else:
                # Higher learning rate for poor models to correct them faster
                adaptive_lr = 0.1 * (1.0 - current_accuracy)

            # Weight adjustment
            if is_correct:
                # Reward correct predictions
                reward = adaptive_lr
                self.weights[model_name] += reward
            else:
                # Penalize incorrect predictions
                penalty = adaptive_lr
                self.weights[model_name] -= penalty
            
            # Keep weights positive
            self.weights[model_name] = max(0.01, self.weights[model_name])
        
        # Normalize weights
        self._normalize_weights()

    def _normalize_weights(self):
        """Normalize weights to sum to 1"""
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for model_name in self.weights:
                self.weights[model_name] /= total_weight

    def _manage_model_activity(self):
        """Manage model activity based on performance"""
        for model_name in self.model_stats:
            tracker = self.performance_tracker[model_name]
            
            # Deactivate models with poor performance
            if tracker['total'] >= 20:
                recent_accuracy = np.mean(list(tracker['recent_correct']))
                
                if recent_accuracy < self.min_accuracy:
                    self.model_stats[model_name]['active'] = False
                    print(f"Model {model_name} deactivated due to poor performance ({recent_accuracy:.3f})")
                elif not self.model_stats[model_name]['active'] and recent_accuracy > 0.6:
                    self.model_stats[model_name]['active'] = True
                    print(f"Model {model_name} reactivated due to improved performance ({recent_accuracy:.3f})")

    def _handle_model_error(self, model_name, error):
        """Handle model errors"""
        self.model_stats[model_name]['error_count'] += 1
        self.model_stats[model_name]['last_error'] = str(error)
        
        # Deactivate model after too many errors
        if self.model_stats[model_name]['error_count'] >= self.max_model_errors:
            self.model_stats[model_name]['active'] = False
            print(f"Model {model_name} deactivated due to repeated errors")

    def _store_prediction_info(self, model_predictions, ensemble_result):
        """Store prediction information for tracking"""
        prediction_info = {
            'timestamp': datetime.now(),
            'models': model_predictions,
            'ensemble': {
                'value': ensemble_result[0],
                'probability': ensemble_result[1],
                'confidence': ensemble_result[2]
            }
        }
        
        self.prediction_history.append(prediction_info)
        self.confidence_history.append(ensemble_result[2])

    def get_model_info(self):
        """Get comprehensive model information"""
        info = {}
        
        for model_name in self.models:
            tracker = self.performance_tracker[model_name]
            stats = self.model_stats[model_name]
            
            info[model_name] = {
                'weight': self.weights[model_name],
                'accuracy': tracker['accuracy'],
                'recent_accuracy': np.mean(list(tracker['recent_correct'])) if tracker['recent_correct'] else 0.0,
                'avg_confidence': tracker['avg_confidence'],
                'total_predictions': tracker['total'],
                'correct_predictions': tracker['correct'],
                'active': stats['active'],
                'error_count': stats['error_count'],
                'prediction_count': stats['prediction_count']
            }
        
        return info

    def get_ensemble_stats(self):
        """Get ensemble performance statistics"""
        if not self.accuracy_history:
            return {
                'accuracy': 0.0,
                'confidence': 0.0,
                'total_predictions': 0,
                'active_models': 0
            }
        
        return {
            'accuracy': np.mean(self.accuracy_history),
            'recent_accuracy': np.mean(list(self.accuracy_history)[-20:]) if len(self.accuracy_history) >= 20 else np.mean(self.accuracy_history),
            'confidence': np.mean(self.confidence_history),
            'total_predictions': len(self.accuracy_history),
            'active_models': sum(1 for stats in self.model_stats.values() if stats['active'])
        }

    def reset_performance(self):
        """Reset performance tracking"""
        for model_name in self.performance_tracker:
            self.performance_tracker[model_name] = {
                'correct': 0,
                'total': 0,
                'accuracy': 0.5,
                'confidence_sum': 0.0,
                'avg_confidence': 0.5,
                'recent_correct': deque(maxlen=50),
                'last_updated': datetime.now()
            }
        
        self.accuracy_history.clear()
        self.confidence_history.clear()
        self.prediction_history.clear()

    def save_ensemble_state(self, filepath):
        """Save ensemble state to file"""
        import pickle
        
        state = {
            'weights': self.weights,
            'performance_tracker': {
                name: {
                    'correct': tracker['correct'],
                    'total': tracker['total'],
                    'accuracy': tracker['accuracy'],
                    'confidence_sum': tracker['confidence_sum'],
                    'avg_confidence': tracker['avg_confidence'],
                    'recent_correct': list(tracker['recent_correct']),
                    'last_updated': tracker['last_updated']
                } for name, tracker in self.performance_tracker.items()
            },
            'model_stats': self.model_stats,
            'accuracy_history': list(self.accuracy_history),
            'confidence_history': list(self.confidence_history),
            'last_update': self.last_update
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_ensemble_state(self, filepath):
        """Load ensemble state from file"""
        import pickle
        import os
        
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.weights = state['weights']
            self.model_stats = state['model_stats']
            self.accuracy_history = deque(state['accuracy_history'], maxlen=self.performance_window)
            self.confidence_history = deque(state['confidence_history'], maxlen=self.performance_window)
            self.last_update = state['last_update']
            
            # Reconstruct performance tracker
            for name, tracker_data in state['performance_tracker'].items():
                self.performance_tracker[name] = {
                    'correct': tracker_data['correct'],
                    'total': tracker_data['total'],
                    'accuracy': tracker_data['accuracy'],
                    'confidence_sum': tracker_data['confidence_sum'],
                    'avg_confidence': tracker_data['avg_confidence'],
                    'recent_correct': deque(tracker_data['recent_correct'], maxlen=50),
                    'last_updated': tracker_data['last_updated']
                }
            
            return True
            
        except Exception as e:
            print(f"Error loading ensemble state: {e}")
            return False
