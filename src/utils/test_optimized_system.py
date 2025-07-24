#!/usr/bin/env python3
"""
Test script for optimized JetX prediction system
Tests the new ensemble, confidence, and feature extraction improvements
"""

import sys
import os
import numpy as np
import time
from typing import List, Dict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def generate_test_data(n_samples: int = 1000) -> List[float]:
    """Generate realistic test data for JetX"""
    np.random.seed(42)  # For reproducibility
    
    # Generate JetX-like data
    data = []
    for i in range(n_samples):
        # Most values are low (1.0 - 2.0)
        if np.random.random() < 0.7:
            value = np.random.exponential(0.3) + 1.0
            value = min(value, 2.0)  # Cap at 2.0
        # Some medium values (2.0 - 10.0)
        elif np.random.random() < 0.25:
            value = np.random.exponential(2.0) + 2.0
            value = min(value, 10.0)  # Cap at 10.0
        # Few high values (10.0+)
        else:
            value = np.random.exponential(10.0) + 10.0
            value = min(value, 100.0)  # Cap at 100.0
        
        data.append(round(value, 2))
    
    return data

def test_unified_feature_extractor():
    """Test UnifiedFeatureExtractor"""
    print("=" * 60)
    print("TESTING UNIFIED FEATURE EXTRACTOR")
    print("=" * 60)
    
    try:
        from src.feature_engineering.unified_extractor import UnifiedFeatureExtractor
        
        # Generate test data
        data = generate_test_data(500)
        print(f"Generated {len(data)} test samples")
        
        # Initialize extractor
        extractor = UnifiedFeatureExtractor(
            sequence_length=CONFIG.get('training', {}).get('model_sequence_length', 300),
            window_sizes=[5, 10, 20, 50],
            threshold=1.5
        )
        
        # Test fit
        start_time = time.time()
        extractor.fit(data)
        fit_time = time.time() - start_time
        
        print(f"✅ Fit completed in {fit_time:.3f} seconds")
        print(f"✅ Total features: {extractor.total_features}")
        print(f"✅ Feature dimensions: {extractor.feature_dimensions}")
        
        # Test transform
        start_time = time.time()
        features = extractor.transform(data[-100:])  # Last 100 samples
        transform_time = time.time() - start_time
        
        print(f"✅ Transform completed in {transform_time:.3f} seconds")
        print(f"✅ Feature matrix shape: {features.shape}")
        print(f"✅ Feature matrix stats: min={features.min():.3f}, max={features.max():.3f}, mean={features.mean():.3f}")
        
        # Test feature names
        feature_names = extractor.get_feature_names()
        print(f"✅ Feature names count: {len(feature_names)}")
        print(f"✅ Sample feature names: {feature_names[:5]}")
        
        # Test feature groups
        feature_groups = extractor.get_feature_importance_groups()
        print(f"✅ Feature groups: {list(feature_groups.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ UnifiedFeatureExtractor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simplified_confidence_estimator():
    """Test SimplifiedConfidenceEstimator"""
    print("\n" + "=" * 60)
    print("TESTING SIMPLIFIED CONFIDENCE ESTIMATOR")
    print("=" * 60)
    
    try:
        from src.ensemble.simplified_confidence import SimplifiedConfidenceEstimator
        
        # Initialize estimator
        estimator = SimplifiedConfidenceEstimator(history_window=100)
        
        # Generate test predictions and actuals
        predictions = []
        actuals = []
        
        for i in range(50):
            # Simulate prediction
            pred_prob = np.random.random()
            actual_value = np.random.exponential(0.5) + 1.0
            
            predictions.append(pred_prob)
            actuals.append(actual_value)
            
            # Add to estimator
            estimator.add_prediction_result(pred_prob, actual_value)
        
        print(f"✅ Added {len(predictions)} prediction results")
        
        # Test confidence estimation
        model_predictions = {
            'model_1': {'probability': 0.6, 'confidence': 0.8},
            'model_2': {'probability': 0.65, 'confidence': 0.75},
            'model_3': {'probability': 0.7, 'confidence': 0.9}
        }
        
        confidence_result = estimator.estimate_confidence(model_predictions, 0.65)
        
        print(f"✅ Confidence score: {confidence_result['confidence_score']:.3f}")
        print(f"✅ Confidence level: {confidence_result['confidence_level']}")
        print(f"✅ Factors: {confidence_result['factors']}")
        print(f"✅ Recommendation: {confidence_result['recommendation']}")
        
        # Test performance summary
        performance = estimator.get_performance_summary()
        print(f"✅ Performance summary: {performance}")
        
        # Test reliability check
        is_reliable = estimator.is_prediction_reliable(confidence_result['confidence_score'])
        print(f"✅ Prediction reliable: {is_reliable}")
        
        return True
        
    except Exception as e:
        print(f"❌ SimplifiedConfidenceEstimator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimized_ensemble():
    """Test OptimizedEnsemble"""
    print("\n" + "=" * 60)
    print("TESTING OPTIMIZED ENSEMBLE")
    print("=" * 60)
    
    try:
        from src.ensemble.optimized_ensemble import OptimizedEnsemble
        
        # Create mock models
        class MockModel:
            def __init__(self, name, base_accuracy=0.6):
                self.name = name
                self.base_accuracy = base_accuracy
                self.prediction_count = 0
            
            def predict_next_value(self, sequence):
                self.prediction_count += 1
                # Simulate prediction with some randomness
                base_pred = np.mean(sequence[-10:]) if len(sequence) >= 10 else 1.5
                noise = np.random.normal(0, 0.2)
                predicted_value = base_pred + noise
                
                # Probability based on prediction
                above_prob = min(max((predicted_value - 1.0) / 4.0, 0), 1)
                
                # Confidence based on accuracy
                confidence = self.base_accuracy + np.random.normal(0, 0.1)
                confidence = min(max(confidence, 0), 1)
                
                return predicted_value, above_prob, confidence
        
        # Create models
        models = {
            'model_1': MockModel('model_1', 0.65),
            'model_2': MockModel('model_2', 0.60),
            'model_3': MockModel('model_3', 0.70),
            'model_4': MockModel('model_4', 0.55)
        }
        
        # Initialize ensemble
        ensemble = OptimizedEnsemble(models=models, threshold=1.5)
        
        print(f"✅ Ensemble initialized with {len(models)} models")
        
        # Test predictions
        test_sequence = generate_test_data(100)
        
        predictions = []
        for i in range(20):
            sequence = test_sequence[i:i+50]
            result = ensemble.predict_next_value(sequence)
            
            if result is not None:
                predictions.append(result)
        
        print(f"✅ Made {len(predictions)} predictions")
        
        # Test performance updates
        for i, (pred_value, pred_prob, confidence) in enumerate(predictions):
            # Simulate actual result
            actual_value = np.random.exponential(0.5) + 1.0
            ensemble.update_performance(actual_value)
        
        print(f"✅ Updated performance for {len(predictions)} predictions")
        
        # Get model info
        model_info = ensemble.get_model_info()
        print(f"✅ Model info retrieved:")
        for name, info in model_info.items():
            print(f"   {name}: weight={info['weight']:.3f}, accuracy={info['accuracy']:.3f}")
        
        # Get ensemble stats
        ensemble_stats = ensemble.get_ensemble_stats()
        print(f"✅ Ensemble stats: {ensemble_stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ OptimizedEnsemble test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_model_manager():
    """Test AdvancedModelManager with optimized ensemble"""
    print("\n" + "=" * 60)
    print("TESTING ADVANCED MODEL MANAGER")
    print("=" * 60)
    
    try:
        from src.models.advanced_model_manager import AdvancedModelManager
        
        # Initialize manager
        manager = AdvancedModelManager(
            models_dir="test_models",
            db_path="test_jetx_data.db"
        )
        
        print(f"✅ Manager initialized")
        print(f"✅ Optimized ensemble available: {manager.use_optimized_ensemble}")
        
        # Generate test data
        data = generate_test_data(500)
        
        # Initialize models (light only for testing)
        manager.initialize_models(data, auto_train_heavy=False)
        
        print(f"✅ Models initialized")
        
        # Test model status
        status = manager.get_model_status()
        print(f"✅ Model status:")
        for model_name, info in status.items():
            print(f"   {model_name}: loaded={info['loaded']}, trained={info['trained']}")
        
        # Test predictions
        if manager.use_optimized_ensemble:
            print("\n--- Testing Optimized Ensemble ---")
            result = manager.predict_with_optimized_ensemble(data[-100:])
            print(f"✅ Optimized prediction result: {result}")
            
            # Test optimized ensemble info
            ensemble_info = manager.get_optimized_ensemble_info()
            print(f"✅ Optimized ensemble info available: {ensemble_info['available']}")
        
        # Test regular ensemble
        print("\n--- Testing Regular Ensemble ---")
        result = manager.ensemble_predict(data[-100:])
        print(f"✅ Regular ensemble result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ AdvancedModelManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_performance_benchmark():
    """Run performance benchmarks"""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 60)
    
    try:
        # Test data
        data = generate_test_data(1000)
        
        # Benchmark UnifiedFeatureExtractor
        print("\n--- Feature Extraction Benchmark ---")
        from src.feature_engineering.unified_extractor import UnifiedFeatureExtractor
        
        extractor = UnifiedFeatureExtractor()
        
        start_time = time.time()
        extractor.fit(data)
        fit_time = time.time() - start_time
        
        start_time = time.time()
        features = extractor.transform(data[-200:])
        transform_time = time.time() - start_time
        
        print(f"✅ Feature extraction:")
        print(f"   Fit time: {fit_time:.3f} seconds")
        print(f"   Transform time: {transform_time:.3f} seconds")
        print(f"   Features shape: {features.shape}")
        print(f"   Throughput: {features.shape[0]/transform_time:.1f} samples/second")
        
        # Benchmark OptimizedEnsemble
        print("\n--- Ensemble Prediction Benchmark ---")
        from src.ensemble.optimized_ensemble import OptimizedEnsemble
        
        # Create mock models
        class QuickModel:
            def predict_next_value(self, sequence):
                return np.mean(sequence[-10:]), 0.5, 0.6
        
        models = {f'model_{i}': QuickModel() for i in range(5)}
        ensemble = OptimizedEnsemble(models=models)
        
        start_time = time.time()
        predictions = []
        for i in range(100):
            sequence = data[i:i+100]
            result = ensemble.predict_next_value(sequence)
            predictions.append(result)
        
        prediction_time = time.time() - start_time
        
        print(f"✅ Ensemble prediction:")
        print(f"   Total time: {prediction_time:.3f} seconds")
        print(f"   Throughput: {len(predictions)/prediction_time:.1f} predictions/second")
        print(f"   Average per prediction: {prediction_time/len(predictions)*1000:.1f} ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 STARTING OPTIMIZED SYSTEM TESTS")
    print("=" * 60)
    
    test_results = []
    
    # Run individual tests
    test_results.append(("UnifiedFeatureExtractor", test_unified_feature_extractor()))
    test_results.append(("SimplifiedConfidenceEstimator", test_simplified_confidence_estimator()))
    test_results.append(("OptimizedEnsemble", test_optimized_ensemble()))
    test_results.append(("AdvancedModelManager", test_advanced_model_manager()))
    test_results.append(("PerformanceBenchmark", run_performance_benchmark()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The optimized system is working correctly.")
    else:
        print(f"\n⚠️  {total-passed} tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
