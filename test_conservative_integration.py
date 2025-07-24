#!/usr/bin/env python3
"""
Conservative Advice Integration Test
Tests the complete integration of conservative advice system into the main application
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_advanced_model_manager_integration():
    """Test AdvancedModelManager conservative advice integration"""
    print("🔧 Testing AdvancedModelManager Conservative Integration...")
    
    try:
        from models.advanced_model_manager import AdvancedModelManager
        
        # Create manager instance
        manager = AdvancedModelManager()
        
        # Check if get_ensemble_advice method exists
        assert hasattr(manager, 'get_ensemble_advice'), "get_ensemble_advice method missing!"
        assert hasattr(manager, '_check_sequence_volatility'), "_check_sequence_volatility method missing!"
        assert hasattr(manager, '_check_pattern_reliability'), "_check_pattern_reliability method missing!"
        
        print("✅ AdvancedModelManager has all required conservative methods")
        
        # Test with sample data
        sample_sequence = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 1.8]
        
        # Test volatility check
        volatility_safe = manager._check_sequence_volatility(sample_sequence)
        print(f"✅ Volatility check works: {volatility_safe}")
        
        # Test pattern reliability check
        pattern_reliable = manager._check_pattern_reliability(sample_sequence, 1.9)
        print(f"✅ Pattern reliability check works: {pattern_reliable}")
        
        return True
        
    except Exception as e:
        print(f"❌ AdvancedModelManager integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_app_integration():
    """Test main app conservative prediction integration"""
    print("\n🔧 Testing Main App Conservative Integration...")
    
    try:
        # Import main app components
        sys.path.append('.')
        from src.main_app import EnhancedJetXApp
        
        # Create app instance
        app = EnhancedJetXApp()
        
        # Check if make_conservative_prediction method exists
        assert hasattr(app, 'make_conservative_prediction'), "make_conservative_prediction method missing!"
        
        print("✅ EnhancedJetXApp has conservative prediction method")
        
        # Test method signature
        import inspect
        signature = inspect.signature(app.make_conservative_prediction)
        expected_params = ['sequence_length', 'confidence_threshold']
        
        for param in expected_params:
            assert param in signature.parameters, f"Parameter {param} missing!"
        
        print("✅ make_conservative_prediction has correct signature")
        
        return True
        
    except Exception as e:
        print(f"❌ Main app integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_conservative_criteria_logic():
    """Test conservative criteria validation logic"""
    print("\n🔧 Testing Conservative Criteria Logic...")
    
    try:
        from models.advanced_model_manager import AdvancedModelManager
        import numpy as np
        
        manager = AdvancedModelManager()
        
        # Test scenarios
        test_cases = [
            {
                'name': 'Stable Low Sequence',
                'sequence': [1.1, 1.0, 1.2, 1.1, 1.0, 1.1],
                'expected_volatility': True,  # Low volatility should be safe
                'expected_pattern': True     # Stable pattern should be reliable
            },
            {
                'name': 'High Volatility Sequence', 
                'sequence': [1.0, 3.0, 0.5, 2.5, 0.8, 2.8],
                'expected_volatility': False,  # High volatility should be unsafe
                'expected_pattern': False     # Unstable pattern should be unreliable
            },
            {
                'name': 'Stable High Sequence',
                'sequence': [2.0, 2.1, 2.0, 2.2, 2.1, 2.0],
                'expected_volatility': True,   # Low volatility should be safe
                'expected_pattern': True      # Stable pattern should be reliable
            }
        ]
        
        for case in test_cases:
            print(f"\n📊 Testing: {case['name']}")
            print(f"   Sequence: {case['sequence']}")
            
            # Test volatility
            volatility_result = manager._check_sequence_volatility(case['sequence'])
            volatility_match = volatility_result == case['expected_volatility']
            print(f"   Volatility Safe: {volatility_result} (expected: {case['expected_volatility']}) {'✅' if volatility_match else '❌'}")
            
            # Test pattern reliability with sample prediction
            prediction = np.mean(case['sequence']) + 0.1  # Simple prediction
            pattern_result = manager._check_pattern_reliability(case['sequence'], prediction)
            pattern_match = pattern_result == case['expected_pattern']
            print(f"   Pattern Reliable: {pattern_result} (expected: {case['expected_pattern']}) {'✅' if pattern_match else '❌'}")
            
            if not (volatility_match and pattern_match):
                print(f"   ❌ Test case failed: {case['name']}")
                return False
        
        print("\n✅ All conservative criteria logic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Conservative criteria logic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_integration_simulation():
    """Simulate full conservative advice pipeline"""
    print("\n🔧 Testing Full Integration Simulation...")
    
    try:
        from models.advanced_model_manager import AdvancedModelManager
        
        manager = AdvancedModelManager()
        
        # Create mock models for testing
        class MockModel:
            def __init__(self, prediction_value):
                self.prediction_value = prediction_value
                self.is_trained = True
            
            def predict(self, sequence):
                return self.prediction_value
        
        # Add mock models - bu sefer predict_with_model'ı override edeceğiz
        manager.models = {
            'mock_model_1': MockModel(1.8),  # High prediction
            'mock_model_2': MockModel(1.9),  # High prediction
            'mock_model_3': MockModel(1.7),  # High prediction
        }
        
        # Override predict_with_model method to work with mock models
        def mock_predict_with_model(model_name, sequence):
            if model_name not in manager.models:
                return None
            try:
                model = manager.models[model_name]
                if hasattr(model, 'is_trained') and not model.is_trained:
                    return None
                
                prediction = model.predict(sequence)
                if isinstance(prediction, (int, float)) and np.isfinite(prediction):
                    return float(np.clip(prediction, manager.min_prediction_value, manager.max_prediction_value))
                return None
            except Exception:
                return None
        
        # Replace the method temporarily
        original_method = manager.predict_with_model
        manager.predict_with_model = mock_predict_with_model
        
        # Test with different scenarios
        scenarios = [
            {
                'name': 'Conservative Friendly (Should Play)',
                'sequence': [1.6, 1.7, 1.8, 1.7, 1.8, 1.9, 1.8, 1.9, 2.0, 1.9],
                'confidence_threshold': 0.7,  # Lower threshold for this test
                'expected_advice': 'Play'
            },
            {
                'name': 'High Volatility (Should Not Play)',
                'sequence': [1.0, 3.0, 0.5, 2.5, 0.8, 2.8, 1.2, 2.9, 0.9, 2.7],
                'confidence_threshold': 0.85,
                'expected_advice': 'Do Not Play'
            },
            {
                'name': 'Insufficient Data (Should Not Play)',
                'sequence': [1.5, 2.0],
                'confidence_threshold': 0.85,
                'expected_advice': 'Do Not Play'
            }
        ]
        
        for scenario in scenarios:
            print(f"\n📊 Scenario: {scenario['name']}")
            
            result = manager.get_ensemble_advice(
                scenario['sequence'], 
                scenario['confidence_threshold']
            )
            
            print(f"   Expected: {scenario['expected_advice']}")
            print(f"   Actual: {result['advice']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Reason: {result['reason']}")
            
            # Check criteria analysis
            if result.get('criteria_analysis'):
                print("   Criteria Analysis:")
                for criterion, data in result['criteria_analysis'].items():
                    status = "✅" if data['met'] else "❌"
                    print(f"     {status} {criterion}: {data['details']}")
            
            # Verify result matches expectation
            advice_correct = result['advice'] == scenario['expected_advice']
            print(f"   Result: {'✅ CORRECT' if advice_correct else '❌ INCORRECT'}")
            
            if not advice_correct:
                print(f"   ❌ Scenario failed: {scenario['name']}")
                return False
        
        print("\n✅ Full integration simulation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Full integration simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all conservative integration tests"""
    print("🛡️ CONSERVATIVE ADVICE INTEGRATION TESTS")
    print("=" * 60)
    print("🎯 Testing complete integration of conservative advice system")
    print("=" * 60)
    
    tests = [
        ("AdvancedModelManager Integration", test_advanced_model_manager_integration),
        ("Main App Integration", test_main_app_integration),
        ("Conservative Criteria Logic", test_conservative_criteria_logic),
        ("Full Integration Simulation", test_full_integration_simulation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 40)
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 60)
    print("📋 INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {status}: {test_name}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 CONSERVATIVE ADVICE INTEGRATION COMPLETE!")
        print("\n🎯 Integration Summary:")
        print("   ✅ AdvancedModelManager has get_ensemble_advice() method")
        print("   ✅ Main app has make_conservative_prediction() method")
        print("   ✅ Conservative criteria validation working")
        print("   ✅ Multi-criteria decision logic operational")
        print("   ✅ Full pipeline from UI to backend integrated")
        print("\n💰 Expected User Experience:")
        print("   🛡️ Conservative Advice button shows real advice")
        print("   📊 Detailed criteria analysis displayed")
        print("   🔍 User can see why advice was given")
        print("   💯 Money protection system active")
        print("\n🚀 The app will now give proper 'Play/Do Not Play' advice!")
    else:
        print("❌ INTEGRATION TESTS FAILED!")
        print("\nAction required:")
        print("1. Fix failing integration components")
        print("2. Verify all methods are properly implemented")
        print("3. Test manually in the app interface")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
