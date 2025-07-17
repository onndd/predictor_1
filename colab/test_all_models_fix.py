"""
Comprehensive test script for all models in Rolling Training System
Tests N-Beats, TFT, and LSTM models with fixed tensor dimensions and compatibility
"""

import sys
sys.path.append('/content/predictor_1/src')

import numpy as np
import torch

def test_all_models():
    """Test all models with Rolling Training System compatibility"""
    print("🧪 Testing All Models for Rolling Training System...")
    
    # Create test data (similar to JetX data format)
    test_data = []
    for i in range(1000):
        # Generate random JetX-like values with tuple format (id, value)
        value = np.random.uniform(1.0, 3.0)
        test_data.append((i, value))
    
    print(f"📊 Test data created: {len(test_data)} samples")
    print(f"📊 Sample data format: {test_data[:3]}")
    
    # Test parameters
    sequence_length = 300
    epochs = 3  # Small number for quick test
    batch_size = 16
    
    models_to_test = {
        'N-Beats': {
            'module': 'models.deep_learning.n_beats.n_beats_model',
            'class': 'NBeatsPredictor',
            'params': {
                'sequence_length': sequence_length,
                'hidden_size': 128,
                'num_stacks': 2,
                'num_blocks': 2,
                'learning_rate': 0.001,
                'threshold': 1.5
            }
        },
        'TFT': {
            'module': 'models.deep_learning.tft.tft_model',
            'class': 'TFTPredictor',
            'params': {
                'sequence_length': sequence_length,
                'hidden_size': 128,
                'num_heads': 4,
                'num_layers': 1,
                'learning_rate': 0.001
            }
        },
        'LSTM': {
            'module': 'models.sequential.lstm_model',
            'class': 'LSTMModel',
            'params': {
                'seq_length': sequence_length,
                'n_features': 1,
                'threshold': 1.5
            }
        }
    }
    
    results = {}
    
    for model_name, model_config in models_to_test.items():
        print(f"\n{'='*50}")
        print(f"🔬 Testing {model_name} Model")
        print(f"{'='*50}")
        
        try:
            # Import model
            module = __import__(model_config['module'], fromlist=[model_config['class']])
            model_class = getattr(module, model_config['class'])
            
            # Create model
            print(f"✅ Creating {model_name} model...")
            model = model_class(**model_config['params'])
            print(f"✅ {model_name} model created successfully")
            
            # Test data preparation
            print(f"🔧 Testing data preparation...")
            if hasattr(model, 'prepare_sequences'):
                sequences, targets = model.prepare_sequences(test_data)
                print(f"✅ Data preparation successful")
                print(f"📊 Sequences shape: {sequences.shape if hasattr(sequences, 'shape') else type(sequences)}")
                print(f"📊 Targets shape: {targets.shape if hasattr(targets, 'shape') else type(targets)}")
            
            # Test training
            print(f"🚀 Testing training...")
            history = model.train(
                data=test_data,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=True
            )
            print(f"✅ Training completed successfully!")
            
            # Test prediction with confidence
            print(f"🔮 Testing prediction with confidence...")
            test_sequence = [float(item[1]) for item in test_data[sequence_length:sequence_length*2]]
            
            value, prob, conf = model.predict_with_confidence(test_sequence)
            print(f"✅ Prediction successful:")
            print(f"   Value: {value}")
            print(f"   Probability: {prob}")
            print(f"   Confidence: {conf}")
            
            # Test multiple predictions
            print(f"🔮 Testing multiple predictions...")
            for i in range(3):
                start_idx = sequence_length + i * 50
                test_seq = [float(item[1]) for item in test_data[start_idx:start_idx + sequence_length]]
                val, pr, co = model.predict_with_confidence(test_seq)
                print(f"   Prediction {i+1}: value={val:.3f}, prob={pr:.3f}, conf={co:.3f}")
            
            # Test model saving (if available)
            try:
                print(f"💾 Testing model saving...")
                model.save_model(f"/tmp/test_{model_name.lower()}_model.pth")
                print(f"✅ Model saving successful")
            except Exception as save_e:
                print(f"⚠️ Model saving failed: {save_e}")
            
            results[model_name] = {
                'status': 'SUCCESS',
                'training_history': history,
                'sample_prediction': {'value': value, 'probability': prob, 'confidence': conf}
            }
            
            print(f"🎉 {model_name} model test completed successfully!")
            
        except Exception as e:
            print(f"❌ {model_name} model test failed: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {
                'status': 'FAILED',
                'error': str(e)
            }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    success_count = 0
    for model_name, result in results.items():
        status = result['status']
        if status == 'SUCCESS':
            success_count += 1
            print(f"✅ {model_name}: {status}")
            if 'sample_prediction' in result:
                pred = result['sample_prediction']
                print(f"   Sample prediction: value={pred['value']:.3f}, prob={pred['probability']:.3f}, conf={pred['confidence']:.3f}")
        else:
            print(f"❌ {model_name}: {status}")
            print(f"   Error: {result.get('error', 'Unknown error')}")
    
    print(f"\n🎯 Results: {success_count}/{len(models_to_test)} models passed all tests")
    
    if success_count == len(models_to_test):
        print("🎉 All models are working correctly with Rolling Training System!")
    else:
        print("⚠️ Some models need additional fixes.")
    
    return results

if __name__ == "__main__":
    results = test_all_models()
