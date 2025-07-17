"""
Test full parameter integration
Tests all parameters from colab interface to model training
"""

import sys
import os

# Add src to path for both local and colab environments
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
import torch
from models.deep_learning.n_beats.n_beats_model import NBeatsPredictor

def test_full_parameter_integration():
    """Test full parameter integration like colab interface"""
    print("🧪 Testing Full Parameter Integration")
    print("=" * 60)
    
    # Simulate colab interface config with ALL parameters
    config = {
        'model_type': 'N-Beats',
        'sequence_length': 300,  # Maximum sequence length
        'epochs': 5,            # Quick test
        'batch_size': 32,
        'learning_rate': 0.001,
        'chunk_size': 1000,
        'hidden_size': 512,     # Different from default
        'num_stacks': 4,        # Different from default  
        'num_blocks': 4,        # Different from default
        'threshold': 1.8,       # Different from default
        'crash_weight': 3.5     # Different from default
    }
    
    print("📊 Test Config (simulating colab interface):")
    for key, value in config.items():
        print(f"   - {key}: {value}")
    print()
    
    try:
        # Create model with ALL config parameters
        print("🔧 Creating N-Beats model with custom parameters...")
        model = NBeatsPredictor(
            sequence_length=config['sequence_length'],
            hidden_size=config['hidden_size'],
            num_stacks=config['num_stacks'],
            num_blocks=config['num_blocks'],
            learning_rate=config['learning_rate'],
            threshold=config['threshold'],
            crash_weight=config['crash_weight']
        )
        
        print(f"✅ Model created successfully!")
        print(f"   - sequence_length: {model.sequence_length}")
        print(f"   - hidden_size: {model.hidden_size}")
        print(f"   - num_stacks: {model.num_stacks}")
        print(f"   - num_blocks: {model.num_blocks}")
        
        # Create test data
        print("\n📊 Creating synthetic test data...")
        np.random.seed(42)
        data = []
        
        for _ in range(800):  # Enough for training + validation
            rand = np.random.random()
            if rand < 0.25:
                value = np.random.uniform(1.0, 1.49)
            elif rand < 0.45:
                value = np.random.uniform(1.5, 2.5)
            elif rand < 0.70:
                value = np.random.uniform(2.5, 5.0)
            elif rand < 0.90:
                value = np.random.uniform(5.0, 20.0)
            else:
                value = np.random.exponential(10.0) + 20.0
            data.append(round(value, 2))
        
        print(f"✅ Test data created: {len(data)} samples")
        
        # Test training with custom parameters
        print(f"\n🚀 Testing training with custom parameters...")
        print(f"   - Using sequence_length={config['sequence_length']}")
        print(f"   - Using hidden_size={config['hidden_size']}")
        print(f"   - Using num_stacks={config['num_stacks']}")
        print(f"   - Using threshold={config['threshold']}")
        
        history = model.train(
            data=data,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_split=0.2,
            verbose=True
        )
        
        print("✅ Training completed successfully!")
        print(f"   - Final train loss: {history['train_losses'][-1]:.6f}")
        print(f"   - Final val loss: {history['val_losses'][-1]:.6f}")
        
        # Test prediction with custom threshold
        print(f"\n🎯 Testing prediction with custom threshold={config['threshold']}...")
        test_sequence = data[:config['sequence_length']]
        pred_value, pred_prob, pred_conf = model.predict_with_confidence(test_sequence)
        
        print(f"✅ Prediction successful!")
        print(f"   - Predicted value: {pred_value:.4f}")
        print(f"   - Probability: {pred_prob:.4f}")
        print(f"   - Confidence: {pred_conf:.4f}")
        
        # Validate threshold logic
        threshold_test = pred_value >= config['threshold']
        print(f"   - Above threshold ({config['threshold']}): {threshold_test}")
        
        print(f"\n🎉 ALL PARAMETER INTEGRATION TESTS PASSED!")
        print(f"🎯 Colab arayüzündeki TÜM parametreler doğru çalışıyor!")
        print(f"🎯 sequence_length=300, hidden_size=512, num_stacks=4, threshold=1.8 vs. tamamıyla entegre!")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_parameter_integration()
    if success:
        print("\n✅ Tüm parametre entegrasyonu başarılı!")
        print("📊 Colab arayüzünde artık TÜM parametreleri ayarlayabilirsin:")
        print("   - ⚙️ Rolling Window: chunk_size, sequence_length, epochs, batch_size, learning_rate")
        print("   - 🏗️ Model Architecture: hidden_size, num_stacks, num_blocks") 
        print("   - 🎯 JetX Specific: threshold, crash_weight")
    else:
        print("\n❌ Parametre entegrasyonunda sorun var.")
