"""
Test training with fixed N-Beats model
Tests actual training with sequence_length=300
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

def test_training():
    """Test actual training with sequence_length=300"""
    print("🧪 Testing N-Beats Training with sequence_length=300")
    print("=" * 60)
    
    try:
        # Create model with sequence_length=300 (the problematic one)
        model = NBeatsPredictor(
            sequence_length=300,
            hidden_size=256,
            num_stacks=3,
            num_blocks=3,
            learning_rate=0.001,
            threshold=1.5,
            crash_weight=2.0
        )
        
        print(f"✅ Model created successfully with sequence_length=300")
        
        # Create synthetic training data
        np.random.seed(42)
        n_samples = 1000
        data = []
        
        for _ in range(n_samples):
            rand = np.random.random()
            if rand < 0.25:  # %25 crash
                value = np.random.uniform(1.0, 1.49)
            elif rand < 0.45:  # %20 düşük
                value = np.random.uniform(1.5, 2.5)
            elif rand < 0.70:  # %25 orta
                value = np.random.uniform(2.5, 5.0)
            elif rand < 0.90:  # %20 yüksek
                value = np.random.uniform(5.0, 20.0)
            else:  # %10 çok yüksek
                value = np.random.exponential(10.0) + 20.0
            
            data.append(round(value, 2))
        
        print(f"✅ Synthetic data created: {len(data)} samples")
        
        # Test training - just 5 epochs to verify no errors
        print("🚀 Starting training test (5 epochs)...")
        
        history = model.train(
            data=data,
            epochs=5,
            batch_size=32,
            validation_split=0.2,
            verbose=True
        )
        
        print("✅ Training completed successfully!")
        print(f"   - Train losses: {len(history['train_losses'])} epochs")
        print(f"   - Val losses: {len(history['val_losses'])} epochs")
        print(f"   - Final train loss: {history['train_losses'][-1]:.6f}")
        print(f"   - Final val loss: {history['val_losses'][-1]:.6f}")
        
        # Test prediction after training
        test_sequence = data[:300]
        pred_value, pred_prob, pred_conf = model.predict_with_confidence(test_sequence)
        
        print("✅ Prediction successful after training!")
        print(f"   - Predicted value: {pred_value:.4f}")
        print(f"   - Probability: {pred_prob:.4f}")
        print(f"   - Confidence: {pred_conf:.4f}")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("🎯 sequence_length=300 ile eğitim tamamen çalışıyor!")
        print("🎯 Artık colab arayüzünde herhangi bir sequence_length kullanabilirsiniz!")
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_training()
    if success:
        print("\n✅ Model düzeltmeleri başarıyla tamamlandı!")
    else:
        print("\n❌ Hala sorun var, inceleme gerekli.")
