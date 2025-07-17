"""
Test TFT Model Fix
"""

import torch
import numpy as np
from src.models.deep_learning.tft.enhanced_tft_model import EnhancedTFTPredictor

def test_tft_model():
    """Test TFT model with the fixed tensor dimensions"""
    print("🔧 Testing TFT model fix...")
    
    # Create sample data
    sample_data = []
    for i in range(1000):
        value = np.random.normal(1.5, 0.5)
        value = max(0.1, min(10.0, value))  # Clamp between 0.1 and 10.0
        sample_data.append(value)
    
    # Test configuration
    config = {
        'sequence_length': 250,
        'hidden_size': 256,
        'num_heads': 8,
        'num_layers': 3,
        'learning_rate': 0.0005,
        'threshold': 1.5,
        'device': 'cpu'
    }
    
    try:
        # Create model
        print("🔧 Creating TFT model...")
        model = EnhancedTFTPredictor(**config)
        
        # Test data preparation
        print("🔧 Testing data preparation...")
        X, y = model.prepare_sequences(sample_data)
        print(f"✅ Data prepared successfully. X shape: {X.shape}, y shape: {y.shape}")
        
        # Test forward pass
        print("🔧 Testing forward pass...")
        model.model.eval()
        with torch.no_grad():
            test_batch = X[:2]  # Take first 2 samples
            result = model.model(test_batch)
            print(f"✅ Forward pass successful. Output keys: {result.keys()}")
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    print(f"   - {key}: {value.shape}")
        
        # Test prediction
        print("🔧 Testing prediction...")
        test_sequence = sample_data[:config['sequence_length']]
        
        # First test if model can be "trained" (just few epochs)
        print("🔧 Testing short training...")
        model.train(data=sample_data, epochs=3, batch_size=16, verbose=False)
        
        # Test prediction with confidence
        value, prob, conf = model.predict_with_confidence(test_sequence)
        print(f"✅ Prediction successful: value={value:.4f}, prob={prob:.4f}, conf={conf:.4f}")
        
        print("🎉 TFT model test passed!")
        return True
        
    except Exception as e:
        print(f"❌ TFT model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tft_model()
    if success:
        print("\n✅ TFT fix verified successfully!")
    else:
        print("\n❌ TFT fix verification failed!")
