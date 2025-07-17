"""
Test script for N-Beats model fix
"""

import sys
sys.path.append('/content/predictor_1/src')

import numpy as np
import torch
from models.deep_learning.n_beats.n_beats_model import NBeatsPredictor

def test_n_beats_model():
    """Test N-Beats model with fixed tensor dimensions"""
    print("🧪 Testing N-Beats model...")
    
    # Create test data
    test_data = []
    for i in range(1000):
        # Generate random JetX-like values
        value = np.random.uniform(1.0, 3.0)
        test_data.append((i, value))
    
    print(f"📊 Test data created: {len(test_data)} samples")
    print(f"📊 Sample data: {test_data[:5]}")
    
    # Create N-Beats model
    model = NBeatsPredictor(
        sequence_length=300,
        hidden_size=256,
        num_stacks=3,
        num_blocks=3,
        learning_rate=0.001,
        threshold=1.5
    )
    
    print(f"✅ N-Beats model created successfully")
    print(f"📊 Model sequence length: {model.sequence_length}")
    
    # Test data preparation
    sequences, targets = model.prepare_sequences(test_data)
    print(f"✅ Data preparation successful")
    print(f"📊 Sequences shape: {sequences.shape}")
    print(f"📊 Targets shape: {targets.shape}")
    
    # Test training (small epochs for quick test)
    print("🚀 Starting training test...")
    try:
        history = model.train(
            data=test_data,
            epochs=5,  # Small number for quick test
            batch_size=16,
            validation_split=0.2,
            verbose=True
        )
        print("✅ Training completed successfully!")
        print(f"📊 Training history: {len(history['train_losses'])} epochs")
        
        # Test prediction
        test_sequence = [float(item[1]) for item in test_data[300:600]]
        prediction = model.predict(test_sequence)
        print(f"✅ Prediction successful: {prediction}")
        
        # Test prediction with confidence
        value, prob, conf = model.predict_with_confidence(test_sequence)
        print(f"✅ Prediction with confidence: value={value}, prob={prob}, conf={conf}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_n_beats_model()
    if success:
        print("\n🎉 All tests passed! N-Beats model is working correctly.")
    else:
        print("\n❌ Tests failed. Please check the errors above.")
