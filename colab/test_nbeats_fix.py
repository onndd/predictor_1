"""
Test script for N-Beats model fixes
Tests the fixed N-Beats model with different sequence lengths
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

def test_nbeats_model():
    """Test N-Beats model with different sequence lengths"""
    print("🧪 Testing N-Beats Model Fixes")
    print("=" * 50)
    
    # Test different sequence lengths
    sequence_lengths = [100, 200, 300]
    
    for seq_len in sequence_lengths:
        print(f"\n🔧 Testing sequence_length={seq_len}")
        
        try:
            # Create model
            model = NBeatsPredictor(
                sequence_length=seq_len,
                hidden_size=256,
                num_stacks=3,
                num_blocks=3,
                learning_rate=0.001,
                threshold=1.5,
                crash_weight=2.0
            )
            
            print(f"✅ Model created successfully with sequence_length={seq_len}")
            
            # Create sample data
            sample_data = np.random.uniform(1.0, 10.0, seq_len + 100).tolist()
            
            # Test data preparation
            X, y = model.prepare_sequences(sample_data)
            print(f"✅ Data preparation successful: X={X.shape}, y={y.shape}")
            
            # Test forward pass
            with torch.no_grad():
                batch_X = X[:4]  # Take first 4 samples
                predictions = model.model(batch_X)
                print(f"✅ Forward pass successful")
                print(f"   - Value shape: {predictions['value'].shape}")
                print(f"   - Probability shape: {predictions['probability'].shape}")
                print(f"   - Confidence shape: {predictions['confidence'].shape}")
                print(f"   - Crash risk shape: {predictions['crash_risk'].shape}")
                print(f"   - Pattern shape: {predictions['pattern'].shape}")
            
            # Test prediction
            test_sequence = sample_data[:seq_len]
            pred_value, pred_prob, pred_conf = model.predict_with_confidence(test_sequence)
            print(f"✅ Prediction successful")
            print(f"   - Predicted value: {pred_value:.4f}")
            print(f"   - Probability: {pred_prob:.4f}")
            print(f"   - Confidence: {pred_conf:.4f}")
            
            print(f"🎉 All tests passed for sequence_length={seq_len}")
            
        except Exception as e:
            print(f"❌ Test failed for sequence_length={seq_len}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🎯 Test Summary Complete!")

if __name__ == "__main__":
    test_nbeats_model()
