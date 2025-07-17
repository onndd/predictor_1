#!/usr/bin/env python3
"""
N-Beats Model Quick Fix Test
Test the tensor dimension fixes
"""

import sys
import os
sys.path.append('src')

import torch
import numpy as np
from models.deep_learning.n_beats.n_beats_model import NBeatsPredictor, JetXNBeatsModel, JetXNBeatsBlock

def test_basic_forward_pass():
    """Test basic forward pass without training"""
    print("🧪 Testing N-Beats Forward Pass...")
    
    # Test parameters
    input_size = 300
    batch_size = 32
    
    try:
        # Test JetXNBeatsBlock directly
        print(f"🔧 Testing JetXNBeatsBlock with input_size={input_size}")
        
        # Test different basis functions
        basis_functions = ['jetx_crash', 'jetx_pump', 'jetx_trend']
        
        for basis_func in basis_functions:
            print(f"🔧 Testing basis function: {basis_func}")
            
            block = JetXNBeatsBlock(
                input_size=input_size,
                basis_function=basis_func,
                hidden_size=256,
                num_layers=4
            )
            
            # Create random input
            x = torch.randn(batch_size, input_size)
            
            # Forward pass
            backcast, forecast = block(x)
            
            print(f"✅ {basis_func}: Input {x.shape} -> Backcast {backcast.shape}, Forecast {forecast.shape}")
            
            # Verify shapes
            assert backcast.shape == (batch_size, input_size), f"Backcast shape mismatch: {backcast.shape}"
            assert forecast.shape == (batch_size, 1), f"Forecast shape mismatch: {forecast.shape}"
            
            # Check for NaN/Inf
            assert not torch.isnan(backcast).any(), f"NaN in backcast for {basis_func}"
            assert not torch.isnan(forecast).any(), f"NaN in forecast for {basis_func}"
            assert not torch.isinf(backcast).any(), f"Inf in backcast for {basis_func}"
            assert not torch.isinf(forecast).any(), f"Inf in forecast for {basis_func}"
            
        print("✅ JetXNBeatsBlock tests passed!")
        
        # Test full model
        print("🔧 Testing full JetXNBeatsModel...")
        model = JetXNBeatsModel(
            input_size=input_size,
            forecast_size=1,
            num_stacks=3,
            num_blocks=3,
            hidden_size=256
        )
        
        x = torch.randn(batch_size, input_size)
        predictions = model(x)
        
        print(f"✅ Model output keys: {predictions.keys()}")
        for key, value in predictions.items():
            print(f"✅ {key}: {value.shape}")
            assert not torch.isnan(value).any(), f"NaN in {key}"
            assert not torch.isinf(value).any(), f"Inf in {key}"
        
        print("✅ Full model test passed!")
        
        # Test NBeatsPredictor initialization
        print("🔧 Testing NBeatsPredictor initialization...")
        predictor = NBeatsPredictor(
            sequence_length=input_size,
            hidden_size=256,
            num_stacks=3,
            num_blocks=3
        )
        
        print("✅ NBeatsPredictor initialized successfully!")
        
        # Test with small data
        print("🔧 Testing data preparation...")
        dummy_data = np.random.uniform(1.0, 3.0, 1000).tolist()
        
        X, y = predictor.prepare_sequences(dummy_data)
        print(f"✅ Prepared sequences: X {X.shape}, y {y.shape}")
        
        # Test small forward pass (no training)
        print("🔧 Testing predictor forward pass...")
        predictor.model.eval()
        with torch.no_grad():
            small_batch = X[:4]  # Take first 4 samples
            pred_output = predictor.model(small_batch)
            print("✅ Predictor forward pass successful!")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_tensor_shapes():
    """Test specific tensor shape scenarios that were causing issues"""
    print("🧪 Testing Tensor Shape Scenarios...")
    
    try:
        # Scenario 1: Input size 300, theta_size should be calculated correctly
        input_size = 300
        
        # Test jetx_trend basis (should get theta_size = 12)
        block_trend = JetXNBeatsBlock(
            input_size=input_size,
            basis_function='jetx_trend',
            hidden_size=256
        )
        
        print(f"✅ jetx_trend theta_size: {block_trend.theta_size}")
        
        # Test jetx_crash basis (should get theta_size = 15)
        block_crash = JetXNBeatsBlock(
            input_size=input_size,
            basis_function='jetx_crash',
            hidden_size=256
        )
        
        print(f"✅ jetx_crash theta_size: {block_crash.theta_size}")
        
        # Test with different batch sizes
        for batch_size in [1, 8, 32, 128]:
            x = torch.randn(batch_size, input_size)
            
            backcast_trend, forecast_trend = block_trend(x)
            backcast_crash, forecast_crash = block_crash(x)
            
            print(f"✅ Batch {batch_size}: trend shapes {backcast_trend.shape}, {forecast_trend.shape}")
            print(f"✅ Batch {batch_size}: crash shapes {backcast_crash.shape}, {forecast_crash.shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ Tensor shape test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 N-Beats Quick Fix Test Started")
    print("=" * 50)
    
    success = True
    
    # Test 1: Basic forward pass
    if not test_basic_forward_pass():
        success = False
    
    print("-" * 50)
    
    # Test 2: Tensor shapes
    if not test_tensor_shapes():
        success = False
    
    print("=" * 50)
    
    if success:
        print("🎉 All tests passed! N-Beats fixes are working correctly.")
    else:
        print("❌ Some tests failed. Check the errors above.")
    
    print("🏁 Test completed")
