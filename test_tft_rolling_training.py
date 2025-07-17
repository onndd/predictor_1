"""
Test TFT Rolling Training with Fixed Model
"""

import numpy as np
import torch
from src.training.rolling_trainer import RollingTrainer
from src.training.model_registry import ModelRegistry

def test_tft_rolling_training():
    """Test TFT model with rolling training system"""
    print("🔧 Testing TFT rolling training...")
    
    # Create sample data chunks
    chunks = []
    for chunk_idx in range(6):
        chunk_data = []
        for i in range(1000):
            value = np.random.normal(1.5, 0.5)
            value = max(0.1, min(10.0, value))
            chunk_data.append(value)
        chunks.append(chunk_data)
    
    # Initialize model registry
    model_registry = ModelRegistry()
    
    # TFT config
    tft_config = {
        'sequence_length': 200,  # Reduced from 250 to 200 for faster testing
        'hidden_size': 128,      # Reduced from 256 to 128 for faster testing
        'num_heads': 4,          # Reduced from 8 to 4 for faster testing
        'num_layers': 2,         # Reduced from 3 to 2 for faster testing
        'learning_rate': 0.001,
        'threshold': 1.5,
        'device': 'cpu',
        'train_params': {
            'epochs': 10,        # Reduced from 60 to 10 for faster testing
            'batch_size': 16,
            'validation_split': 0.15,
            'verbose': False
        }
    }
    
    try:
        # Create rolling trainer for TFT
        rolling_trainer = RollingTrainer(
            model_registry=model_registry,
            chunks=chunks,
            model_type='TFT',
            config=tft_config,
            device='cpu'
        )
        
        # Execute rolling training
        results = rolling_trainer.execute_rolling_training()
        
        # Check results
        if results and len(results) > 0:
            print(f"✅ TFT rolling training completed successfully!")
            print(f"📊 Total cycles completed: {len(results)}")
            
            for i, result in enumerate(results):
                performance = result['performance']
                print(f"   - Cycle {i+1}: MAE={performance['mae']:.4f}, Accuracy={performance['accuracy']:.4f}")
            
            # Test model registry
            best_model = model_registry.get_best_model('TFT')
            if best_model:
                print(f"🏆 Best TFT model: {best_model['model_name']} (MAE: {best_model['performance']['mae']:.4f})")
            
            return True
        else:
            print("❌ No results returned from rolling training")
            return False
            
    except Exception as e:
        print(f"❌ TFT rolling training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tft_rolling_training()
    if success:
        print("\n🎉 TFT rolling training test passed!")
    else:
        print("\n❌ TFT rolling training test failed!")
