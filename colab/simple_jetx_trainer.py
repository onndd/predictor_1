#!/usr/bin/env python3
"""
Simple JetX Trainer - One-click automated training
No manual adjustments - just run and get results
"""

# Import the optimized trainer
exec(open('/content/predictor_1/colab/optimized_jetx_trainer.py').read())

print("🚀 SIMPLE JETX TRAINER")
print("=" * 50)
print("✨ Fully automated training with optimized parameters")
print("🎯 No sliders, no buttons, no manual work needed!")
print("")

def train_all_models_automatically():
    """One-click training function"""
    print("🔥 Starting automatic training...")
    print("📊 Using optimized parameters:")
    print("   - N-Beats: seq_len=300, hidden=512, stacks=4, blocks=4")
    print("   - TFT: seq_len=300, hidden=384, heads=8, layers=3") 
    print("   - LSTM: seq_len=250, hidden=256, layers=3")
    print("")
    
    # Create trainer and run
    trainer = OptimizedJetXTrainer()
    models = trainer.train_all_models()
    
    print("\n🎉 TRAINING COMPLETED!")
    print("💾 All models saved to /content/trained_models/")
    
    return trainer, models

# Auto-run when imported
if __name__ == "__main__":
    trainer, models = train_all_models_automatically()
else:
    print("📌 To start training, run: train_all_models_automatically()")
