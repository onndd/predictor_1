#!/usr/bin/env python3
"""
Optimized JetX Model Trainer
Automatic training with optimized parameters - no manual adjustments needed
"""

import numpy as np
import torch
import json
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add source path for imports
if '/content/predictor_1/src' not in sys.path:
    sys.path.append('/content/predictor_1/src')


def create_sample_jetx_data(n_samples=6000):
    """Create sample JetX data for testing"""
    np.random.seed(42)
    data = []
    
    for _ in range(n_samples):
        rand = np.random.random()
        if rand < 0.25:  # 25% crash
            value = np.random.uniform(1.0, 1.49)
        elif rand < 0.45:  # 20% low
            value = np.random.uniform(1.5, 2.5)
        elif rand < 0.70:  # 25% medium
            value = np.random.uniform(2.5, 5.0)
        elif rand < 0.90:  # 20% high
            value = np.random.uniform(5.0, 20.0)
        else:  # 10% very high
            value = np.random.exponential(10.0) + 20.0
        
        data.append(round(value, 2))
    
    return data


def create_rolling_chunks(data, chunk_size=1000):
    """Split data into rolling window chunks"""
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        if len(chunk) >= chunk_size:
            chunks.append(chunk)
    return chunks


class OptimizedJetXTrainer:
    """
    Optimized JetX Model Trainer with pre-configured parameters
    No manual adjustments needed - fully automatic
    """
    
    def __init__(self):
        self.models_config = {
            'N-Beats': {
                'sequence_length': 300,
                'hidden_size': 512,
                'num_stacks': 4,
                'num_blocks': 4,
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 32,
                'threshold': 1.5,
                'crash_weight': 3.0,
                'validation_split': 0.2
            },
            'TFT': {
                'sequence_length': 300,
                'hidden_size': 384,
                'num_heads': 8,
                'num_layers': 3,
                'learning_rate': 0.0008,
                'epochs': 80,
                'batch_size': 24,
                'threshold': 1.5,
                'validation_split': 0.2
            },
            'LSTM': {
                'sequence_length': 250,
                'hidden_size': 256,
                'num_layers': 3,
                'learning_rate': 0.001,
                'epochs': 80,
                'batch_size': 32,
                'threshold': 1.5,
                'validation_split': 0.2
            }
        }
        self.training_results = {}
        
    def setup_directories(self):
        """Setup required directories"""
        os.makedirs('/content/trained_models', exist_ok=True)
        print("📁 Directories ready")
        
    def load_data(self):
        """Load or create JetX data"""
        try:
            from data_processing.loader import load_data_from_sqlite
            
            repo_db_path = "/content/predictor_1/data/jetx_data.db"
            
            if os.path.exists(repo_db_path):
                data = load_data_from_sqlite(repo_db_path)
                if data and len(data) > 1000:
                    print(f"✅ Loaded {len(data)} records from database")
                    return data
            
            print("📊 Creating sample data...")
            data = create_sample_jetx_data(6000)
            print(f"✅ Created {len(data)} sample records")
            return data
            
        except Exception as e:
            print(f"⚠️ Data loading failed: {e}")
            print("📊 Creating sample data...")
            return create_sample_jetx_data(6000)
    
    def train_nbeats(self, data, chunks):
        """Train optimized N-Beats model"""
        print("\n🚀 Training N-Beats Model (Optimized)")
        print("=" * 50)
        
        try:
            from models.deep_learning.n_beats.n_beats_model import NBeatsPredictor
            
            config = self.models_config['N-Beats']
            print(f"📊 Configuration: {config}")
            
            # Use cumulative data for training (more data = better performance)
            train_data = []
            for chunk in chunks[:-1]:  # All chunks except last
                train_data.extend(chunk)
            
            test_data = chunks[-1]  # Last chunk for testing
            
            print(f"📈 Training size: {len(train_data)}")
            print(f"🧪 Test size: {len(test_data)}")
            
            # Create and train model
            model = NBeatsPredictor(
                sequence_length=config['sequence_length'],
                hidden_size=config['hidden_size'],
                num_stacks=config['num_stacks'],
                num_blocks=config['num_blocks'],
                learning_rate=config['learning_rate'],
                threshold=config['threshold'],
                crash_weight=config['crash_weight']
            )
            
            print("🔧 Training started...")
            history = model.train(
                data=train_data,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                validation_split=config['validation_split'],
                verbose=True
            )
            
            # Test model
            print("🧪 Testing model...")
            test_results = self.test_model(model, test_data, config['sequence_length'])
            
            # Save model
            model_name = f"N-Beats_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = f"/content/trained_models/{model_name}.pth"
            model.save_model(model_path)
            
            # Save results
            self.training_results['N-Beats'] = {
                'config': config,
                'performance': test_results,
                'model_path': model_path,
                'training_history': history
            }
            
            print(f"✅ N-Beats training completed!")
            print(f"📊 Performance: MAE={test_results['mae']:.4f}, Correct Play Advice={test_results.get('correct_play_advice_accuracy', 0):.2%}")
            print(f"💾 Model saved: {model_path}")
            
            return model, test_results
            
        except Exception as e:
            print(f"❌ N-Beats training failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def train_tft(self, data, chunks):
        """Train optimized TFT model"""
        print("\n🚀 Training TFT Model (Optimized)")
        print("=" * 50)
        
        try:
            from models.deep_learning.tft.enhanced_tft_model import EnhancedTFTPredictor
            
            config = self.models_config['TFT']
            print(f"📊 Configuration: {config}")
            
            # Use cumulative data for training
            train_data = []
            for chunk in chunks[:-1]:
                train_data.extend(chunk)
            
            test_data = chunks[-1]
            
            print(f"📈 Training size: {len(train_data)}")
            print(f"🧪 Test size: {len(test_data)}")
            
            # Create and train model
            model = EnhancedTFTPredictor(
                sequence_length=config['sequence_length'],
                hidden_size=config['hidden_size'],
                num_heads=config['num_heads'],
                num_layers=config['num_layers'],
                learning_rate=config['learning_rate'],
                threshold=config['threshold']
            )
            
            print("🔧 Training started...")
            history = model.train(
                data=train_data,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                validation_split=config['validation_split'],
                verbose=True
            )
            
            # Test model
            print("🧪 Testing model...")
            test_results = self.test_model(model, test_data, config['sequence_length'])
            
            # Save model
            model_name = f"TFT_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = f"/content/trained_models/{model_name}.pth"
            model.save_model(model_path)
            
            # Save results
            self.training_results['TFT'] = {
                'config': config,
                'performance': test_results,
                'model_path': model_path,
                'training_history': history
            }
            
            print(f"✅ TFT training completed!")
            print(f"📊 Performance: MAE={test_results['mae']:.4f}, Correct Play Advice={test_results.get('correct_play_advice_accuracy', 0):.2%}")
            print(f"💾 Model saved: {model_path}")
            
            return model, test_results
            
        except Exception as e:
            print(f"❌ TFT training failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def train_lstm(self, data, chunks):
        """Train optimized LSTM model"""
        print("\n🚀 Training LSTM Model (Optimized)")
        print("=" * 50)
        
        try:
            from models.sequential.enhanced_lstm_pytorch import EnhancedLSTMPredictor
            
            config = self.models_config['LSTM']
            print(f"📊 Configuration: {config}")
            
            # Use cumulative data for training
            train_data = []
            for chunk in chunks[:-1]:
                train_data.extend(chunk)
            
            test_data = chunks[-1]
            
            print(f"📈 Training size: {len(train_data)}")
            print(f"🧪 Test size: {len(test_data)}")
            
            # Create and train model
            model = EnhancedLSTMPredictor(
                seq_length=config['sequence_length'],
                n_features=1,
                threshold=config['threshold'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                learning_rate=config['learning_rate']
            )
            
            print("🔧 Training started...")
            history = model.train(
                data=train_data,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                validation_split=config['validation_split'],
                verbose=True
            )
            
            # Test model
            print("🧪 Testing model...")
            test_results = self.test_model(model, test_data, config['sequence_length'])
            
            # Save model
            model_name = f"LSTM_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = f"/content/trained_models/{model_name}.pth"
            model.save_model(model_path)
            
            # Save results
            self.training_results['LSTM'] = {
                'config': config,
                'performance': test_results,
                'model_path': model_path,
                'training_history': history
            }
            
            print(f"✅ LSTM training completed!")
            print(f"📊 Performance: MAE={test_results['mae']:.4f}, Correct Play Advice={test_results.get('correct_play_advice_accuracy', 0):.2%}")
            print(f"💾 Model saved: {model_path}")
            
            return model, test_results
            
        except Exception as e:
            print(f"❌ LSTM training failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def test_model(self, model, test_data, sequence_length):
        """Test model performance"""
        try:
            if len(test_data) < sequence_length + 100:
                return {'mae': 999, 'accuracy': 0, 'rmse': 999, 'play_advice_ratio': 0}

            predictions = []
            actuals = []

            # Test on sequences from test data
            for i in range(sequence_length, min(len(test_data), sequence_length + 500)):
                raw_sequence = test_data[i-sequence_length:i]
                raw_actual = test_data[i]

                sequence = [item[1] if isinstance(item, tuple) else item for item in raw_sequence]
                actual = raw_actual[1] if isinstance(raw_actual, tuple) else raw_actual
                
                # Get prediction with advice using the new method
                result = model.get_prediction_with_advice(sequence)
                predictions.append(result)
                actuals.append(actual)

            if not predictions:
                return {'mae': 999, 'accuracy': 0, 'rmse': 999, 'play_advice_ratio': 0}

            # --- Calculate Metrics ---
            pred_values = [p.prediction for p in predictions]
            mae = np.mean(np.abs(np.array(pred_values) - np.array(actuals)))
            rmse = np.sqrt(np.mean((np.array(pred_values) - np.array(actuals))**2))

            # Standard accuracy (prediction >= 1.5 vs actual >= 1.5)
            pred_above_1_5 = [p.prediction >= 1.5 for p in predictions]
            actual_above_1_5 = [a >= 1.5 for a in actuals]
            accuracy = np.mean([p == a for p, a in zip(pred_above_1_5, actual_above_1_5)])

            # --- New Advice-Based Metrics ---
            play_advices = [p.advice == "Play" for p in predictions]
            play_advice_ratio = np.mean(play_advices) if play_advices else 0

            # Accuracy of "Play" advice: Of all "Play" advices, how many were correct (actual >= 1.5)
            correct_play_predictions = [
                actual_above_1_5[i] for i, advised_play in enumerate(play_advices) if advised_play
            ]
            correct_play_advice_accuracy = np.mean(correct_play_predictions) if correct_play_predictions else 0
            
            # Overall advice accuracy is the average confidence
            overall_advice_accuracy = np.mean([p.advice_accuracy for p in predictions])

            return {
                'mae': float(mae),
                'rmse': float(rmse),
                'accuracy': float(accuracy),
                'play_advice_ratio': float(play_advice_ratio),
                'correct_play_advice_accuracy': float(correct_play_advice_accuracy),
                'overall_advice_accuracy': float(overall_advice_accuracy)
            }

        except Exception as e:
            print(f"❌ Model test error: {e}")
            import traceback
            traceback.print_exc()
            return {'mae': 999, 'accuracy': 0, 'rmse': 999, 'play_advice_ratio': 0}
    
    def train_all_models(self):
        """Train all models with optimized parameters"""
        print("🚀 OPTIMIZED JETX MODEL TRAINER")
        print("=" * 50)
        print("✨ Pre-configured for maximum performance")
        print("🎯 No manual adjustments needed")
        print("")
        
        # Setup
        self.setup_directories()
        
        # Load data
        print("📊 Loading data...")
        data = self.load_data()
        
        # Create chunks
        chunks = create_rolling_chunks(data, 1000)
        print(f"📊 Created {len(chunks)} chunks for training")
        
        # Data statistics
        crash_values = [item[1] if isinstance(item, tuple) else item for item in data]
        print(f"📈 Data statistics:")
        print(f"   - Total records: {len(crash_values)}")
        print(f"   - Min: {min(crash_values):.2f}")
        print(f"   - Max: {max(crash_values):.2f}")
        print(f"   - Average: {np.mean(crash_values):.2f}")
        print(f"   - Crash rate: {sum(1 for x in crash_values if x < 1.5)/len(crash_values)*100:.1f}%")
        
        # Train models
        models_trained = {}
        
        # 1. Train N-Beats (highest priority)
        print("\n" + "="*50)
        model, results = self.train_nbeats(data, chunks)
        if model and results:
            models_trained['N-Beats'] = (model, results)
        
        # 2. Train TFT
        print("\n" + "="*50)
        model, results = self.train_tft(data, chunks)
        if model and results:
            models_trained['TFT'] = (model, results)
        
        # 3. Train LSTM
        print("\n" + "="*50)
        model, results = self.train_lstm(data, chunks)
        if model and results:
            models_trained['LSTM'] = (model, results)
        
        # Summary
        self.print_summary()
        
        return models_trained
    
    def print_summary(self):
        """Print training summary"""
        print("\n" + "="*50)
        print("🎉 TRAINING COMPLETED - SUMMARY")
        print("="*50)
        
        if not self.training_results:
            print("❌ No models were successfully trained")
            return
        
        # Find best model
        best_model = None
        best_mae = float('inf')
        
        for model_name, result in self.training_results.items():
            mae = result['performance']['mae']
            accuracy = result['performance']['accuracy']
            
            print(f"\n📊 {model_name} Results:")
            print(f"   MAE: {mae:.4f}")
            print(f"   RMSE: {result['performance']['rmse']:.4f}")
            print(f"   Accuracy (Pred >= 1.5): {accuracy:.4f}")
            print(f"   Play Advice Ratio: {result['performance'].get('play_advice_ratio', 0):.2%}")
            print(f"   Correct 'Play' Advice Acc: {result['performance'].get('correct_play_advice_accuracy', 0):.2%}")
            print(f"   Overall Advice Confidence: {result['performance'].get('overall_advice_accuracy', 0):.2%}")
            print(f"   Model: {result['model_path']}")
            
            if mae < best_mae:
                best_mae = mae
                best_model = model_name
        
        if best_model:
            print(f"\n🏆 BEST MODEL: {best_model}")
            print(f"🎯 Best MAE: {best_mae:.4f}")
            print(f"📈 Best Accuracy: {self.training_results[best_model]['performance']['accuracy']:.4f}")
        
        print(f"\n💾 All models saved to: /content/trained_models/")
        print("✨ Training completed successfully!")


def main():
    """Main training function"""
    trainer = OptimizedJetXTrainer()
    models = trainer.train_all_models()
    return trainer, models


if __name__ == "__main__":
    # Run automatic training
    trainer, models = main()
