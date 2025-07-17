"""
Rolling Window Training System for JetX Models
Real model training with rolling window approach
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import os
import json
import pickle
import traceback  # Hata ayıklama için eklendi
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output


class RollingWindowTrainer:
    """
    Rolling window training system for JetX models
    """
    
    def __init__(self, chunks, chunk_size=1000, sequence_length=300):
        self.chunks = chunks
        self.chunk_size = chunk_size
        self.sequence_length = sequence_length
        self.training_results = []
        self.models_trained = {}
        
    def prepare_sequences(self, data, sequence_length):
        """Prepare sequences for training, handling both lists of floats and lists of tuples."""
        # Check if the data is a list of tuples and extract the second element if so.
        if data and isinstance(data[0], tuple):
            processed_data = [float(item[1]) for item in data]
        else:
            processed_data = [float(item) for item in data]

        sequences = []
        targets = []
        
        for i in range(len(processed_data) - sequence_length):
            seq = processed_data[i:i + sequence_length]
            target = processed_data[i + sequence_length]
            
            sequences.append(seq)
            targets.append(target)
        
        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        
        print(f"📊 Prepared sequences shape: {sequences.shape}")
        print(f"📊 Prepared targets shape: {targets.shape}")
        
        return sequences, targets
    
    def train_nbeats_model(self, train_data, config, progress_callback=None):
        """Train N-Beats model with enhanced parameter support"""
        try:
            # Import N-Beats model - fixed path for colab
            import sys
            sys.path.append('/content/predictor_1/src')
            from models.deep_learning.n_beats.n_beats_model import NBeatsPredictor
            
            print(f"🔧 N-Beats: Creating model with sequence_length={config['sequence_length']}")
            print(f"🔧 N-Beats: Training data size: {len(train_data)}")
            print(f"🔧 N-Beats: Hidden size: {config.get('hidden_size', 256)}")
            print(f"🔧 N-Beats: Num stacks: {config.get('num_stacks', 3)}")
            print(f"🔧 N-Beats: Num blocks: {config.get('num_blocks', 3)}")
            
            # Create model with all parameters from config
            model = NBeatsPredictor(
                sequence_length=config['sequence_length'],
                hidden_size=config.get('hidden_size', 256),
                num_stacks=config.get('num_stacks', 3),
                num_blocks=config.get('num_blocks', 3),
                learning_rate=config['learning_rate'],
                threshold=config.get('threshold', 1.5),
                crash_weight=config.get('crash_weight', 2.0)
            )
            
            # Train model
            history = model.train(
                data=train_data,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                validation_split=config.get('validation_split', 0.2),
                verbose=config.get('verbose', True)
            )
            
            return model, history
            
        except Exception as e:
            print(f"❌ N-Beats training error: {e}")
            traceback.print_exc()  # Detaylı hata kaydı
            return None, None
    
    def train_tft_model(self, train_data, config, progress_callback=None):
        """Train Enhanced TFT model with advanced features"""
        try:
            # Import Enhanced TFT model - fixed path for colab
            import sys
            sys.path.append('/content/predictor_1/src')
            from models.deep_learning.tft.enhanced_tft_model import EnhancedTFTPredictor
            
            print(f"🔧 Enhanced TFT: Creating model with sequence_length={config['sequence_length']}")
            print(f"🔧 Enhanced TFT: Training data size: {len(train_data)}")
            print(f"🔧 Enhanced TFT: Hidden size: {config.get('hidden_size', 256)}")
            print(f"🔧 Enhanced TFT: Num heads: {config.get('num_heads', 8)}")
            print(f"🔧 Enhanced TFT: Num layers: {config.get('num_layers', 2)}")
            
            # Create enhanced model with all parameters from config
            model = EnhancedTFTPredictor(
                sequence_length=config['sequence_length'],
                hidden_size=config.get('hidden_size', 256),
                num_heads=config.get('num_heads', 8),
                num_layers=config.get('num_layers', 2),
                learning_rate=config['learning_rate'],
                threshold=config.get('threshold', 1.5)
            )
            
            # Train model
            history = model.train(
                data=train_data,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                validation_split=config.get('validation_split', 0.2),
                verbose=config.get('verbose', True)
            )
            
            return model, history
            
        except Exception as e:
            print(f"❌ Enhanced TFT training error: {e}")
            traceback.print_exc()  # Detaylı hata kaydı
            return None, None
    
    def train_lstm_model(self, train_data, config, progress_callback=None):
        """Train Enhanced LSTM model with PyTorch backend"""
        try:
            # Import Enhanced LSTM model - fixed path for colab
            import sys
            sys.path.append('/content/predictor_1/src')
            from models.sequential.enhanced_lstm_pytorch import EnhancedLSTMPredictor
            
            print(f"🔧 Enhanced LSTM: Creating model with sequence_length={config['sequence_length']}")
            print(f"🔧 Enhanced LSTM: Training data size: {len(train_data)}")
            print(f"🔧 Enhanced LSTM: Hidden size: {config.get('hidden_size', 128)}")
            print(f"🔧 Enhanced LSTM: Num layers: {config.get('num_layers', 2)}")
            print(f"🔧 Enhanced LSTM: Threshold: {config.get('threshold', 1.5)}")
            
            # Create enhanced model with all parameters from config
            model = EnhancedLSTMPredictor(
                seq_length=config['sequence_length'],
                n_features=1,
                threshold=config.get('threshold', 1.5),
                hidden_size=config.get('hidden_size', 128),
                num_layers=config.get('num_layers', 2),
                learning_rate=config['learning_rate']
            )
            
            # Train model
            history = model.train(
                data=train_data,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                validation_split=config.get('validation_split', 0.2),
                verbose=config.get('verbose', True)
            )
            
            return model, history
            
        except Exception as e:
            print(f"❌ Enhanced LSTM training error: {e}")
            traceback.print_exc()  # Detaylı hata kaydı
            return None, None
    
    def test_model(self, model, test_data, sequence_length):
        """Test model on test data"""
        try:
            if len(test_data) < sequence_length + 100:
                return None
            
            predictions = []
            actuals = []
            
            # Test on sequences from test data
            for i in range(sequence_length, len(test_data)):
                raw_sequence = test_data[i-sequence_length:i]
                raw_actual = test_data[i]

                # Ensure sequence and actual are lists of floats
                sequence = [item[1] if isinstance(item, tuple) else item for item in raw_sequence]
                actual = raw_actual[1] if isinstance(raw_actual, tuple) else raw_actual
                
                # Get prediction
                pred_value, pred_prob, pred_conf = model.predict_with_confidence(sequence)
                
                predictions.append({
                    'value': pred_value,
                    'probability': pred_prob,
                    'confidence': pred_conf
                })
                actuals.append(actual)
            
            # Calculate metrics
            pred_values = [p['value'] for p in predictions]
            mae = np.mean(np.abs(np.array(pred_values) - np.array(actuals)))
            rmse = np.sqrt(np.mean((np.array(pred_values) - np.array(actuals))**2))
            
            # Classification metrics
            pred_above = [p['probability'] > 0.5 for p in predictions]
            actual_above = [a >= 1.5 for a in actuals]
            accuracy = np.mean([p == a for p, a in zip(pred_above, actual_above)])
            
            # Crash detection
            actual_crashes = [a < 1.5 for a in actuals]
            pred_crashes = [p['probability'] < 0.5 for p in predictions]
            crash_detection = np.mean([p == a for p, a in zip(pred_crashes, actual_crashes)])
            
            return {
                'mae': mae,
                'rmse': rmse,
                'accuracy': accuracy,
                'crash_detection': crash_detection,
                'predictions': predictions[:50],  # Store first 50 for analysis
                'actuals': actuals[:50]
            }
            
        except Exception as e:
            print(f"❌ Model test error: {e}")
            traceback.print_exc()  # Detaylı hata kaydı
            return None
    
    def save_model(self, model, model_name, config, performance):
        """Save trained model"""
        try:
            model_path = f"/content/trained_models/{model_name}.pth"
            model.save_model(model_path)
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'config': config,
                'performance': performance,
                'timestamp': datetime.now().isoformat()
            }
            
            metadata_path = f"/content/trained_models/{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"💾 Model saved: {model_path}")
            return model_path, metadata_path
            
        except Exception as e:
            print(f"❌ Model save error: {e}")
            traceback.print_exc()  # Detaylı hata kaydı
            return None, None
    
    def execute_rolling_training(self, model_type, config, progress_callback=None):
        """Execute rolling window training"""
        print(f"🚀 Rolling window training started: {model_type}")
        print(f"📊 Total number of chunks: {len(self.chunks)}")
        print(f"📊 Chunk size: {self.chunk_size}")
        
        cycle_results = []
        
        for cycle in range(len(self.chunks) - 1):
            print(f"\n🔄 Cycle {cycle + 1}/{len(self.chunks) - 1}")
            
            # Cumulative training data
            train_data = []
            for i in range(cycle + 1):
                train_data.extend(self.chunks[i])
            
            # Test data
            test_data = self.chunks[cycle + 1]
            
            print(f"📈 Training size: {len(train_data)}")
            print(f"🧪 Test size: {len(test_data)}")
            
            # Progress callback
            if progress_callback:
                progress_callback(cycle + 1, len(self.chunks) - 1, f"Training cycle {cycle + 1}")
            
            # Train model
            if model_type == 'N-Beats':
                model, history = self.train_nbeats_model(train_data, config, progress_callback)
            elif model_type == 'TFT':
                model, history = self.train_tft_model(train_data, config, progress_callback)
            elif model_type == 'LSTM':
                model, history = self.train_lstm_model(train_data, config, progress_callback)
            else:
                print(f"❌ Unsupported model type: {model_type}")
                continue
            
            if model is None:
                print(f"❌ Model training failed for cycle {cycle + 1}")
                continue
            
            # Test model
            print(f"🧪 Testing model on cycle {cycle + 1}")
            test_results = self.test_model(model, test_data, config['sequence_length'])
            
            if test_results is None:
                print(f"❌ Model testing failed for cycle {cycle + 1}")
                continue
            
            # Save model
            model_name = f"{model_type}_cycle_{cycle + 1}_{datetime.now().strftime('%H%M%S')}"
            model_path, metadata_path = self.save_model(model, model_name, config, test_results)
            
            # Store results
            cycle_result = {
                'cycle': cycle + 1,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'train_data_range': f"1-{len(train_data)}",
                'test_data_range': f"{len(train_data)+1}-{len(train_data)+len(test_data)}",
                'model_name': model_name,
                'model_path': model_path,
                'metadata_path': metadata_path,
                'performance': test_results,
                'training_history': history.get('train_losses', []) if history else []
            }
            
            cycle_results.append(cycle_result)
            
            # Store in class
            self.models_trained[model_name] = {
                'model': model,
                'config': config,
                'performance': test_results,
                'cycle': cycle + 1
            }
            
            # Print results
            print(f"✅ Cycle {cycle + 1} completed:")
            print(f"   📊 MAE: {test_results['mae']:.4f}")
            print(f"   📊 RMSE: {test_results['rmse']:.4f}")
            print(f"   📊 Accuracy: {test_results['accuracy']:.4f}")
            print(f"   📊 Crash Detection: {test_results['crash_detection']:.4f}")
        
        self.training_results.append({
            'model_type': model_type,
            'config': config,
            'cycles': cycle_results,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"\n🎉 Rolling window training completed for {model_type}")
        print(f"📊 Total cycles: {len(cycle_results)}")
        
        return cycle_results
    
    def get_best_model(self, model_type, metric='mae'):
        """Get best model for a specific type"""
        models = {k: v for k, v in self.models_trained.items() 
                 if k.startswith(model_type)}
        
        if not models:
            return None
        
        best_model = None
        best_score = float('inf') if metric == 'mae' else 0
        
        for name, info in models.items():
            score = info['performance'][metric]
            if metric == 'mae' and score < best_score:
                best_score = score
                best_model = name
            elif metric == 'accuracy' and score > best_score:
                best_score = score
                best_model = name
        
        return best_model, self.models_trained[best_model]
    
    def plot_training_evolution(self, model_type):
        """Plot training evolution across cycles"""
        results = None
        for result in self.training_results:
            if result['model_type'] == model_type:
                results = result
                break
        
        if not results:
            print(f"No results found for {model_type}")
            return
        
        cycles = results['cycles']
        cycle_nums = [c['cycle'] for c in cycles]
        maes = [c['performance']['mae'] for c in cycles]
        accuracies = [c['performance']['accuracy'] for c in cycles]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # MAE evolution
        ax1.plot(cycle_nums, maes, marker='o', linewidth=2, markersize=8)
        ax1.set_xlabel('Cycle')
        ax1.set_ylabel('MAE')
        ax1.set_title(f'{model_type} - MAE Evolution')
        ax1.grid(True, alpha=0.3)
        
        # Accuracy evolution
        ax2.plot(cycle_nums, accuracies, marker='s', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('Cycle')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'{model_type} - Accuracy Evolution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, filename=None):
        """Export training results to file"""
        if filename is None:
            filename = f"rolling_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = f"/content/trained_models/{filename}"
        
        with open(filepath, 'w') as f:
            json.dump(self.training_results, f, indent=2)
        
        print(f"📊 Results exported to: {filepath}")
        return filepath
