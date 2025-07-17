"""
Rolling Window Training Integration for Colab Notebook
Provides interface for rolling window training with JetX models
"""

import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
from datetime import datetime
import json
import os
import traceback
from typing import Dict, List, Any, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

class RealJetXTrainingInterface:
    """
    Real rolling window training interface for JetX model training
    """
    
    def __init__(self, model_registry: Any, rolling_chunks: List[Any]):
        self.model_registry = model_registry
        self.rolling_chunks = rolling_chunks
        self.current_training: Optional[bool] = None
        self.rolling_trainer: Optional[Any] = None
        self.setup_widgets()
        
    def setup_widgets(self):
        """Setup widget components"""
        
        # Model selection
        self.model_selector = widgets.Dropdown(
            options=['N-Beats', 'TFT', 'LSTM'],
            value='N-Beats',
            description='Model:',
            style={'description_width': 'initial'}
        )
        
        # Rolling Window parameters
        self.chunk_size = widgets.IntSlider(
            value=1000, min=500, max=2000, step=100,
            description='Chunk Size:',
            style={'description_width': 'initial'}
        )
        
        self.sequence_length = widgets.IntSlider(
            value=200, min=50, max=300, step=25,
            description='Sequence Length:',
            style={'description_width': 'initial'}
        )
        
        self.epochs_per_cycle = widgets.IntSlider(
            value=30, min=10, max=100, step=10,
            description='Epochs per Cycle:',
            style={'description_width': 'initial'}
        )
        
        self.batch_size = widgets.Dropdown(
            options=[16, 32, 64, 128],
            value=32,
            description='Batch Size:',
            style={'description_width': 'initial'}
        )
        
        self.learning_rate = widgets.FloatLogSlider(
            value=0.001, base=10, min=-5, max=-1,
            description='Learning Rate:',
            style={'description_width': 'initial'}
        )
        
        # Model architecture parameters
        self.hidden_size = widgets.Dropdown(
            options=[128, 256, 512],
            value=256,
            description='Hidden Size:',
            style={'description_width': 'initial'}
        )
        
        self.num_stacks = widgets.IntSlider(
            value=3, min=2, max=5, step=1,
            description='Num Stacks:',
            style={'description_width': 'initial'}
        )
        
        self.num_blocks = widgets.IntSlider(
            value=3, min=2, max=5, step=1,
            description='Num Blocks:',
            style={'description_width': 'initial'}
        )
        
        self.threshold = widgets.FloatSlider(
            value=1.5, min=1.1, max=2.0, step=0.1,
            description='Threshold:',
            style={'description_width': 'initial'}
        )
        
        self.crash_weight = widgets.FloatSlider(
            value=2.0, min=1.0, max=5.0, step=0.5,
            description='Crash Weight:',
            style={'description_width': 'initial'}
        )
        
        # Training control buttons
        self.train_button = widgets.Button(
            description='🚀 Start Rolling Training',
            button_style='success',
            layout=widgets.Layout(width='250px')
        )
        
        self.stop_button = widgets.Button(
            description='⏹️ Stop',
            button_style='danger',
            layout=widgets.Layout(width='200px'),
            disabled=True
        )
        
        # Progress bar
        self.progress = widgets.IntProgress(
            value=0, min=0, max=100,
            description='Progress:',
            bar_style='info',
            layout=widgets.Layout(width='500px')
        )
        
        # Cycle info
        self.cycle_info = widgets.HTML(
            value="Rolling window training not started yet."
        )
        
        # Output areas
        self.output_area = widgets.Output()
        self.model_info_area = widgets.Output()
        
        # Event handlers
        self.train_button.on_click(self.on_train_click)
        self.stop_button.on_click(self.on_stop_click)
        
    def on_train_click(self, button):
        """Handle training button click"""
        self.start_rolling_training()
    
    def on_stop_click(self, button):
        """Handle stop button click"""
        self.stop_training()
    
    def start_rolling_training(self):
        """Start rolling window training"""
        self.train_button.disabled = True
        self.stop_button.disabled = False
        self.progress.value = 0
        self.current_training = True
        
        # Get training parameters
        config = {
            'model_type': self.model_selector.value,
            'sequence_length': self.sequence_length.value,
            'epochs': self.epochs_per_cycle.value,
            'batch_size': self.batch_size.value,
            'learning_rate': self.learning_rate.value,
            'chunk_size': self.chunk_size.value,
            'hidden_size': self.hidden_size.value,
            'num_stacks': self.num_stacks.value,
            'num_blocks': self.num_blocks.value,
            'threshold': self.threshold.value,
            'crash_weight': self.crash_weight.value
        }
        
        with self.output_area:
            clear_output()
            print(f"🚀 Starting Rolling Window Training: {config['model_type']}")
            print(f"📊 Number of chunks: {len(self.rolling_chunks)}")
            print(f"📊 Chunk size: {config['chunk_size']}")
            print(f"📊 Sequence length: {config['sequence_length']}")
            print(f"📊 Epochs per cycle: {config['epochs']}")
            print(f"📊 Hidden size: {config['hidden_size']}")
            print(f"📊 Num stacks: {config['num_stacks']}")
            print(f"📊 Num blocks: {config['num_blocks']}")
            print(f"📊 Threshold: {config['threshold']}")
            print(f"📊 Crash weight: {config['crash_weight']}")
            print("=" * 50)
            
            # Start rolling trainer
            self.execute_rolling_training(config)
    
    def execute_rolling_training(self, config):
        """Execute rolling training"""
        try:
            # Import rolling training system
            # The RollingWindowTrainer class is already available in the global scope
            # of the notebook via exec(), so a direct import is not needed here.
            
            # Create trainer
            self.rolling_trainer = RollingWindowTrainer(
                chunks=self.rolling_chunks,
                chunk_size=config['chunk_size'],
                sequence_length=config['sequence_length']
            )
            
            # Progress callback
            def progress_callback(current_cycle, total_cycles, message):
                if self.current_training:
                    progress_percent = int((current_cycle / total_cycles) * 100)
                    self.progress.value = progress_percent
                    self.cycle_info.value = f"Cycle {current_cycle}/{total_cycles}: {message}"
            
            # Execute training
            results = self.rolling_trainer.execute_rolling_training(
                model_type=config['model_type'],
                config=config,
                progress_callback=progress_callback
            )
            
            # Process results
            self.process_training_results(results, config)
            
        except Exception as e:
            with self.output_area:
                print(f"❌ Rolling training error: {e}")
                traceback.print_exc()
        finally:
            self.train_button.disabled = False
            self.stop_button.disabled = True
            self.current_training = None
    
    def process_training_results(self, results, config):
        """Eğitim sonuçlarını işle"""
        if not results:
            with self.output_area:
                print("❌ Eğitim sonuçları alınamadı!")
            return
        
        with self.output_area:
            print(f"\n🎉 Rolling Window Training tamamlandı!")
            print(f"📊 Toplam cycle sayısı: {len(results)}")
            print(f"🏆 En iyi performans analizi:")
            
            # En iyi cycle'ı bul
            best_cycle = None
            best_mae = float('inf')
            
            for result in results:
                if result['performance']['mae'] < best_mae:
                    best_mae = result['performance']['mae']
                    best_cycle = result
            
            if best_cycle:
                print(f"   🥇 En iyi cycle: {best_cycle['cycle']}")
                print(f"   📊 MAE: {best_cycle['performance']['mae']:.4f}")
                print(f"   📊 Accuracy: {best_cycle['performance']['accuracy']:.4f}")
                print(f"   📊 RMSE: {best_cycle['performance']['rmse']:.4f}")
                print(f"   📊 Crash Detection: {best_cycle['performance']['crash_detection']:.4f}")
                
                # En iyi modeli registry'e ekle
                self.model_registry.register_model(
                    best_cycle['model_name'],
                    config['model_type'],
                    config,
                    best_cycle['performance']
                )
                
                print(f"💾 En iyi model registry'e eklendi: {best_cycle['model_name']}")
            
            # Cycle evolution göster
            print(f"\n📈 Cycle Evolution:")
            for i, result in enumerate(results):
                print(f"   Cycle {result['cycle']}: MAE={result['performance']['mae']:.4f}, "
                      f"Acc={result['performance']['accuracy']:.4f}")
            
            # Model dosyalarını listele
            print(f"\n💾 Kaydedilen modeller:")
            for result in results:
                print(f"   - {result['model_name']}")
                print(f"     Path: {result['model_path']}")
                print(f"     Metadata: {result['metadata_path']}")
        
        # Progress tamamlandı
        self.progress.value = 100
        self.cycle_info.value = f"✅ Rolling training tamamlandı! {len(results)} cycle"
        
        # Plot evolution
        self.plot_cycle_evolution(results)
    
    def plot_cycle_evolution(self, results):
        """Cycle evolution grafiği"""
        try:
            import matplotlib.pyplot as plt
            
            cycles = [r['cycle'] for r in results]
            maes = [r['performance']['mae'] for r in results]
            accuracies = [r['performance']['accuracy'] for r in results]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # MAE evolution
            ax1.plot(cycles, maes, marker='o', linewidth=2, markersize=8)
            ax1.set_xlabel('Cycle')
            ax1.set_ylabel('MAE')
            ax1.set_title('MAE Evolution Across Cycles')
            ax1.grid(True, alpha=0.3)
            
            # Accuracy evolution
            ax2.plot(cycles, accuracies, marker='s', linewidth=2, markersize=8, color='green')
            ax2.set_xlabel('Cycle')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Accuracy Evolution Across Cycles')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            with self.output_area:
                print(f"⚠️ Grafik çizim hatası: {e}")
    
    def stop_training(self):
        """Eğitimi durdur"""
        self.train_button.disabled = False
        self.stop_button.disabled = True
        self.current_training = None
        
        with self.output_area:
            print("⏹️ Rolling training durduruldu!")
        
        self.cycle_info.value = "Rolling training durduruldu."
    
    def display_interface(self):
        """Arayüzü göster"""
        # Model seçimi ve parametreler
        model_section = widgets.VBox([
            widgets.HTML("<h3>🔄 Rolling Window Model Seçimi</h3>"),
            self.model_selector,
            widgets.HTML("<br>"),
            widgets.HTML("<h4>⚙️ Rolling Window Parametreleri</h4>"),
            self.chunk_size,
            self.sequence_length,
            self.epochs_per_cycle,
            self.batch_size,
            self.learning_rate,
            widgets.HTML("<br>"),
            widgets.HTML("<h4>🏗️ Model Architecture</h4>"),
            self.hidden_size,
            self.num_stacks,
            self.num_blocks,
            widgets.HTML("<br>"),
            widgets.HTML("<h4>🎯 JetX Specific</h4>"),
            self.threshold,
            self.crash_weight
        ])
        
        # Eğitim kontrolleri
        training_section = widgets.VBox([
            widgets.HTML("<h3>🚀 Rolling Training Kontrolleri</h3>"),
            widgets.HBox([self.train_button, self.stop_button]),
            self.progress,
            self.cycle_info
        ])
        
        # Ana arayüz
        main_interface = widgets.VBox([
            widgets.HTML("<h2>🔄 JetX Rolling Window Trainer</h2>"),
            widgets.HTML("<p>Bu sistem 1000'er kayıtlık chunk'larda rolling window training yapar.</p>"),
            model_section,
            training_section,
            widgets.HTML("<h3>📊 Eğitim Çıktısı</h3>"),
            self.output_area
        ])
        
        display(main_interface)


# Model Download Utilities
def download_trained_models():
    """Download trained models from Colab"""
    try:
        from google.colab import files
        
        models_dir = "/content/trained_models"
        if not os.path.exists(models_dir):
            print("❌ Trained models directory not found!")
            return
        
        # List available models
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        
        if not model_files:
            print("❌ No trained models found!")
            return
        
        print(f"📊 Available models: {len(model_files)}")
        
        # Download each model
        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            print(f"📥 Downloading: {model_file}")
            files.download(model_path)
            
            # Download metadata if exists
            metadata_file = model_file.replace('.pth', '_metadata.json')
            metadata_path = os.path.join(models_dir, metadata_file)
            if os.path.exists(metadata_path):
                files.download(metadata_path)
        
        print("✅ All models downloaded!")
        
    except Exception as e:
        print(f"❌ Download error: {e}")
        traceback.print_exc()  # Detaylı hata kaydı


def create_download_interface():
    """Create download interface"""
    download_button = widgets.Button(
        description='📥 Download All Models',
        button_style='info',
        layout=widgets.Layout(width='200px')
    )
    
    def on_download_click(button):
        download_trained_models()
    
    download_button.on_click(on_download_click)
    
    download_interface = widgets.VBox([
        widgets.HTML("<h3>📥 Model Download</h3>"),
        widgets.HTML("<p>Eğitilmiş modelleri bilgisayarınıza indirin.</p>"),
        download_button
    ])
    
    return download_interface


# Results Analysis
def analyze_training_results():
    """Analyze training results"""
    try:
        models_dir = "/content/trained_models"
        results_files = [f for f in os.listdir(models_dir) if f.startswith('rolling_training_results')]
        
        if not results_files:
            print("❌ No training results found!")
            return
        
        # Load latest results
        latest_results = sorted(results_files)[-1]
        results_path = os.path.join(models_dir, latest_results)
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        print(f"📊 Training Results Analysis")
        print(f"📂 File: {latest_results}")
        print("=" * 50)
        
        for result in results:
            model_type = result['model_type']
            cycles = result['cycles']
            
            print(f"\n🎯 {model_type} Results:")
            print(f"   Total cycles: {len(cycles)}")
            
            # Best performance
            best_cycle = min(cycles, key=lambda x: x['performance']['mae'])
            print(f"   Best cycle: {best_cycle['cycle']}")
            print(f"   Best MAE: {best_cycle['performance']['mae']:.4f}")
            print(f"   Best Accuracy: {best_cycle['performance']['accuracy']:.4f}")
            
            # Performance trend
            maes = [c['performance']['mae'] for c in cycles]
            trend = "📈 Improving" if maes[-1] < maes[0] else "📉 Declining"
            print(f"   Trend: {trend}")
    
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        traceback.print_exc()  # Detaylı hata kaydı


print("✅ Rolling Window Training Integration hazır!")
print("📊 Kullanım:")
print("1. real_trainer = RealJetXTrainingInterface(model_registry, ROLLING_CHUNKS)")
print("2. real_trainer.display_interface()")
