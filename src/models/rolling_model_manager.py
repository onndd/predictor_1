"""
Rolling Window Model Manager
Ana uygulama ile Colab rolling training entegrasyonu
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

class RollingModelManager:
    """
    Rolling window training sonuçlarını ana uygulama ile entegre eden manager
    """
    
    def __init__(self, models_dir: str = "trained_models", db_path: str = "data/jetx_data.db"):
        self.models_dir = models_dir
        self.db_path = db_path
        self.rolling_models = {}
        self.rolling_results = []
        
        # Directories oluştur
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    def load_rolling_results(self) -> List[Dict]:
        """Colab'dan rolling training sonuçlarını yükle"""
        try:
            results_files = [f for f in os.listdir(self.models_dir) 
                           if f.startswith('rolling_training_results_') and f.endswith('.json')]
            
            if not results_files:
                return []
            
            # En son sonuçları al
            latest_file = sorted(results_files)[-1]
            results_path = os.path.join(self.models_dir, latest_file)
            
            with open(results_path, 'r') as f:
                self.rolling_results = json.load(f)
            
            print(f"✅ Rolling results loaded: {len(self.rolling_results)} training sessions")
            return self.rolling_results
            
        except Exception as e:
            print(f"❌ Error loading rolling results: {e}")
            return []
    
    def get_available_rolling_models(self) -> Dict[str, List[Dict]]:
        """Mevcut rolling modellerini getir"""
        try:
            models_by_type = {}
            
            # Model dosyalarını tara
            model_files = [f for f in os.listdir(self.models_dir) 
                          if f.endswith('.pth') and ('_cycle_' in f)]
            
            for model_file in model_files:
                # Metadata dosyasını kontrol et
                metadata_file = model_file.replace('.pth', '_metadata.json')
                metadata_path = os.path.join(self.models_dir, metadata_file)
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    model_type = metadata['config'].get('model_type', 'Unknown')
                    
                    if model_type not in models_by_type:
                        models_by_type[model_type] = []
                    
                    models_by_type[model_type].append({
                        'model_file': model_file,
                        'metadata': metadata,
                        'performance': metadata.get('performance', {}),
                        'cycle': metadata.get('cycle', 0)
                    })
            
            # Her model type için en iyi modeli bul
            for model_type in models_by_type:
                models_by_type[model_type].sort(
                    key=lambda x: x['performance'].get('mae', float('inf'))
                )
            
            return models_by_type
            
        except Exception as e:
            print(f"❌ Error getting rolling models: {e}")
            return {}
    
    def get_best_rolling_model(self, model_type: str = None) -> Optional[Dict]:
        """En iyi rolling modeli getir"""
        available_models = self.get_available_rolling_models()
        
        if model_type:
            # Belirli model type için en iyisi
            if model_type in available_models and available_models[model_type]:
                return available_models[model_type][0]  # En iyi performanslı
            return None
        else:
            # Genel olarak en iyi model
            best_model = None
            best_mae = float('inf')
            
            for models in available_models.values():
                if models and models[0]['performance'].get('mae', float('inf')) < best_mae:
                    best_mae = models[0]['performance']['mae']
                    best_model = models[0]
            
            return best_model
    
    def load_rolling_model_for_prediction(self, model_info: Dict):
        """Rolling modeli tahmin için yükle"""
        try:
            model_path = os.path.join(self.models_dir, model_info['model_file'])
            model_type = model_info['metadata']['config']['model_type']
            
            # Model tipine göre import
            if model_type == 'N-Beats':
                from models.deep_learning.n_beats.n_beats_model import NBeatsPredictor
                model = NBeatsPredictor(
                    sequence_length=model_info['metadata']['config']['sequence_length'],
                    hidden_size=model_info['metadata']['config'].get('hidden_size', 256),
                    num_stacks=model_info['metadata']['config'].get('num_stacks', 3),
                    num_blocks=model_info['metadata']['config'].get('num_blocks', 3),
                    learning_rate=model_info['metadata']['config']['learning_rate'],
                    threshold=1.5
                )
                
            elif model_type == 'TFT':
                from models.deep_learning.tft.tft_model import TFTPredictor
                model = TFTPredictor(
                    sequence_length=model_info['metadata']['config']['sequence_length'],
                    hidden_size=model_info['metadata']['config'].get('hidden_size', 256),
                    num_heads=model_info['metadata']['config'].get('num_heads', 8),
                    num_layers=model_info['metadata']['config'].get('num_layers', 2),
                    learning_rate=model_info['metadata']['config']['learning_rate']
                )
                
            elif model_type == 'LSTM':
                from models.sequential.lstm_model import LSTMModel
                model = LSTMModel(
                    seq_length=model_info['metadata']['config']['sequence_length'],
                    n_features=1,
                    threshold=1.5
                )
            else:
                print(f"❌ Unsupported model type: {model_type}")
                return None
            
            # Modeli yükle
            model.load_model(model_path)
            model.is_trained = True
            
            self.rolling_models[model_type] = {
                'model': model,
                'info': model_info
            }
            
            print(f"✅ Rolling model loaded: {model_type}")
            return model
            
        except Exception as e:
            print(f"❌ Error loading rolling model: {e}")
            return None
    
    def predict_with_rolling_models(self, sequence: List[float]) -> Dict:
        """Rolling modellerle tahmin yap"""
        results = {}
        
        for model_type, model_data in self.rolling_models.items():
            try:
                model = model_data['model']
                sequence_length = model_data['info']['metadata']['config']['sequence_length']
                
                # Sequence uzunluğunu kontrol et
                if len(sequence) < sequence_length:
                    print(f"⚠️ Sequence too short for {model_type}: need {sequence_length}, got {len(sequence)}")
                    continue
                
                # Son sequence_length kadar veri al
                model_sequence = sequence[-sequence_length:]
                
                # Tahmin yap
                value, prob, conf = model.predict_with_confidence(model_sequence)
                
                results[model_type] = {
                    'value': value,
                    'probability': prob,
                    'confidence': conf,
                    'performance': model_data['info']['performance']
                }
                
            except Exception as e:
                print(f"❌ Error predicting with {model_type}: {e}")
        
        return results
    
    def get_ensemble_rolling_prediction(self, sequence: List[float]) -> Dict:
        """Rolling modellerden ensemble tahmin"""
        predictions = self.predict_with_rolling_models(sequence)
        
        if not predictions:
            return {
                'ensemble_value': None,
                'ensemble_confidence': 0.0,
                'model_count': 0,
                'individual_predictions': {}
            }
        
        # Weighted ensemble (MAE'ye göre ağırlık)
        weights = []
        values = []
        confidences = []
        
        for model_type, pred in predictions.items():
            mae = pred['performance'].get('mae', 1.0)
            weight = 1.0 / (mae + 0.001)  # Lower MAE = higher weight
            
            weights.append(weight)
            values.append(pred['value'])
            confidences.append(pred['confidence'])
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Weighted average
        ensemble_value = sum(w * v for w, v in zip(weights, values))
        ensemble_confidence = sum(w * c for w, c in zip(weights, confidences))
        
        return {
            'ensemble_value': ensemble_value,
            'ensemble_confidence': ensemble_confidence,
            'model_count': len(predictions),
            'individual_predictions': predictions,
            'weights': dict(zip(predictions.keys(), weights))
        }
    
    def get_rolling_training_summary(self) -> Dict:
        """Rolling training özeti getir"""
        if not self.rolling_results:
            self.load_rolling_results()
        
        summary = {
            'total_sessions': len(self.rolling_results),
            'model_types': [],
            'best_performances': {},
            'training_dates': []
        }
        
        for session in self.rolling_results:
            model_type = session['model_type']
            summary['model_types'].append(model_type)
            summary['training_dates'].append(session.get('timestamp', ''))
            
            if 'cycles' in session and session['cycles']:
                best_cycle = min(session['cycles'], key=lambda x: x['performance']['mae'])
                
                if model_type not in summary['best_performances']:
                    summary['best_performances'][model_type] = best_cycle['performance']
                else:
                    if best_cycle['performance']['mae'] < summary['best_performances'][model_type]['mae']:
                        summary['best_performances'][model_type] = best_cycle['performance']
        
        summary['model_types'] = list(set(summary['model_types']))
        
        return summary
    
    def export_rolling_models_to_main_app(self) -> Dict:
        """Rolling modelleri ana uygulamaya export et"""
        try:
            available_models = self.get_available_rolling_models()
            export_info = {}
            
            for model_type, models in available_models.items():
                if models:
                    best_model = models[0]  # En iyi performanslı
                    
                    export_info[model_type] = {
                        'model_path': os.path.join(self.models_dir, best_model['model_file']),
                        'config': best_model['metadata']['config'],
                        'performance': best_model['performance'],
                        'cycle': best_model['metadata'].get('cycle', 0)
                    }
            
            # Export bilgilerini kaydet
            export_path = os.path.join(self.models_dir, 'rolling_models_export.json')
            with open(export_path, 'w') as f:
                json.dump(export_info, f, indent=2)
            
            print(f"✅ Rolling models exported: {len(export_info)} models")
            return export_info
            
        except Exception as e:
            print(f"❌ Export error: {e}")
            return {}
