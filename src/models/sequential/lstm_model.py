"""
Modern LSTM Model Implementation for JetX Prediction System
Enhanced with attention mechanisms, residual connections, and multi-output predictions
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Bidirectional, 
    LayerNormalization, Add, MultiHeadAttention,
    GlobalAveragePooling1D, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')

class JetXLSTMFeatureExtractor:
    """
    JetX-specific feature extraction for LSTM
    """
    def __init__(self, threshold: float = 1.5):
        self.threshold = threshold
    
    def extract_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Extract JetX-specific features from sequence
        
        Args:
            sequence: Input sequence
            
        Returns:
            Enhanced feature array
        """
        seq = np.array(sequence)
        features = []
        
        # Original values
        features.append(seq)
        
        # Moving averages
        ma_5 = self._moving_average(seq, 5)
        ma_10 = self._moving_average(seq, 10)
        features.extend([ma_5, ma_10])
        
        # Momentum
        momentum = self._calculate_momentum(seq)
        features.append(momentum)
        
        # Volatility
        volatility = self._calculate_volatility(seq)
        features.append(volatility)
        
        # Threshold indicator
        threshold_indicator = (seq >= self.threshold).astype(float)
        features.append(threshold_indicator)
        
        return np.stack(features, axis=-1)
    
    def _moving_average(self, sequence: np.ndarray, window: int) -> np.ndarray:
        """Calculate moving average"""
        result = np.zeros_like(sequence)
        for i in range(len(sequence)):
            start = max(0, i - window + 1)
            end = min(len(sequence), i + 1)
            result[i] = np.mean(sequence[start:end])
        return result
    
    def _calculate_momentum(self, sequence: np.ndarray) -> np.ndarray:
        """Calculate momentum"""
        momentum = np.zeros_like(sequence)
        for i in range(1, len(sequence)):
            momentum[i] = (sequence[i] - sequence[i-1]) / sequence[i-1] if sequence[i-1] != 0 else 0
        return momentum
    
    def _calculate_volatility(self, sequence: np.ndarray) -> np.ndarray:
        """Calculate rolling volatility"""
        volatility = np.zeros_like(sequence)
        window = 10
        for i in range(len(sequence)):
            start = max(0, i - window + 1)
            window_seq = sequence[start:i+1]
            if len(window_seq) > 1:
                volatility[i] = np.std(window_seq)
        return volatility

class ModernLSTMModel:
    def __init__(self, seq_length=200, n_features=1, threshold=1.5, 
                 multi_output=True, use_attention=True, use_residual=True):
        """
        Modern LSTM model - optimized for JetX
        
        Args:
            seq_length: Sequence length
            n_features: Number of features
            threshold: Threshold value
            multi_output: Multi-output prediction
            use_attention: Use attention mechanism
            use_residual: Use residual connections
        """
        self.seq_length = seq_length
        self.n_features = n_features
        self.threshold = threshold
        self.multi_output = multi_output
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.model = None
        self.history = None
        self.is_binary = True  # Binary classification (above/below threshold)
        
        # JetX-specific feature extractor
        self.feature_extractor = JetXLSTMFeatureExtractor(threshold=threshold)

    def build_model(self, lstm_units=128, dropout_rate=0.2, learning_rate=0.001,
                   attention_heads=8, num_layers=2):
        """
        Build modern LSTM model
        Args:
            lstm_units: LSTM units count
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            attention_heads: Attention heads count
            num_layers: LSTM layers count
        """
        # Input - enhanced features will be 6 features per timestep
        enhanced_features = 6 if self.multi_output else self.n_features
        inputs = Input(shape=(self.seq_length, enhanced_features))
        
        # Multi-layer Bidirectional LSTM with residual connections
        x = inputs
        lstm_outputs = []
        
        for i in range(num_layers):
            # Bidirectional LSTM
            lstm_out = Bidirectional(
                LSTM(lstm_units, return_sequences=True, dropout=dropout_rate/2,
                     recurrent_dropout=dropout_rate/2, kernel_regularizer=l2(0.01))
            )(x)
            
            # Layer normalization
            lstm_out = LayerNormalization()(lstm_out)
            
            # Residual connection (if enabled and dimensions match)
            if self.use_residual and i > 0 and lstm_out.shape[-1] == x.shape[-1]:
                lstm_out = Add()([lstm_out, x])
            
            # Dropout
            lstm_out = Dropout(dropout_rate)(lstm_out)
            
            lstm_outputs.append(lstm_out)
            x = lstm_out
        
        # Attention mechanism
        if self.use_attention:
            attention_output = self._add_attention_layer(x, attention_heads)
            x = attention_output
        
        # Global features
        global_avg = GlobalAveragePooling1D()(x)
        global_max = tf.keras.layers.GlobalMaxPooling1D()(x)
        
        # Last timestep
        last_timestep = x[:, -1, :]
        
        # Combine features
        combined = Concatenate()([global_avg, global_max, last_timestep])
        
        # Dense layers with residual connections
        dense1 = Dense(lstm_units, activation='relu', kernel_regularizer=l2(0.01))(combined)
        dense1 = LayerNormalization()(dense1)
        dense1 = Dropout(dropout_rate)(dense1)
        
        dense2 = Dense(lstm_units // 2, activation='relu', kernel_regularizer=l2(0.01))(dense1)
        dense2 = LayerNormalization()(dense2)
        dense2 = Dropout(dropout_rate)(dense2)
        
        # Multi-output predictions
        if self.multi_output:
            # Value prediction
            value_output = Dense(1, activation='linear', name='value_output')(dense2)
            
            # Probability prediction (above threshold)
            prob_output = Dense(1, activation='sigmoid', name='probability_output')(dense2)
            
            # Confidence prediction
            confidence_output = Dense(1, activation='sigmoid', name='confidence_output')(dense2)
            
            # Crash risk prediction
            crash_risk_output = Dense(1, activation='sigmoid', name='crash_risk_output')(dense2)
            
            outputs = {
                'value_output': value_output,
                'probability_output': prob_output,
                'confidence_output': confidence_output,
                'crash_risk_output': crash_risk_output
            }
            
            losses = {
                'value_output': 'mse',
                'probability_output': 'binary_crossentropy',
                'confidence_output': 'binary_crossentropy',
                'crash_risk_output': 'binary_crossentropy'
            }
            
            loss_weights = {
                'value_output': 0.5,
                'probability_output': 0.3,
                'confidence_output': 0.1,
                'crash_risk_output': 0.3
            }
            
            metrics = {
                'value_output': ['mae'],
                'probability_output': ['accuracy'],
                'confidence_output': ['accuracy'],
                'crash_risk_output': ['accuracy']
            }
        else:
            # Single output (backward compatibility)
            if self.is_binary:
                outputs = Dense(1, activation='sigmoid')(dense2)
                losses = 'binary_crossentropy'
                metrics = ['accuracy']
                loss_weights = None
            else:
                outputs = Dense(1, activation='linear')(dense2)
                losses = 'mse'
                metrics = ['mae']
                loss_weights = None
        
        # Model oluştur
        model = Model(inputs=inputs, outputs=outputs)
        
        # Modeli derle
        optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=losses, metrics=metrics, loss_weights=loss_weights)
        
        self.model = model
        return model
    
    def _add_attention_layer(self, x, num_heads):
        """Add multi-head attention layer"""
        # Multi-head self-attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=x.shape[-1] // num_heads,
            dropout=0.1
        )(x, x)
        
        # Residual connection
        attention_output = Add()([x, attention_output])
        
        # Layer normalization
        attention_output = LayerNormalization()(attention_output)
        
        return attention_output

    def prepare_sequences(self, data):
        """
        Eğitim dizilerini hazırlar - Enhanced version with tuple handling
        
        Args:
            data: Veri dizisi (can be floats or (id, value) tuples)
            
        Returns:
            tuple: X (girdi dizileri) ve y (hedef değerler)
        """
        # Handle tuple data
        if data and isinstance(data[0], (tuple, list)):
            processed_data = [float(item[1]) for item in data]
        else:
            processed_data = [float(item) for item in data]
        
        X, y = [], []
        
        for i in range(len(processed_data) - self.seq_length):
            # Dizi
            seq = processed_data[i:i+self.seq_length]
            
            # Enhanced features
            if self.multi_output:
                enhanced_seq = self.feature_extractor.extract_features(seq)
                X.append(enhanced_seq)
            else:
                X.append(np.array(seq).reshape(-1, 1))
            
            # Sonraki değer
            next_val = processed_data[i+self.seq_length]
            
            if self.multi_output:
                # Multi-output targets
                y_dict = {
                    'value_output': next_val,
                    'probability_output': 1.0 if next_val >= self.threshold else 0.0,
                    'confidence_output': 1.0,  # Will be calculated during training
                    'crash_risk_output': 1.0 if next_val < self.threshold else 0.0
                }
                y.append(y_dict)
            else:
                # Single output
                if self.is_binary:
                    y.append(1 if next_val >= self.threshold else 0)
                else:
                    y.append(next_val)
        
        X = np.array(X)
        
        if self.multi_output:
            # Convert to multi-output format
            y_value = np.array([item['value_output'] for item in y])
            y_prob = np.array([item['probability_output'] for item in y])
            y_conf = np.array([item['confidence_output'] for item in y])
            y_crash = np.array([item['crash_risk_output'] for item in y])
            
            y = {
                'value_output': y_value,
                'probability_output': y_prob,
                'confidence_output': y_conf,
                'crash_risk_output': y_crash
            }
        else:
            y = np.array(y)
        
        print(f"🔧 LSTM: Prepared sequences shape: {X.shape}")
        print(f"🔧 LSTM: Prepared targets type: {type(y)}")
        
        return X, y

    def fit(self, X, y, epochs=100, batch_size=32, validation_split=0.2, verbose=1):
        """
        Modeli eğitir - Enhanced version
        
        Args:
            X: Girdi dizileri
            y: Hedef değerler
            epochs: Epoch sayısı
            batch_size: Batch boyutu
            validation_split: Doğrulama seti oranı
            verbose: Çıktı detay seviyesi
            
        Returns:
            history: Eğitim geçmişi
        """
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Modeli eğit
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.history = history
        return history

    def predict(self, X):
        """
        Tahmin yapar
        
        Args:
            X: Girdi dizisi
            
        Returns:
            Tahminler
        """
        return self.model.predict(X)

    def predict_next(self, sequence):
        """
        Bir sonraki değeri tahmin eder - Enhanced version
        
        Args:
            sequence: Girdi dizisi
            
        Returns:
            tuple: (tahmin değeri, eşik üstü olasılığı, güven skoru)
        """
        # Boyut kontrolü
        if len(sequence) < self.seq_length:
            # Yeterli veri yoksa, mevcut dizi ile doldur
            padding = [sequence[0]] * (self.seq_length - len(sequence))
            sequence = padding + list(sequence)
        
        # Son seq_length değeri al
        sequence = sequence[-self.seq_length:]
        
        # Enhanced features
        if self.multi_output:
            enhanced_seq = self.feature_extractor.extract_features(sequence)
            X = enhanced_seq.reshape(1, self.seq_length, -1)
        else:
            X = np.array(sequence).reshape(1, self.seq_length, self.n_features)
        
        # Tahmin
        prediction = self.model.predict(X, verbose=0)
        
        if self.multi_output:
            # Multi-output predictions
            value_pred = prediction['value_output'][0][0]
            prob_pred = prediction['probability_output'][0][0]
            conf_pred = prediction['confidence_output'][0][0]
            crash_risk = prediction['crash_risk_output'][0][0]
            
            return value_pred, prob_pred, conf_pred, crash_risk
        else:
            # Single output
            pred_value = prediction[0][0]
            
            if self.is_binary:
                # İkili sınıflandırma: Eşik üstü olasılığı
                return None, pred_value, 0.5
            else:
                # Regresyon: Tahmin değeri ve eşik üstü olma durumu
                above_threshold = pred_value >= self.threshold
                return pred_value, 1.0 if above_threshold else 0.0, 0.5

    def predict_with_confidence(self, sequence):
        """
        Make a prediction with confidence metrics - required for rolling training system
        
        Args:
            sequence: Input sequence
            
        Returns:
            tuple: (predicted_value, above_threshold_probability, confidence_score)
        """
        try:
            if self.multi_output:
                value_pred, prob_pred, conf_pred, crash_risk = self.predict_next(sequence)
                return float(value_pred), float(prob_pred), float(conf_pred)
            else:
                result = self.predict_next(sequence)
                if len(result) == 3:
                    return result
                else:
                    # Handle different return formats
                    if result[0] is not None:
                        return float(result[0]), float(result[1]), float(result[2])
                    else:
                        return 1.5, float(result[1]), 0.5
        except Exception as e:
            raise RuntimeError(f"LSTM prediction failed: {str(e)}")
    
    def predict_next_value(self, sequence):
        """
        Compatibility method for ensemble systems
        
        Args:
            sequence: Input sequence
            
        Returns:
            tuple: (predicted_value, above_threshold_probability, confidence)
        """
        return self.predict_with_confidence(sequence)
    
    def train(self, data, epochs=100, batch_size=32, validation_split=0.2, verbose=True):
        """
        Train the LSTM model - compatibility method for rolling training system
        
        Args:
            data: Training data
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            verbose: Whether to print training progress
            
        Returns:
            Training history
        """
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Prepare data
        X, y = self.prepare_sequences(data)
        
        # Train model
        history = self.fit(X, y, epochs, batch_size, validation_split, verbose)
        
        return {'train_losses': history.history.get('loss', []), 'val_losses': history.history.get('val_loss', [])}
    
    def save_model(self, filepath):
        """Save the trained model - compatibility method"""
        self.save(filepath)
    
    def load_model(self, filepath):
        """Load a trained model - compatibility method"""
        self.load(filepath)

    def evaluate(self, X, y):
        """
        Model performansını değerlendirir
        
        Args:
            X: Test girdileri
            y: Test hedefleri
            
        Returns:
            Evaluation results
        """
        return self.model.evaluate(X, y)

    def save(self, filepath):
        """
        Modeli kaydeder
        
        Args:
            filepath: Dosya yolu
        """
        self.model.save(filepath)

    def load(self, filepath):
        """
        Modeli yükler
        
        Args:
            filepath: Dosya yolu
        """
        self.model = tf.keras.models.load_model(filepath)

    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model:
            return self.model.summary()
        return "Model not built yet"

    def get_training_history(self):
        """Get training history"""
        if self.history:
            return self.history.history
        return None


# Backward compatibility
class LSTMModel(ModernLSTMModel):
    def __init__(self, seq_length=200, n_features=1, threshold=1.5):
        super().__init__(seq_length, n_features, threshold, 
                        multi_output=False, use_attention=False, use_residual=False)
