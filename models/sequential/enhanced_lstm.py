import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization, 
    Attention, MultiHeadAttention, LayerNormalization,
    Concatenate, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import warnings

warnings.filterwarnings('ignore')

class EnhancedLSTMModel:
    """
    Gelişmiş LSTM modeli - Feature extraction ve prediction için optimize edilmiş
    """
    
    def __init__(self, seq_length=200, n_features=1, threshold=1.5):
        self.seq_length = seq_length
        self.n_features = n_features
        self.threshold = threshold
        self.model = None
        self.feature_model = None  # Feature extraction için ayrı model
        self.history = None
        self.is_fitted = False
        
    def build_model(self, 
                   lstm_units=[128, 64, 32], 
                   dense_units=[64, 32],
                   dropout_rate=0.3,
                   l2_reg=0.001,
                   learning_rate=0.001,
                   use_attention=True,
                   use_cnn_features=True):
        """
        Gelişmiş LSTM modeli oluştur
        
        Args:
            lstm_units: LSTM layer boyutları
            dense_units: Dense layer boyutları
            dropout_rate: Dropout oranı
            l2_reg: L2 regularization
            learning_rate: Öğrenme oranı
            use_attention: Attention mechanism kullan
            use_cnn_features: CNN özelliklerini ekle
        """
        print("Gelişmiş LSTM modeli oluşturuluyor...")
        
        # Input layer
        inputs = Input(shape=(self.seq_length, self.n_features), name='main_input')
        
        # Normalization
        x = LayerNormalization()(inputs)
        
        # CNN Features (isteğe bağlı)
        cnn_features = None
        if use_cnn_features:
            # 1D Convolution patterns için
            conv1 = Conv1D(64, 3, activation='relu', padding='same')(x)
            conv1 = Dropout(dropout_rate)(conv1)
            
            conv2 = Conv1D(32, 5, activation='relu', padding='same')(conv1)
            conv2 = Dropout(dropout_rate)(conv2)
            
            # Global pooling
            cnn_global_max = GlobalMaxPooling1D()(conv2)
            cnn_global_avg = GlobalAveragePooling1D()(conv2)
            cnn_features = Concatenate()([cnn_global_max, cnn_global_avg])
        
        # Multi-layer LSTM
        lstm_outputs = []
        for i, units in enumerate(lstm_units):
            return_sequences = (i < len(lstm_units) - 1) or use_attention
            
            if i == 0:
                lstm_out = LSTM(
                    units, 
                    return_sequences=return_sequences,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate,
                    kernel_regularizer=l2(l2_reg),
                    name=f'lstm_{i}'
                )(x)
            else:
                lstm_out = LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate,
                    kernel_regularizer=l2(l2_reg),
                    name=f'lstm_{i}'
                )(lstm_out)
            
            lstm_out = BatchNormalization()(lstm_out)
            
            # Her LSTM layer'dan output al
            if return_sequences and use_attention:
                lstm_outputs.append(lstm_out)
            elif not return_sequences:
                # Son layer
                final_lstm = lstm_out
        
        # Attention Mechanism
        attention_output = None
        if use_attention and lstm_outputs:
            # Self-attention
            attention_layer = MultiHeadAttention(
                num_heads=8,
                key_dim=lstm_units[-1] // 8,
                name='multi_head_attention'
            )
            
            # Son LSTM output üzerinde attention
            last_lstm = lstm_outputs[-1]
            attention_output = attention_layer(last_lstm, last_lstm)
            
            # Attention pooling
            attention_avg = tf.reduce_mean(attention_output, axis=1)
            attention_max = tf.reduce_max(attention_output, axis=1)
            attention_features = Concatenate()([attention_avg, attention_max])
        
        # Feature concatenation
        features_list = []
        
        if 'final_lstm' in locals():
            features_list.append(final_lstm)
            
        if attention_output is not None:
            features_list.append(attention_features)
            
        if cnn_features is not None:
            features_list.append(cnn_features)
        
        # Combine all features
        if len(features_list) > 1:
            combined_features = Concatenate(name='feature_concat')(features_list)
        else:
            combined_features = features_list[0]
        
        # Feature extraction output (hibrit sistem için)
        feature_output = Dense(
            64, 
            activation='relu',
            kernel_regularizer=l2(l2_reg),
            name='feature_dense'
        )(combined_features)
        feature_output = Dropout(dropout_rate)(feature_output)
        feature_output = BatchNormalization()(feature_output)
        
        # Dense layers for final prediction
        x = combined_features
        for units in dense_units:
            x = Dense(
                units, 
                activation='relu',
                kernel_regularizer=l2(l2_reg)
            )(x)
            x = Dropout(dropout_rate)(x)
            x = BatchNormalization()(x)
        
        # Output layers
        # Binary classification (threshold üstü/altı)
        binary_output = Dense(
            1, 
            activation='sigmoid',
            name='binary_output'
        )(x)
        
        # Regression (değer tahmini)
        regression_output = Dense(
            1, 
            activation='linear',
            name='regression_output'
        )(x)
        
        # Ana model (eğitim için)
        self.model = Model(
            inputs=inputs,
            outputs=[binary_output, regression_output],
            name='enhanced_lstm_model'
        )
        
        # Feature extraction modeli (hibrit sistem için)
        self.feature_model = Model(
            inputs=inputs,
            outputs=feature_output,
            name='feature_extractor'
        )
        
        # Multi-output loss
        losses = {
            'binary_output': 'binary_crossentropy',
            'regression_output': 'huber'  # Outlier'lara karşı robust
        }
        
        loss_weights = {
            'binary_output': 1.0,
            'regression_output': 0.5
        }
        
        metrics = {
            'binary_output': ['accuracy'],
            'regression_output': ['mae']
        }
        
        # Optimizer
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )
        
        # Compile
        self.model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
        
        print(f"Model oluşturuldu. Parametreler: {self.model.count_params():,}")
        return self.model
    
    def prepare_sequences(self, data):
        """Eğitim verilerini hazırla"""
        X, y_binary, y_regression = [], [], []
        
        for i in range(len(data) - self.seq_length):
            # Input sequence
            seq = data[i:i + self.seq_length]
            next_val = data[i + self.seq_length]
            
            X.append(seq)
            
            # Binary target (threshold üstü/altı)
            y_binary.append(1 if next_val >= self.threshold else 0)
            
            # Regression target (gerçek değer)
            y_regression.append(next_val)
        
        X = np.array(X).reshape(-1, self.seq_length, self.n_features)
        y_binary = np.array(y_binary)
        y_regression = np.array(y_regression)
        
        return X, {'binary_output': y_binary, 'regression_output': y_regression}
    
    def fit(self, data, epochs=200, batch_size=64, validation_split=0.2, verbose=1):
        """Model eğitimi"""
        print(f"Enhanced LSTM eğitimi başlıyor... {len(data)} veri noktası")
        
        # Veri hazırlama
        X, y = self.prepare_sequences(data)
        print(f"Hazırlanan sequences: {X.shape}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Eğitim
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.history = history
        self.is_fitted = True
        
        print("Enhanced LSTM eğitimi tamamlandı!")
        return history
    
    def extract_features(self, sequence):
        """
        Sequence'den özellik çıkar (hibrit sistem için)
        
        Args:
            sequence: Input sequence
            
        Returns:
            np.array: Extracted features
        """
        if not self.is_fitted or self.feature_model is None:
            return np.array([0.5] * 64)  # Default features
        
        try:
            # Sequence preparation
            if len(sequence) < self.seq_length:
                padding = [sequence[0]] * (self.seq_length - len(sequence))
                sequence = padding + list(sequence)
            
            sequence = sequence[-self.seq_length:]
            X = np.array(sequence).reshape(1, self.seq_length, self.n_features)
            
            # Feature extraction
            features = self.feature_model.predict(X, verbose=0)[0]
            return features
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.array([0.5] * 64)
    
    def predict_next_value(self, sequence):
        """
        Sonraki değeri tahmin et
        
        Returns:
            tuple: (predicted_value, above_threshold_probability, confidence)
        """
        if not self.is_fitted:
            return None, 0.5, 0.0
        
        try:
            # Sequence preparation
            if len(sequence) < self.seq_length:
                padding = [sequence[0]] * (self.seq_length - len(sequence))
                sequence = padding + list(sequence)
            
            sequence = sequence[-self.seq_length:]
            X = np.array(sequence).reshape(1, self.seq_length, self.n_features)
            
            # Prediction
            predictions = self.model.predict(X, verbose=0)
            binary_pred = predictions[0][0][0]  # Threshold üstü probability
            regression_pred = predictions[1][0][0]  # Predicted value
            
            # Confidence (binary prediction'ın kesinliği)
            confidence = abs(binary_pred - 0.5) * 2  # 0.5'ten uzaklık
            confidence = max(0.0, min(1.0, confidence))
            
            return regression_pred, binary_pred, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.5, 0.0
    
    def evaluate_model(self, data, verbose=1):
        """Model performansını değerlendir"""
        if not self.is_fitted:
            print("Model eğitilmemiş!")
            return None
        
        X, y = self.prepare_sequences(data)
        results = self.model.evaluate(X, y, verbose=verbose)
        
        return {
            'total_loss': results[0],
            'binary_loss': results[1],
            'regression_loss': results[2],
            'binary_accuracy': results[3],
            'regression_mae': results[4]
        }
    
    def save(self, filepath):
        """Modeli kaydet"""
        try:
            # Ana modeli kaydet
            self.model.save(f"{filepath}_main.h5")
            
            # Feature extraction modelini kaydet
            if self.feature_model:
                self.feature_model.save(f"{filepath}_features.h5")
            
            print(f"Model kaydedildi: {filepath}")
            return True
            
        except Exception as e:
            print(f"Model kaydetme hatası: {e}")
            return False
    
    def load(self, filepath):
        """Modeli yükle"""
        try:
            # Ana modeli yükle
            self.model = tf.keras.models.load_model(f"{filepath}_main.h5")
            
            # Feature extraction modelini yükle
            feature_path = f"{filepath}_features.h5"
            if tf.io.gfile.exists(feature_path):
                self.feature_model = tf.keras.models.load_model(feature_path)
            
            self.is_fitted = True
            print(f"Model yüklendi: {filepath}")
            return True
            
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            return False