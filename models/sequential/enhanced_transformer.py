import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Concatenate, Conv1D, GlobalMaxPooling1D,
    Embedding, Add, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
import warnings

warnings.filterwarnings('ignore')

class TransformerBlock(tf.keras.layers.Layer):
    """Custom Transformer Block"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="gelu"),
            Dropout(dropout_rate),
            Dense(embed_dim),
        ])
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
        })
        return config


class EnhancedTransformerModel:
    """
    Gelişmiş Transformer modeli - JetX tahmin için optimize edilmiş
    """
    
    def __init__(self, seq_length=200, n_features=1, threshold=1.5):
        self.seq_length = seq_length
        self.n_features = n_features
        self.threshold = threshold
        self.model = None
        self.feature_model = None
        self.history = None
        self.is_fitted = False
        
    def build_model(self,
                   embed_dim=128,
                   num_heads=8,
                   ff_dim=512,
                   num_transformer_blocks=6,
                   dense_units=[256, 128, 64],
                   dropout_rate=0.2,
                   l2_reg=0.001,
                   learning_rate=0.0005,
                   use_positional_encoding=True,
                   use_cnn_features=True):
        """
        Gelişmiş Transformer modeli oluştur
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Attention head sayısı
            ff_dim: Feed-forward dimension
            num_transformer_blocks: Transformer block sayısı
            dense_units: Dense layer boyutları
            dropout_rate: Dropout oranı
            l2_reg: L2 regularization
            learning_rate: Öğrenme oranı
            use_positional_encoding: Positional encoding kullan
            use_cnn_features: CNN özellikleri ekle
        """
        print("Gelişmiş Transformer modeli oluşturuluyor...")
        
        # Input
        inputs = Input(shape=(self.seq_length, self.n_features), name='sequence_input')
        
        # Input projection
        x = Dense(embed_dim, kernel_regularizer=l2(l2_reg))(inputs)
        x = LayerNormalization()(x)
        
        # Positional Encoding
        if use_positional_encoding:
            positions = tf.range(start=0, limit=self.seq_length, delta=1)
            positions = tf.cast(positions, dtype=tf.float32)
            
            # Sinusoidal positional encoding
            def get_angles(pos, i, d_model):
                angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
                return pos * angle_rates
            
            angle_rads = get_angles(
                positions[:, np.newaxis],
                np.arange(embed_dim)[np.newaxis, :],
                embed_dim
            )
            
            # Apply sin to even indices
            angle_rads = tf.cast(angle_rads, dtype=tf.float32)
            sines = tf.sin(angle_rads[:, 0::2])
            cosines = tf.cos(angle_rads[:, 1::2])
            
            pos_encoding = tf.concat([sines, cosines], axis=-1)
            pos_encoding = pos_encoding[tf.newaxis, ...]
            
            x = x + pos_encoding
        
        # CNN Features (local patterns için)
        cnn_features = None
        if use_cnn_features:
            # Multi-scale CNN features
            conv_outputs = []
            
            for kernel_size in [3, 5, 7]:
                conv = Conv1D(
                    filters=64,
                    kernel_size=kernel_size,
                    activation='gelu',
                    padding='same',
                    kernel_regularizer=l2(l2_reg)
                )(inputs)
                conv = BatchNormalization()(conv)
                conv = Dropout(dropout_rate)(conv)
                conv_outputs.append(conv)
            
            # Combine CNN features
            cnn_combined = Concatenate()(conv_outputs)
            cnn_global_avg = GlobalAveragePooling1D()(cnn_combined)
            cnn_global_max = GlobalMaxPooling1D()(cnn_combined)
            cnn_features = Concatenate()([cnn_global_avg, cnn_global_max])
        
        # Transformer Blocks
        for i in range(num_transformer_blocks):
            x = TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate,
                name=f'transformer_block_{i}'
            )(x)
        
        # Global pooling for sequence-level features
        transformer_avg = GlobalAveragePooling1D()(x)
        transformer_max = GlobalMaxPooling1D()(x)
        
        # Attention pooling (weighted average)
        attention_weights = Dense(1, activation='sigmoid')(x)
        attention_pooled = tf.reduce_sum(x * attention_weights, axis=1)
        
        # Combine all sequence features
        sequence_features = Concatenate()([
            transformer_avg,
            transformer_max,
            attention_pooled
        ])
        
        # Combine with CNN features if available
        if cnn_features is not None:
            combined_features = Concatenate()([sequence_features, cnn_features])
        else:
            combined_features = sequence_features
        
        # Feature extraction layer (hibrit sistem için)
        feature_output = Dense(
            128,
            activation='gelu',
            kernel_regularizer=l2(l2_reg),
            name='feature_extraction'
        )(combined_features)
        feature_output = Dropout(dropout_rate)(feature_output)
        feature_output = LayerNormalization()(feature_output)
        
        # Dense layers for prediction
        x = combined_features
        for units in dense_units:
            x = Dense(
                units,
                activation='gelu',
                kernel_regularizer=l2(l2_reg)
            )(x)
            x = Dropout(dropout_rate)(x)
            x = LayerNormalization()(x)
        
        # Output layers
        # Binary classification
        binary_output = Dense(
            1,
            activation='sigmoid',
            name='binary_output'
        )(x)
        
        # Regression
        regression_output = Dense(
            1,
            activation='linear',
            name='regression_output'
        )(x)
        
        # Confidence estimation
        confidence_output = Dense(
            1,
            activation='sigmoid',
            name='confidence_output'
        )(x)
        
        # Main model
        self.model = Model(
            inputs=inputs,
            outputs=[binary_output, regression_output, confidence_output],
            name='enhanced_transformer'
        )
        
        # Feature extraction model
        self.feature_model = Model(
            inputs=inputs,
            outputs=feature_output,
            name='transformer_features'
        )
        
        # Multi-output loss
        losses = {
            'binary_output': 'binary_crossentropy',
            'regression_output': 'huber',
            'confidence_output': 'binary_crossentropy'
        }
        
        loss_weights = {
            'binary_output': 1.0,
            'regression_output': 0.5,
            'confidence_output': 0.3
        }
        
        metrics = {
            'binary_output': ['accuracy'],
            'regression_output': ['mae'],
            'confidence_output': ['accuracy']
        }
        
        # Optimizer (AdamW for better weight decay)
        optimizer = AdamW(
            learning_rate=learning_rate,
            weight_decay=l2_reg,
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
        
        print(f"Transformer modeli oluşturuldu. Parametreler: {self.model.count_params():,}")
        return self.model
    
    def prepare_sequences(self, data):
        """Eğitim verilerini hazırla"""
        X, y_binary, y_regression, y_confidence = [], [], [], []
        
        for i in range(len(data) - self.seq_length):
            seq = data[i:i + self.seq_length]
            next_val = data[i + self.seq_length]
            
            X.append(seq)
            
            # Binary target
            binary_target = 1 if next_val >= self.threshold else 0
            y_binary.append(binary_target)
            
            # Regression target
            y_regression.append(next_val)
            
            # Confidence target (extreme values = high confidence)
            confidence_target = 1 if (next_val >= 2.0 or next_val <= 1.3) else 0
            y_confidence.append(confidence_target)
        
        X = np.array(X).reshape(-1, self.seq_length, self.n_features)
        
        return X, {
            'binary_output': np.array(y_binary),
            'regression_output': np.array(y_regression),
            'confidence_output': np.array(y_confidence)
        }
    
    def fit(self, data, epochs=150, batch_size=32, validation_split=0.2, verbose=1):
        """Model eğitimi"""
        print(f"Enhanced Transformer eğitimi başlıyor... {len(data)} veri noktası")
        
        # Veri hazırlama
        X, y = self.prepare_sequences(data)
        print(f"Hazırlanan sequences: {X.shape}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=8,
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
        
        print("Enhanced Transformer eğitimi tamamlandı!")
        return history
    
    def extract_features(self, sequence):
        """Sequence'den özellik çıkar"""
        if not self.is_fitted or self.feature_model is None:
            return np.array([0.5] * 128)
        
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
            print(f"Transformer feature extraction error: {e}")
            return np.array([0.5] * 128)
    
    def predict_next_value(self, sequence):
        """Sonraki değeri tahmin et"""
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
            binary_pred = predictions[0][0][0]
            regression_pred = predictions[1][0][0]
            confidence_pred = predictions[2][0][0]
            
            return regression_pred, binary_pred, confidence_pred
            
        except Exception as e:
            print(f"Transformer prediction error: {e}")
            return None, 0.5, 0.0
    
    def save(self, filepath):
        """Modeli kaydet"""
        try:
            self.model.save(f"{filepath}_main.h5")
            if self.feature_model:
                self.feature_model.save(f"{filepath}_features.h5")
            print(f"Transformer model kaydedildi: {filepath}")
            return True
        except Exception as e:
            print(f"Transformer kaydetme hatası: {e}")
            return False
    
    def load(self, filepath):
        """Modeli yükle"""
        try:
            # Custom objects için
            custom_objects = {'TransformerBlock': TransformerBlock}
            
            self.model = tf.keras.models.load_model(
                f"{filepath}_main.h5",
                custom_objects=custom_objects
            )
            
            feature_path = f"{filepath}_features.h5"
            if tf.io.gfile.exists(feature_path):
                self.feature_model = tf.keras.models.load_model(
                    feature_path,
                    custom_objects=custom_objects
                )
            
            self.is_fitted = True
            print(f"Transformer model yüklendi: {filepath}")
            return True
        except Exception as e:
            print(f"Transformer yükleme hatası: {e}")
            return False