"""
Informer Model Implementation for Long Sequence Time Series Forecasting
Based on the paper: Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math
import warnings
warnings.filterwarnings('ignore')

class ProbAttention(nn.Module):
    """
    Probabilistic Attention mechanism for Informer
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        Sample top-k queries and keys for efficient attention
        """
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        
        # Sample k keys for each query
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(0, L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        
        # Find the top-k queries
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        
        # Use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        
        return Q_K, M_top
    
    def _get_initial_context(self, V, L_Q):
        """
        Get initial context for attention
        """
        B, H, L_V, D = V.shape
        V_sum = V.mean(dim=-2)
        context = V_sum.unsqueeze(-2).expand(B, H, L_Q, D)
        return context
    
    def forward(self, queries, keys, values, attn_mask):
        """
        Forward pass for probabilistic attention
        """
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)
        
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()
        
        U_part = min(U_part, L_K)
        u = min(u, L_Q)
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        
        # Add scale factor
        scale = self.scale or 1. / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        
        # Get the context
        context = self._get_initial_context(values, L_Q)
        
        # Update the context with selected top-k
        values_in = torch.matmul(queries, values.transpose(-2, -1))
        context_in = torch.matmul(scores_top, values_in)
        context = context + context_in
        
        return context.transpose(2, 1), None

class EncoderLayer(nn.Module):
    """
    Encoder layer for Informer
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(self, x, attn_mask=None):
        # Multi-head attention
        new_x, attn = self.attention(
            x, x, x, attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        
        # Position-wise feed-forward
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm2(x + y), attn

class Encoder(nn.Module):
    """
    Encoder for Informer
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        
    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x, attns

class DecoderLayer(nn.Module):
    """
    Decoder layer for Informer
    """
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Self-attention
        x = x + self.dropout(self.self_attention(
            x, x, x, attn_mask=x_mask
        ))
        x = self.norm1(x)
        
        # Cross-attention
        x = x + self.dropout(self.cross_attention(
            x, cross, cross, attn_mask=cross_mask
        ))
        x = self.norm2(x)
        
        # Position-wise feed-forward
        y = x = self.norm3(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm3(x + y)

class Decoder(nn.Module):
    """
    Decoder for Informer
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        
        if self.norm is not None:
            x = self.norm(x)
        
        if self.projection is not None:
            x = self.projection(x)
        
        return x

class InformerModel(nn.Module):
    """
    Complete Informer Model
    """
    def __init__(self, input_size=1, output_size=1, d_model=512, n_heads=8, 
                 e_layers=2, d_layers=1, d_ff=2048, factor=5, dropout=0.1, 
                 activation='gelu', output_attention=False, distil=True, 
                 mix=True, seq_len=200, label_len=50, pred_len=1):
        super(InformerModel, self).__init__()
        
        # Encoding
        self.enc_embedding = nn.Linear(input_size, d_model)
        self.dec_embedding = nn.Linear(input_size, d_model)
        
        # Attention
        Attn = ProbAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                    d_model, d_ff, dropout=dropout, activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )
        
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    Attn(True, factor, attention_dropout=dropout, output_attention=False),
                    Attn(False, factor, attention_dropout=dropout, output_attention=False),
                    d_model, d_ff, dropout=dropout, activation=activation
                ) for l in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, output_size, bias=True)
        )
        
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        Forward pass
        
        Args:
            x_enc: Encoder input
            x_mark_enc: Encoder time features
            x_dec: Decoder input
            x_mark_dec: Decoder time features
            enc_self_mask: Encoder self-attention mask
            dec_self_mask: Decoder self-attention mask
            dec_enc_mask: Decoder-encoder attention mask
        """
        # Encoder
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        
        # Decoder
        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        
        return dec_out, attns

class InformerPredictor:
    """
    Informer based predictor for JetX time series
    """
    def __init__(self, sequence_length: int = 200, d_model: int = 512, n_heads: int = 8,
                 e_layers: int = 2, d_layers: int = 1, learning_rate: float = 0.001):
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.learning_rate = learning_rate
        
        # Initialize model
        self.model = InformerModel(
            input_size=1,
            output_size=1,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            seq_len=sequence_length,
            label_len=sequence_length // 4,
            pred_len=1
        )
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training state
        self.is_trained = False
        self.training_history = []
        
    def prepare_sequences(self, data: List[float]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare sequences for training
        
        Args:
            data: List of time series values
            
        Returns:
            Tuple of (enc_input, dec_input, enc_target, dec_target)
        """
        enc_inputs = []
        dec_inputs = []
        enc_targets = []
        dec_targets = []
        
        for i in range(len(data) - self.sequence_length - 1):
            # Encoder input
            enc_seq = data[i:i + self.sequence_length]
            enc_inputs.append(enc_seq)
            enc_targets.append(data[i + self.sequence_length])
            
            # Decoder input (use last part of encoder sequence)
            dec_seq = data[i + self.sequence_length - self.sequence_length // 4:i + self.sequence_length]
            dec_inputs.append(dec_seq)
            dec_targets.append(data[i + self.sequence_length])
        
        return (torch.tensor(enc_inputs, dtype=torch.float32).unsqueeze(-1),
                torch.tensor(dec_inputs, dtype=torch.float32).unsqueeze(-1),
                torch.tensor(enc_targets, dtype=torch.float32),
                torch.tensor(dec_targets, dtype=torch.float32))
    
    def train(self, data: List[float], epochs: int = 100, batch_size: int = 32, 
              validation_split: float = 0.2, verbose: bool = True) -> dict:
        """
        Train the Informer model
        
        Args:
            data: Training data
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            verbose: Whether to print training progress
            
        Returns:
            Training history
        """
        # Prepare data
        enc_inputs, dec_inputs, enc_targets, dec_targets = self.prepare_sequences(data)
        
        # Split into train and validation
        split_idx = int(len(enc_inputs) * (1 - validation_split))
        enc_train, enc_val = enc_inputs[:split_idx], enc_inputs[split_idx:]
        dec_train, dec_val = dec_inputs[:split_idx], dec_inputs[split_idx:]
        enc_target_train, enc_target_val = enc_targets[:split_idx], enc_targets[split_idx:]
        dec_target_train, dec_target_val = dec_targets[:split_idx], dec_targets[split_idx:]
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_train_loss = 0
            num_batches = 0
            
            for i in range(0, len(enc_train), batch_size):
                batch_enc = enc_train[i:i + batch_size]
                batch_dec = dec_train[i:i + batch_size]
                batch_enc_target = enc_target_train[i:i + batch_size]
                batch_dec_target = dec_target_train[i:i + batch_size]
                
                # Create dummy time features
                batch_enc_mark = torch.zeros_like(batch_enc)
                batch_dec_mark = torch.zeros_like(batch_dec)
                
                self.optimizer.zero_grad()
                predictions, _ = self.model(batch_enc, batch_enc_mark, batch_dec, batch_dec_mark)
                loss = self.criterion(predictions.squeeze(), batch_dec_target)
                loss.backward()
                self.optimizer.step()
                
                total_train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = total_train_loss / num_batches
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_predictions, _ = self.model(enc_val, torch.zeros_like(enc_val), 
                                              dec_val, torch.zeros_like(dec_val))
                val_loss = self.criterion(val_predictions.squeeze(), dec_target_val).item()
            
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {val_loss:.6f}")
        
        self.is_trained = True
        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        return self.training_history
    
    def predict(self, sequence: List[float]) -> float:
        """
        Make a prediction
        
        Args:
            sequence: Input sequence of length sequence_length
            
        Returns:
            Predicted next value
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if len(sequence) != self.sequence_length:
            raise ValueError(f"Sequence must have length {self.sequence_length}")
        
        self.model.eval()
        with torch.no_grad():
            # Prepare encoder and decoder inputs
            enc_input = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            dec_input = torch.tensor(sequence[-self.sequence_length // 4:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            
            # Create dummy time features
            enc_mark = torch.zeros_like(enc_input)
            dec_mark = torch.zeros_like(dec_input)
            
            prediction, _ = self.model(enc_input, enc_mark, dec_input, dec_mark)
        
        return prediction.item()
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'model_config': {
                'sequence_length': self.sequence_length,
                'd_model': self.d_model,
                'n_heads': self.n_heads,
                'e_layers': self.e_layers,
                'd_layers': self.d_layers
            }
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        self.is_trained = checkpoint['is_trained']