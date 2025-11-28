"""
Hierarchical Transformer model for next-location prediction.

This model uses:
1. Hierarchical location encoding (H3 multi-resolution)
2. Rich temporal features (time of day, day of week, duration, time gaps)
3. User embeddings
4. Multi-head self-attention with positional encoding
5. Feature-aware fusion mechanisms
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Positional encoding of shape (batch_size, seq_len, d_model)
        """
        return self.pe[:x.size(1), :].unsqueeze(0)


class TemporalEncoding(nn.Module):
    """
    Temporal encoding for time-of-day and day-of-week.
    Uses sinusoidal encoding for cyclical time features.
    """
    
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, time_minutes, weekday):
        """
        Args:
            time_minutes: (batch, seq_len) - minutes since midnight [0, 1439]
            weekday: (batch, seq_len) - day of week [0, 6]
        Returns:
            Temporal encoding (batch, seq_len, d_model)
        """
        batch_size, seq_len = time_minutes.shape
        device = time_minutes.device
        
        # Normalize time to [0, 2π]
        time_norm = (time_minutes.float() / 1440.0) * 2 * math.pi
        weekday_norm = (weekday.float() / 7.0) * 2 * math.pi
        
        # Create encoding
        encoding = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        # Use half dimensions for time-of-day, half for day-of-week
        d_half = self.d_model // 2
        
        # Time-of-day encoding
        for i in range(d_half // 2):
            encoding[:, :, 2*i] = torch.sin(time_norm * (i + 1))
            encoding[:, :, 2*i + 1] = torch.cos(time_norm * (i + 1))
        
        # Day-of-week encoding
        for i in range(d_half // 2):
            encoding[:, :, d_half + 2*i] = torch.sin(weekday_norm * (i + 1))
            encoding[:, :, d_half + 2*i + 1] = torch.cos(weekday_norm * (i + 1))
        
        return encoding


class HierarchicalLocationEncoder(nn.Module):
    """
    Hierarchical location encoder using multi-resolution spatial indices.
    Combines H3 resolutions to capture location at different granularities.
    """
    
    def __init__(self, vocab_sizes, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Embedding for each H3 resolution
        d_each = d_model // 4  # Divide embedding space
        self.h3_res5_emb = nn.Embedding(vocab_sizes['h3_res5'], d_each, padding_idx=0)
        self.h3_res6_emb = nn.Embedding(vocab_sizes['h3_res6'], d_each, padding_idx=0)
        self.h3_res7_emb = nn.Embedding(vocab_sizes['h3_res7'], d_each, padding_idx=0)
        self.h3_res8_emb = nn.Embedding(vocab_sizes['h3_res8'], d_each, padding_idx=0)
        
        # Layer to fuse hierarchical representations
        self.fusion = nn.Sequential(
            nn.Linear(d_each * 4, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, h3_res5, h3_res6, h3_res7, h3_res8):
        """
        Args:
            h3_res5-8: (batch, seq_len) - H3 indices at different resolutions
        Returns:
            Hierarchical location encoding (batch, seq_len, d_model)
        """
        emb5 = self.h3_res5_emb(h3_res5)
        emb6 = self.h3_res6_emb(h3_res6)
        emb7 = self.h3_res7_emb(h3_res7)
        emb8 = self.h3_res8_emb(h3_res8)
        
        # Concatenate and fuse
        combined = torch.cat([emb5, emb6, emb7, emb8], dim=-1)
        fused = self.fusion(combined)
        
        return fused


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) - attention mask
        """
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=~mask if mask is not None else None)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class HierarchicalTransformer(nn.Module):
    """
    Hierarchical Transformer for next-location prediction.
    
    Features:
    - Hierarchical location encoding (H3 multi-resolution)
    - Temporal encoding (time-of-day, day-of-week)
    - Duration and time-gap features
    - User embeddings
    - Multi-head self-attention
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.D_MODEL
        
        # Hierarchical location encoder
        self.location_encoder = HierarchicalLocationEncoder(
            config.VOCAB_SIZES,
            config.D_MODEL,
            config.DROPOUT
        )
        
        # User embedding
        self.user_emb = nn.Embedding(config.VOCAB_SIZES['user'], config.D_MODEL // 4, padding_idx=0)
        
        # Temporal encoding
        self.temporal_encoder = TemporalEncoding(config.D_MODEL // 2)
        
        # Duration and time-gap embeddings
        self.duration_proj = nn.Linear(1, config.D_MODEL // 4)
        self.diff_emb = nn.Embedding(config.VOCAB_SIZES['diff'], config.D_MODEL // 4)
        
        # Feature fusion layer
        # Combines: location (d_model) + user (d_model//4) + temporal (d_model//2) + duration (d_model//4) + diff (d_model//4)
        total_dim = config.D_MODEL + config.D_MODEL // 4 + config.D_MODEL // 2 + config.D_MODEL // 4 + config.D_MODEL // 4
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_dim, config.D_MODEL),
            nn.LayerNorm(config.D_MODEL),
            nn.Dropout(config.DROPOUT)
        )
        
        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(config.D_MODEL, config.MAX_SEQ_LEN)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(config.D_MODEL, config.N_HEADS, config.D_FF, config.DROPOUT)
            for _ in range(config.N_LAYERS)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(config.D_MODEL)
        self.output_proj = nn.Linear(config.D_MODEL, config.VOCAB_SIZES['location'])
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, features):
        """
        Args:
            features: Dictionary containing:
                - h3_res5-8: (batch, seq_len)
                - user: (batch, seq_len)
                - start_min: (batch, seq_len)
                - weekday: (batch, seq_len)
                - duration: (batch, seq_len)
                - diff: (batch, seq_len)
                - mask: (batch, seq_len)
        Returns:
            logits: (batch, num_locations)
        """
        # Extract features
        batch_size, seq_len = features['h3_res5'].shape
        
        # Hierarchical location encoding
        loc_enc = self.location_encoder(
            features['h3_res5'],
            features['h3_res6'],
            features['h3_res7'],
            features['h3_res8']
        )
        
        # User embedding
        user_enc = self.user_emb(features['user'])
        
        # Temporal encoding
        temp_enc = self.temporal_encoder(features['start_min'], features['weekday'])
        
        # Duration encoding
        dur_enc = self.duration_proj(features['duration'].unsqueeze(-1))
        
        # Time gap encoding
        diff_enc = self.diff_emb(features['diff'])
        
        # Fuse all features
        combined = torch.cat([loc_enc, user_enc, temp_enc, dur_enc, diff_enc], dim=-1)
        x = self.feature_fusion(combined)
        
        # Add positional encoding
        x = x + self.pos_encoder(x)
        
        # Apply Transformer encoder layers
        mask = features['mask']
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # Use the last valid token for each sequence
        x = self.output_norm(x)
        
        # Get last valid position for each sequence
        seq_lens = features['seq_lens']
        last_hidden = x[torch.arange(batch_size), seq_lens - 1]
        
        # Project to location vocabulary
        logits = self.output_proj(last_hidden)
        
        return logits

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
