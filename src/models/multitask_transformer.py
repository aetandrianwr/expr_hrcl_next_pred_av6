"""
Improved Hierarchical Transformer with Multi-Task Learning.

Key improvements:
1. Multi-task learning with hierarchical location predictions
2. Better feature fusion with attention
3. More effective use of both H3 and S2 spatial hierarchies
4. Auxiliary losses to guide training
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
        return self.pe[:x.size(1), :].unsqueeze(0)


class TemporalEncoding(nn.Module):
    """Temporal encoding for time features."""
    
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, time_minutes, weekday):
        batch_size, seq_len = time_minutes.shape
        device = time_minutes.device
        
        time_norm = (time_minutes.float() / 1440.0) * 2 * math.pi
        weekday_norm = (weekday.float() / 7.0) * 2 * math.pi
        
        encoding = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        
        d_half = self.d_model // 2
        
        for i in range(d_half // 2):
            encoding[:, :, 2*i] = torch.sin(time_norm * (i + 1))
            encoding[:, :, 2*i + 1] = torch.cos(time_norm * (i + 1))
        
        for i in range(d_half // 2):
            encoding[:, :, d_half + 2*i] = torch.sin(weekday_norm * (i + 1))
            encoding[:, :, d_half + 2*i + 1] = torch.cos(weekday_norm * (i + 1))
        
        return encoding


class LocationEmbedding(nn.Module):
    """
    Hierarchical location embedding using both H3 and S2.
    Uses attention to fuse different resolutions.
    """
    
    def __init__(self, vocab_sizes, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Reduce dimension per level to fit budget
        d_level = d_model // 8
        
        # H3 embeddings
        self.h3_res5_emb = nn.Embedding(vocab_sizes['h3_res5'], d_level, padding_idx=0)
        self.h3_res6_emb = nn.Embedding(vocab_sizes['h3_res6'], d_level, padding_idx=0)
        self.h3_res7_emb = nn.Embedding(vocab_sizes['h3_res7'], d_level, padding_idx=0)
        self.h3_res8_emb = nn.Embedding(vocab_sizes['h3_res8'], d_level, padding_idx=0)
        
        # S2 embeddings
        self.s2_l11_emb = nn.Embedding(vocab_sizes['s2_level11'], d_level, padding_idx=0)
        self.s2_l13_emb = nn.Embedding(vocab_sizes['s2_level13'], d_level, padding_idx=0)
        self.s2_l14_emb = nn.Embedding(vocab_sizes['s2_level14'], d_level, padding_idx=0)
        self.s2_l15_emb = nn.Embedding(vocab_sizes['s2_level15'], d_level, padding_idx=0)
        
        # Attention-based fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_level * 8, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, h3_5, h3_6, h3_7, h3_8, s2_11, s2_13, s2_14, s2_15):
        # Get all embeddings
        h5 = self.h3_res5_emb(h3_5)
        h6 = self.h3_res6_emb(h3_6)
        h7 = self.h3_res7_emb(h3_7)
        h8 = self.h3_res8_emb(h3_8)
        
        s11 = self.s2_l11_emb(s2_11)
        s13 = self.s2_l13_emb(s2_13)
        s14 = self.s2_l14_emb(s2_14)
        s15 = self.s2_l15_emb(s2_15)
        
        # Concatenate all levels
        combined = torch.cat([h5, h6, h7, h8, s11, s13, s14, s15], dim=-1)
        
        # Fuse with transformation
        fused = self.fusion(combined)
        
        return fused


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer."""
    
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
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=~mask if mask is not None else None)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x


class MultiTaskHierarchicalTransformer(nn.Module):
    """
    Multi-task hierarchical Transformer.
    
    Predicts both the main location and hierarchical auxiliary targets
    to improve learning.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.D_MODEL
        
        # Location embedding
        self.location_encoder = LocationEmbedding(
            config.VOCAB_SIZES,
            config.D_MODEL,
            config.DROPOUT
        )
        
        # User embedding
        self.user_emb = nn.Embedding(config.VOCAB_SIZES['user'], config.D_MODEL // 8, padding_idx=0)
        
        # Temporal encoding
        self.temporal_encoder = TemporalEncoding(config.D_MODEL // 4)
        
        # Duration and time-gap
        self.duration_proj = nn.Linear(1, config.D_MODEL // 8)
        self.diff_emb = nn.Embedding(config.VOCAB_SIZES['diff'], config.D_MODEL // 8)
        
        # Feature fusion
        total_dim = config.D_MODEL + config.D_MODEL // 8 + config.D_MODEL // 4 + config.D_MODEL // 8 + config.D_MODEL // 8
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_dim, config.D_MODEL),
            nn.LayerNorm(config.D_MODEL),
            nn.Dropout(config.DROPOUT)
        )
        
        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(config.D_MODEL, config.MAX_SEQ_LEN)
        
        # Transformer layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(config.D_MODEL, config.N_HEADS, config.D_FF, config.DROPOUT)
            for _ in range(config.N_LAYERS)
        ])
        
        # Output heads
        self.output_norm = nn.LayerNorm(config.D_MODEL)
        
        # Main prediction head
        self.output_proj = nn.Linear(config.D_MODEL, config.VOCAB_SIZES['location'])
        
        # Auxiliary prediction heads for hierarchical locations
        self.h3_res7_head = nn.Linear(config.D_MODEL, config.VOCAB_SIZES['h3_res7'])
        self.s2_l13_head = nn.Linear(config.D_MODEL, config.VOCAB_SIZES['s2_level13'])
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, features, return_aux=False):
        batch_size, seq_len = features['h3_res5'].shape
        
        # Location encoding
        loc_enc = self.location_encoder(
            features['h3_res5'],
            features['h3_res6'],
            features['h3_res7'],
            features['h3_res8'],
            features['s2_level11'],
            features['s2_level13'],
            features['s2_level14'],
            features['s2_level15']
        )
        
        # Other features
        user_enc = self.user_emb(features['user'])
        temp_enc = self.temporal_encoder(features['start_min'], features['weekday'])
        dur_enc = self.duration_proj(features['duration'].unsqueeze(-1))
        diff_enc = self.diff_emb(features['diff'])
        
        # Fuse all features
        combined = torch.cat([loc_enc, user_enc, temp_enc, dur_enc, diff_enc], dim=-1)
        x = self.feature_fusion(combined)
        
        # Add positional encoding
        x = x + self.pos_encoder(x)
        
        # Apply Transformer layers
        mask = features['mask']
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        x = self.output_norm(x)
        
        # Get last valid position
        seq_lens = features['seq_lens']
        last_hidden = x[torch.arange(batch_size), seq_lens - 1]
        
        # Main prediction
        logits = self.output_proj(last_hidden)
        
        if return_aux:
            # Auxiliary predictions
            h3_res7_logits = self.h3_res7_head(last_hidden)
            s2_l13_logits = self.s2_l13_head(last_hidden)
            return logits, h3_res7_logits, s2_l13_logits
        
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
