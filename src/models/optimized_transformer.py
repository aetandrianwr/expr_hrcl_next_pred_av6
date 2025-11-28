"""
Final optimized model with frequency-aware embeddings and enhanced architecture.
"""
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyAwareEmbedding(nn.Module):
    """Location embedding that incorporates frequency information."""
    
    def __init__(self, num_embeddings, embedding_dim, freq_map, padding_idx=0):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        
        # Create frequency tensor
        freq_tensor = torch.zeros(num_embeddings)
        for loc_id, freq_val in freq_map.items():
            if loc_id < num_embeddings:
                freq_tensor[loc_id] = freq_val
        self.register_buffer('freq_weights', freq_tensor)
        
        # Learnable frequency projection
        self.freq_proj = nn.Linear(1, embedding_dim)
        
    def forward(self, x):
        # Regular embedding
        emb = self.emb(x)
        
        # Add frequency-based bias
        freq_vals = self.freq_weights[x].unsqueeze(-1)
        freq_emb = self.freq_proj(freq_vals)
        
        return emb + 0.1 * freq_emb  # Small contribution from frequency


class SinusoidalPositionalEncoding(nn.Module):
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


class HierarchicalLocationEncoder(nn.Module):
    def __init__(self, vocab_sizes, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        d_level = d_model // 4
        
        # H3 resolutions
        self.h3_7_emb = nn.Embedding(vocab_sizes['h3_res7'], d_level, padding_idx=0)
        self.h3_8_emb = nn.Embedding(vocab_sizes['h3_res8'], d_level, padding_idx=0)
        
        # S2 levels  
        self.s2_13_emb = nn.Embedding(vocab_sizes['s2_level13'], d_level, padding_idx=0)
        self.s2_14_emb = nn.Embedding(vocab_sizes['s2_level14'], d_level, padding_idx=0)
        
        self.fusion = nn.Sequential(
            nn.Linear(d_level * 4, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, h3_7, h3_8, s2_13, s2_14):
        h7 = self.h3_7_emb(h3_7)
        h8 = self.h3_8_emb(h3_8)
        s13 = self.s2_13_emb(s2_13)
        s14 = self.s2_14_emb(s2_14)
        
        combined = torch.cat([h7, h8, s13, s14], dim=-1)
        return self.fusion(combined)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
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
        x = self.norm2(x + self.ff(x))
        return x


class OptimizedHierarchicalTransformer(nn.Module):
    """Final optimized model."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.D_MODEL
        
        # Load frequency map
        try:
            with open('data/location_freq.json', 'r') as f:
                freq_map = {int(k): v for k, v in json.load(f).items()}
        except:
            freq_map = {}
        
        # Main location embedding with frequency awareness
        self.location_emb = FrequencyAwareEmbedding(
            config.VOCAB_SIZES['location'],
            config.D_MODEL,
            freq_map,
            padding_idx=0
        )
        
        # Hierarchical spatial encoding
        self.spatial_encoder = HierarchicalLocationEncoder(
            config.VOCAB_SIZES,
            config.D_MODEL // 2,
            config.DROPOUT
        )
        
        # User embedding
        self.user_emb = nn.Embedding(config.VOCAB_SIZES['user'], config.D_MODEL // 4, padding_idx=0)
        
        # Temporal encoding
        self.temporal_encoder = TemporalEncoding(config.D_MODEL // 4)
        
        # Duration and diff
        self.duration_proj = nn.Linear(1, config.D_MODEL // 8)
        self.diff_emb = nn.Embedding(config.VOCAB_SIZES['diff'], config.D_MODEL // 8)
        
        # Feature fusion
        total_dim = config.D_MODEL + config.D_MODEL // 2 + config.D_MODEL // 4 + config.D_MODEL // 4 + config.D_MODEL // 8 + config.D_MODEL // 8
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, config.D_MODEL),
            nn.LayerNorm(config.D_MODEL),
            nn.Dropout(config.DROPOUT)
        )
        
        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(config.D_MODEL, config.MAX_SEQ_LEN)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config.D_MODEL, config.N_HEADS, config.D_FF, config.DROPOUT)
            for _ in range(config.N_LAYERS)
        ])
        
        # Output
        self.output_norm = nn.LayerNorm(config.D_MODEL)
        self.output_proj = nn.Linear(config.D_MODEL, config.VOCAB_SIZES['location'])
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, features):
        batch_size = features['location'].shape[0]
        
        # Embeddings
        loc_emb = self.location_emb(features['location'])
        spatial_emb = self.spatial_encoder(
            features['h3_res7'],
            features['h3_res8'],
            features['s2_level13'],
            features['s2_level14']
        )
        user_emb = self.user_emb(features['user'])
        temp_emb = self.temporal_encoder(features['start_min'], features['weekday'])
        dur_emb = self.duration_proj(features['duration'].unsqueeze(-1))
        diff_emb = self.diff_emb(features['diff'])
        
        # Fuse features
        combined = torch.cat([loc_emb, spatial_emb, user_emb, temp_emb, dur_emb, diff_emb], dim=-1)
        x = self.fusion(combined)
        
        # Add positional encoding
        x = x + self.pos_enc(x)
        
        # Transformer layers
        mask = features['mask']
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.output_norm(x)
        
        # Get last valid token
        seq_lens = features['seq_lens']
        last_hidden = x[torch.arange(batch_size), seq_lens - 1]
        
        # Output projection
        logits = self.output_proj(last_hidden)
        
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
