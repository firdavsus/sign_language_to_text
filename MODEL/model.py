import json
import os
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

# ==========================================
# 1. CONFIGURATION & HELPERS
# ==========================================
class Config:
    drop = 0.1
    drop_path_rate = 0.1
    # INCREASED CAPACITY:
    dim = 384          # Jumped from 256 to 384
    heads = 6          # Jumped from 4 to 6
    layers = 6         # Jumped from 4 to 6        
    ffn_dim = 384 * 4  # Fixed ratio (4x dim)
    max_seq_len = 256  # Note: Subsampling will reduce actual seq length fed to attention
    classes_num = 1001

    batch_size = 128
    accum = 1
    epochs = 20
    lr = 3e-4 
    warmup_steps = 240

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# ==========================================
# 2. TRANSFORMER / CONFORMER MODULES
# ==========================================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=256, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q, k, seq_len):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        cos = self.cos_cached[:seq_len].view(1, 1, seq_len, -1)
        sin = self.sin_cached[:seq_len].view(1, 1, seq_len, -1)
        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class XSA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim, self.heads = config.dim, config.heads
        self.d_k = self.dim // self.heads
        self.Wq = nn.Linear(self.dim, self.dim, bias=False)
        self.Wk = nn.Linear(self.dim, self.dim, bias=False)
        self.Wv = nn.Linear(self.dim, self.dim, bias=False)
        self.Wo = nn.Linear(self.dim, self.dim, bias=False)
        self.rope = RotaryEmbedding(self.d_k)
        self.attn_dropout = nn.Dropout(config.drop)

    def forward(self, x, padding_mask=None):
        B, T, D = x.shape
        Q = self.Wq(x).view(B, T, self.heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, T, self.heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.heads, self.d_k).transpose(1, 2)
        Q, K = self.rope(Q, K, T)
        
        attn_mask = padding_mask.view(B, 1, 1, T) if padding_mask is not None else None

        att_output = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask, 
            dropout_p=self.attn_dropout.p if self.training else 0.0
        )
        
        return self.Wo(att_output.transpose(1, 2).contiguous().view(B, T, D))

class FeedForward(nn.Module):
    """Standard FFN, renamed for the Macaron Conformer layout"""
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.dim, 2 * config.ffn_dim, bias=False)
        self.fc2 = nn.Linear(config.ffn_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.drop)

    def forward(self, x):
        x = self.fc1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * F.silu(gate)
        return self.dropout(self.fc2(x))

class ConvModule(nn.Module):
    """Depthwise Temporal Convolution - Captures fine hand motion across frames"""
    def __init__(self, config):
        super().__init__()
        self.ln = nn.LayerNorm(config.dim)
        self.pw1 = nn.Linear(config.dim, 2 * config.dim)
        # Wide kernel (31) for long temporal context. Note: sequences must be >= 31 frames.
        self.dw  = nn.Conv1d(config.dim, config.dim, kernel_size=31, 
                             padding=15, groups=config.dim) 
        self.bn  = nn.BatchNorm1d(config.dim)
        self.pw2 = nn.Linear(config.dim, config.dim)
        self.drop = nn.Dropout(config.drop)
    
    def forward(self, x):
        x = self.ln(x)
        x, gate = self.pw1(x).chunk(2, dim=-1)
        x = x * F.silu(gate)
        # Transpose for Conv1d: (B, T, C) -> (B, C, T)
        x = self.dw(x.transpose(1, 2))
        x = self.bn(x).transpose(1, 2)
        return self.drop(self.pw2(x))

class ConformerBlock(nn.Module):
    """Convolution-Augmented Transformer"""
    def __init__(self, config, dpr=0.0):
        super().__init__()
        # Macaron-style half-step FFNs
        self.ff1 = FeedForward(config)
        self.attn = XSA(config)
        self.conv = ConvModule(config)
        self.ff2 = FeedForward(config)
        self.ln = nn.LayerNorm(config.dim)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()

    def forward(self, x, padding_mask=None):
        x = x + self.drop_path(0.5 * self.ff1(x))
        x = x + self.drop_path(self.attn(self.ln(x), padding_mask=padding_mask))
        x = x + self.drop_path(self.conv(x))
        x = x + self.drop_path(0.5 * self.ff2(x))
        return x

class TemporalSubsample(nn.Module):
    """Reduces temporal length to create 'phoneme-like' chunks (stride=2 -> 2x reduction)"""
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(config.dim, config.dim, kernel_size=5, stride=2, padding=2, groups=config.dim),
            nn.BatchNorm1d(config.dim),
            nn.SiLU(),
        )

    def forward(self, x, padding_mask=None):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2) # Back to (B, T', C)

        # Scale down the padding mask to match the new sequence length
        if padding_mask is not None:
            mask_float = padding_mask.float().unsqueeze(1) # (B, 1, T)
            mask_pooled = F.max_pool1d(mask_float, kernel_size=5, stride=2, padding=2)
            padding_mask = mask_pooled.squeeze(1).bool() # (B, T')

        return x, padding_mask

# ==========================================
# 3. SPATIAL & TEMPORAL MODULES
# ==========================================
# (SEBlock, Res2Net, AttentiveStatsPooling, MotionBlock remain largely the same, 
#  but omitted the purely duplicate code to save space. Paste them back here.)

class SEBlock(nn.Module):
    def __init__(self, input_shape: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_shape, input_shape // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_shape // reduction, input_shape, bias=False),
            nn.Sigmoid()
        )

    def forward(self, X): 
        b, c, t = X.size() 
        y = self.avg_pool(X).view(b, c)
        y = self.fc(y).view(b, c, 1) 
        return X * y.expand_as(X)

class Res2Net(nn.Module):
    def __init__(self, hidden: int, dilation: int, scale: int = 4):
        super().__init__()
        assert hidden % scale == 0, "Hidden channels must be divisible by scale"
        self.width = hidden // scale
        self.scale = scale
        self.convs = nn.ModuleList([
            nn.Conv1d(
                self.width, self.width, 
                kernel_size=3, stride=1, 
                padding=dilation, dilation=dilation, bias=False
            ) for _ in range(scale - 1) 
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(scale - 1)])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        xs = torch.chunk(X, self.scale, dim=1)
        ys = []
        for i in range(self.scale):
            if i == 0:
                ys.append(xs[i])
            elif i == 1:
                ys.append(self.relu(self.bns[i-1](self.convs[i-1](xs[i]))))
            else:
                combined = xs[i] + ys[-1]
                ys.append(self.relu(self.bns[i-1](self.convs[i-1](combined))))
        return torch.cat(ys, dim=1)

class AttentiveStatsPooling(nn.Module):
    def __init__(self, input_shape: int, hidden: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(input_shape * 3, hidden, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden, input_shape, kernel_size=1)
        )

    def forward(self, X, padding_mask=None):
        B, C, T = X.shape
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).float()
            valid_frames = mask.sum(dim=2, keepdim=True).clamp(min=1.0)
            mean = (X * mask).sum(dim=2, keepdim=True) / valid_frames
            variance = (((X - mean) ** 2) * mask).sum(dim=2, keepdim=True) / valid_frames
            std = torch.sqrt(variance.clamp(min=1e-9))
        else:
            mean = torch.mean(X, dim=2, keepdim=True)         
            std = torch.std(X, dim=2, keepdim=True) + 1e-9    
        
        mean_ext = mean.expand_as(X)
        std_ext = std.expand_as(X)
        X_extended = torch.cat([X, mean_ext, std_ext], dim=1)
        
        logits = self.attention(X_extended)
        if padding_mask is not None:
            logits = logits.masked_fill(~padding_mask.unsqueeze(1), float('-inf'))
            
        alpha = F.softmax(logits, dim=2)
        weighted_mean = torch.sum(alpha * X, dim=2)
        weighted_var = torch.sum(alpha * (X**2), dim=2) - (weighted_mean**2)
        weighted_std = torch.sqrt(weighted_var.clamp(min=1e-9))

        return torch.cat([weighted_mean, weighted_std], dim=1)

class MotionBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj_in = nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False)
        self.res2net = Res2Net(hidden=out_dim, dilation=2, scale=4)
        self.se = SEBlock(input_shape=out_dim, reduction=8)
        self.proj_out = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.GELU()
        )

    def forward(self, x):
        x = self.proj_in(x)
        residual = x
        x = self.res2net(x)
        x = self.se(x)
        x = self.proj_out(x)
        return x + residual

class AdvancedSpatialEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        sub_dim = config.dim // 4 
        
        self.face_block = MotionBlock(120 * 2, sub_dim) 
        self.pose_block = MotionBlock(99 * 2, sub_dim)
        self.lh_block = MotionBlock(63 * 2, sub_dim)
        self.rh_block = MotionBlock(63 * 2, sub_dim)

        self.fusion = nn.Sequential(
            nn.Linear(sub_dim * 4, config.dim),
            nn.LayerNorm(config.dim),
            nn.GELU(),
            nn.Dropout(config.drop)
        )

        # TO-DO: Ensure these indices capture lips, eyebrows, and eye apertures!
        face_indices = torch.tensor([
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            276, 283, 282, 295, 285, 46, 53, 52, 65, 55,
            263, 249, 390, 373, 374, 33, 7, 163, 144, 145
        ], dtype=torch.long)
        self.register_buffer('face_indices', face_indices)

    def compute_kinematics(self, x):
        velocity = torch.zeros_like(x)
        velocity[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        return torch.cat([x, velocity], dim=-1)

    def forward(self, x):
        B, T, _ = x.shape
        face_raw = x[:, :, :1434]
        pose = x[:, :, 1434:1533]
        lh_raw = x[:, :, 1533:1596]
        rh_raw = x[:, :, 1596:1659]

        face_3d = face_raw.view(B, T, 478, 3)
        face_decimated = face_3d[:, :, self.face_indices, :]
        face = face_decimated.view(B, T, 120)

        lh_wrist = lh_raw[:, :, :3].repeat(1, 1, 21)
        lh = lh_raw - lh_wrist
        rh_wrist = rh_raw[:, :, :3].repeat(1, 1, 21)
        rh = rh_raw - rh_wrist

        face = self.compute_kinematics(face)
        pose = self.compute_kinematics(pose)
        lh = self.compute_kinematics(lh)
        rh = self.compute_kinematics(rh)

        face = face.transpose(1, 2)
        pose = pose.transpose(1, 2)
        lh = lh.transpose(1, 2)
        rh = rh.transpose(1, 2)

        f_emb = self.face_block(face)
        p_emb = self.pose_block(pose)
        lh_emb = self.lh_block(lh)
        rh_emb = self.rh_block(rh)

        combined = torch.cat([f_emb, p_emb, lh_emb, rh_emb], dim=1) 
        combined = combined.transpose(1, 2)
        
        return self.fusion(combined)

# ==========================================
# 4. MAIN MODEL
# ==========================================
class SignLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Spatial Embedder
        self.spatial_embedder = AdvancedSpatialEmbedder(config) 
        
        # 2. Temporal Subsampling (Reduces sequence length by 2x)
        self.temporal_subsample = TemporalSubsample(config)
        
        # 3. Conformer Blocks (Replaced standard Transformer)
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.layers)]
        self.blocks = nn.ModuleList([ConformerBlock(config, dpr=dpr[i]) for i in range(config.layers)])
        
        # 4. Global Pooling & Classifier
        self.pooler = AttentiveStatsPooling(input_shape=config.dim, hidden=config.dim // 2)
        self.ln_f = nn.LayerNorm(config.dim * 2) 
        self.classifier = nn.Linear(config.dim * 2, config.classes_num)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None: 
                nn.init.zeros_(module.bias)

        if module == self.classifier:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: 
                nn.init.zeros_(module.bias)

    def forward(self, frames, padding_mask=None, labels=None, **kwargs):
        # A) Spatial Embedding
        x = self.spatial_embedder(frames) 
        
        # B) Temporal Subsampling
        x, padding_mask = self.temporal_subsample(x, padding_mask)

        # C) Conformer Encoding
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)
            
        # D) Pooling & Classification
        x_t = x.transpose(1, 2)
        pooled = self.pooler(x_t, padding_mask=padding_mask)
        pooled = self.ln_f(pooled) 
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            
        return {"loss": loss, "logits": logits}