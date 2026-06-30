import json
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

# ========================================== #
# 1. CONFIGURATION
# ========================================== #
class Config:
    drop = 0.1
    dim = 512
    heads = 8
    layers = 12
    ffn_dim = dim * 2
    max_seq_len = 225  # ~ 30s / 0.15s = 200 + 25 for safety
    classes_num = 1000

    batch_size = 4
    epochs = 10
    lr = 3e-4

# ========================================== #
# 2. CORE TRANSFORMER COMPONENTS
# ========================================== #
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=225, base=10000.0):
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
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

class XSA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.heads = config.heads
        self.d_k = self.dim // self.heads

        self.Wq = nn.Linear(self.dim, self.dim, bias=False)
        self.Wk = nn.Linear(self.dim, self.dim, bias=False)
        self.Wv = nn.Linear(self.dim, self.dim, bias=False)
        self.Wo = nn.Linear(self.dim, self.dim, bias=False)
        self.rope = RotaryEmbedding(self.d_k)
        self.attn_dropout_p = config.drop

    def forward(self, x, padding_mask=None):
        B, T, D = x.shape

        Q = self.Wq(x).view(B, T, self.heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, T, self.heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.heads, self.d_k).transpose(1, 2)

        Q, K = self.rope(Q, K, T)

        attn_mask = None
        if padding_mask is not None:
            attn_mask = padding_mask.view(B, 1, 1, T)

        att_output = F.scaled_dot_product_attention(
            Q, K, V, 
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=False  
        )

        Vn = F.normalize(V, dim=-1)
        att_output = att_output - (att_output * Vn).sum(dim=-1, keepdim=True) * Vn
        att_output = att_output.transpose(1, 2).contiguous().view(B, T, D)
        return self.Wo(att_output)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.dim, 2 * config.ffn_dim, bias=False)
        self.swish = nn.SiLU() 
        self.fc2 = nn.Linear(config.ffn_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.drop)

    def forward(self, x):
        x = self.fc1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * self.swish(gate)
        x = self.fc2(x)
        return self.dropout(x)

class DyT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Y = nn.Parameter(torch.ones(dim))
        self.a = nn.Parameter(torch.ones(dim))
        self.s = nn.Parameter(torch.zeros(dim))
        self.d = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.Y * torch.tanh(self.a * x + self.s) + self.d

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = DyT(config.dim)
        self.attn = XSA(config)
        self.ln_2 = DyT(config.dim)
        self.mlp = MLP(config)

    def forward(self, x, padding_mask=None):
        x = x + self.attn(self.ln_1(x), padding_mask=padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

# ========================================== #
# 3. CONTINUOUS MEDIA PIPE EMBEDDER
# ========================================== #
class LandmarkEmbedding(nn.Module):
    def __init__(self, input_dim=1659, dim=512, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, dim)
        self.norm = DyT(dim) 
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # Maps raw floats: (B, T, 1659) -> (B, T, 512)
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.drop(x)

# ========================================== #
# 4. FULL SIGN LANGUAGE BERT
# ========================================== #
class SignLanguageBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Replaced Token Embedding with Landmark Embedding
        self.landmark_embeddings = LandmarkEmbedding(input_dim=1659, dim=config.dim, drop=config.drop)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim))
        
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.ln_f = DyT(config.dim)
        
        self.pooler_linear = nn.Linear(config.dim, config.dim)
        self.pooler_tanh = nn.Tanh()
        self.classifier = nn.Linear(config.dim, config.classes_num)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, DyT):
            nn.init.ones_(module.Y)
            nn.init.ones_(module.a)
            nn.init.zeros_(module.s)
            nn.init.zeros_(module.d)

        if hasattr(self, 'cls_token'):
             torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

        for name, param in module.named_parameters():
            if name in ['Wo.weight', 'fc2.weight']:
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / (2 * self.config.layers)**0.5)

    def forward(self, frames, padding_mask=None, labels=None):
        B, T, _ = frames.shape
        
        # 1. Project continuous 3D landmarks
        x = self.landmark_embeddings(frames) 

        # 2. Append CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 3. Adjust mask for CLS
        if padding_mask is not None:
            cls_mask = torch.ones((B, 1), device=padding_mask.device, dtype=padding_mask.dtype)
            padding_mask = torch.cat((cls_mask, padding_mask), dim=1)

        # 4. Transform
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)
            
        x = self.ln_f(x)

        # 5. Extract CLS representation (Index 0) and classify
        cls_out = x[:, 0, :]
        pooled_output = self.pooler_tanh(self.pooler_linear(cls_out))
        logits = self.classifier(pooled_output)
        
        # 6. Compute Loss (Required by HF Trainer)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.classes_num), labels.view(-1))
            
        return {
            "loss": loss,
            "logits": logits,
            "last_hidden_state": x,    
            "pooler_output": pooled_output 
        }
