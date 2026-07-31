import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .positional_encoding import RotaryEmbedding

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, max_seq_len=1024, attention_dropout=0.1, use_rope=True, use_flash=True):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.resid_drop = nn.Dropout(attention_dropout)
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
        self.register_buffer("causal_mask", torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1,1,max_seq_len,max_seq_len))
        nn.init.normal_(self.qkv_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02/math.sqrt(2))

    def forward(self, x, attention_mask=None, past_kv=None, use_cache=False):
        B, T, _ = x.shape
        q, k, v = self.qkv_proj(x).split(self.hidden_size, dim=-1)
        q = q.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        k = k.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        v = v.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        if self.use_rope:
            q, k = self.rope(q, k, seq_len=T)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v) if use_cache else None
        S = k.shape[2]
        if self.use_flash:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=self.attn_drop.p if self.training else 0.0, is_causal=(past_kv is None))
        else:
            w = (torch.matmul(q, k.transpose(-2,-1)) / self.scale)
            if past_kv is None:
                w = w.masked_fill(self.causal_mask[:,:,:T,:S]==0, float("-inf"))
            if attention_mask is not None:
                w = w + attention_mask
            w = self.attn_drop(F.softmax(w, dim=-1))
            out = torch.matmul(w, v)
        out = out.transpose(1,2).contiguous().view(B,T,self.hidden_size)
        return self.resid_drop(self.out_proj(out)), new_kv
