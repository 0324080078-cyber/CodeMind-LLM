import torch
import torch.nn as nn
from typing import Optional, Tuple
from .attention import CausalSelfAttention
from .feed_forward import SwiGLUFFN

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, max_seq_len=1024, dropout=0.1, attention_dropout=0.1, layer_norm_epsilon=1e-5, use_rope=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.attn = CausalSelfAttention(hidden_size, num_heads, max_seq_len, attention_dropout, use_rope)
        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.ffn = SwiGLUFFN(hidden_size, intermediate_size, dropout)

    def forward(self, x, attention_mask=None, past_kv=None, use_cache=False):
        attn_out, new_kv = self.attn(self.ln1(x), attention_mask=attention_mask, past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_kv
