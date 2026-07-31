import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import os, json

from .config import CodeMindConfig
from .transformer_block import TransformerBlock

class CodeMindLLM(nn.Module):
    def __init__(self, config: CodeMindConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(config.hidden_size, config.num_attention_heads, config.intermediate_size,
                config.max_position_embeddings, config.dropout, config.attention_dropout, config.layer_norm_epsilon)
            for _ in range(config.num_hidden_layers)
        ])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.apply(self._init)
        n = sum(p.numel() for p in self.parameters())
        print(f"CodeMind: {n:,} params ({n*4/1024**2:.1f} MB)")

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=self.config.initializer_range)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=self.config.initializer_range)
            if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, labels=None):
        B, T = input_ids.shape
        if past_key_values is None:
            past_key_values = [None] * self.config.num_hidden_layers
        h = self.drop(self.embed(input_ids))
        ext_mask = None
        if attention_mask is not None:
            ext_mask = (1.0 - attention_mask[:,None,None,:].float()) * torch.finfo(h.dtype).min
        new_past = []
        for layer, pkv in zip(self.layers, past_key_values):
            h, ckv = layer(h, attention_mask=ext_mask, past_kv=pkv, use_cache=use_cache)
            new_past.append(ckv)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[...,:-1,:].contiguous().view(-1,self.config.vocab_size), labels[...,1:].contiguous().view(-1), ignore_index=-100)
        return {"loss": loss, "logits": logits, "past_key_values": new_past if use_cache else None}

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=256, temperature=0.7, top_p=0.95, top_k=50, repetition_penalty=1.1, eos_token_id=None, pad_token_id=None):
        self.eval()
        eos = eos_token_id or self.config.eos_token_id
        pad = pad_token_id or self.config.pad_token_id
        B = input_ids.shape[0]
        device = input_ids.device
        done = torch.zeros(B, dtype=torch.bool, device=device)
        past = None
        gen = input_ids.clone()
        for _ in range(max_new_tokens):
            inp = gen[:,-1:] if past is not None else gen
            out = self.forward(inp, past_key_values=past, use_cache=True)
            past = out["past_key_values"]
            logits = out["logits"][:,-1,:]
            if repetition_penalty != 1.0:
                for i in range(B):
                    for tid in set(gen[i].tolist()):
                        logits[i,tid] = logits[i,tid]/repetition_penalty if logits[i,tid]>0 else logits[i,tid]*repetition_penalty
            if temperature != 1.0: logits = logits / temperature
            if top_k > 0:
                kv = torch.topk(logits, min(top_k, logits.size(-1)))[0]
                logits[logits < kv[:,[-1]]] = float("-inf")
            if top_p < 1.0:
                sl, si = torch.sort(logits, descending=True)
                cp = torch.cumsum(F.softmax(sl,dim=-1),dim=-1)
                sl[cp - F.softmax(sl,dim=-1) > top_p] = float("-inf")
                logits.scatter_(1, si, sl)
            nxt = torch.multinomial(F.softmax(logits,dim=-1), 1)
            nxt[done] = pad
            gen = torch.cat([gen, nxt], dim=1)
            done = done | (nxt.squeeze(-1) == eos)
            if done.all(): break
        return gen

    def save_pretrained(self, d):
        os.makedirs(d, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(d,"model.pt"))
        self.config.save(os.path.join(d,"config.json"))
        print(f"Saved to {d}")

    @classmethod
    def from_pretrained(cls, d):
        cfg = CodeMindConfig.from_json(os.path.join(d,"config.json"))
        m = cls(cfg)
        m.load_state_dict(torch.load(os.path.join(d,"model.pt"), map_location="cpu"))
        print(f"Loaded from {d}")
        return m
