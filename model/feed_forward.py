import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        actual = (int(intermediate_size * 2/3) + 63) // 64 * 64
        self.gate = nn.Linear(hidden_size, actual, bias=False)
        self.up = nn.Linear(hidden_size, actual, bias=False)
        self.down = nn.Linear(actual, hidden_size, bias=False)
        self.drop = nn.Dropout(dropout)
        for m in [self.gate, self.up, self.down]:
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))
