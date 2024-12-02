import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    
    def __init__(self, head_size: int, n_embed: int, context_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        return out 


class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads: int, head_size: int, n_embed: int, context_size: int, dropout:float):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size, n_embed=n_embed, context_size=context_size, dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class Block(nn.Module):
        
    def __init__(self, n_embed, n_head, context_size, dropout):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embed=n_embed, context_size=context_size, dropout=dropout)
        self.ffwd = FeedForward(n_embed=n_embed, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class FeedForward(nn.Module):
    
    def __init__(self, n_embed: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)