import torch
import torch.nn as nn
from torch.nn import functional as F

from .modules import Block 

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, n_embed: int, context_size: int, n_head: int, n_layer: int, dropout: float, device: str):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(context_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head, context_size=context_size, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.context_size = context_size
        self.device = device
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        token_embeddings = self.token_embedding_table(idx) # (Batch, Time Steps, Channels)
                                                 # T in our case is length of text (or context)
                                                 # C in our case is size of embeddings for token
        position_embeddings = self.position_embedding_table(torch.arange(T, device=self.device))
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            n_batch, n_timestep, n_channels = logits.shape
            logits = logits.view(n_batch * n_timestep, n_channels)
            targets = targets.view(n_batch * n_timestep)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # inputs idx which (B, T)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_size:]
            logits, loss = self(idx_cond) # (B * T, vocab_size) input: batch x context_length | output: logits_for_each_position x vocab_size
            logits = logits[:, -1, :] # (B, C) get last timestep 
            probs = F.softmax(logits, dim=-1) # (B, C) probabilities for characters
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) sample single prediction
            idx = torch.cat((idx, idx_next), dim=1) # (B, T + 1)
        return idx
