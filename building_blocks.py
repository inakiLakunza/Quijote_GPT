
import torch
import torch.nn as nn
from torch.nn import functional as F





class FeedForward(nn.Module):
    """
    A simple linear layer followed by a non-linearity
    (this is done in a PER-TOKEN LEVEL, the tokens do this independently,
    so the self-attention is the communication, and then once they
    have gathered the data, they need to think of that data INDIVIDUALLY)
    """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # project up and project down, explained in the paper,
            # in the position-wise Feed-Forward Networks section
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.next(x)

class Head(nn.Module):
    """
    One head of self-attention
    """

    def __init__(self, block_size, n_embd, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # If you have parameters in your model, which should be saved and restored in the state_dict,
        # but not trained by the optimizer, you should register them as buffers.
        # Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, C)
        q = self.query(x) # (B, T, C)

        # compute attention scores, "affinities"
        #===================================
        normalize_C = C**-0.5
        wei = q @ k.transpose(-2, -1) * normalize_C # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))         # (B, T, T) 
        wei = F.softmax(wei, dim=-1)                                         # (B, T, T)
        #===================================

        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out
    
# Multi-head attention is just using multiple attention heads
# in parallel and then concatenating the resutls
class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel
    """

    def __init__(self, num_heads, block_size, n_embd, head_size):
        super().__init__()
        
        # ModuleList can be indexed like a regular Python list,
        # but modules it contains are properly registered, and will
        # be visible by all Module methods.
        self.heads = nn.ModuleList([Head(block_size, n_embd, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        # concatenate outputs of the heads along the last dimension (C)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class Block(nn.Module):
    """
    Transformer block: communication follwed by computation
    """

    def __init__(self, n_embd, block_size, n_heads):
        # n_embd:  embedding dimension
        # n_heads: the number of heads we want
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, block_size, n_embd, head_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        # Using residual connections
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x


