import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical

"""
reference: 
https://github.com/PKU-MARL/Multi-Agent-Transformer
"""


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):

    def __init__(self, embedding_dim, num_head, num_agent, masked=False):
        super(SelfAttention, self).__init__()

        assert embedding_dim % num_head == 0
        self.masked = masked
        self.n_head = num_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(embedding_dim, embedding_dim))
        self.query = init_(nn.Linear(embedding_dim, embedding_dim))
        self.value = init_(nn.Linear(embedding_dim, embedding_dim))
        # output projection
        self.proj = init_(nn.Linear(embedding_dim, embedding_dim))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(num_agent + 1, num_agent + 1))
                             .view(1, 1, num_agent + 1, num_agent + 1))

        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, embedding_dim, num_head, num_agent):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        # self.attn = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn = SelfAttention(embedding_dim, num_head, num_agent, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(embedding_dim, 1 * embedding_dim), activate=True),
            nn.GeLU(),  # TODO
            init_(nn.Linear(1 * embedding_dim, embedding_dim))
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, embedding_dim, num_head, num_agent):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ln3 = nn.LayerNorm(embedding_dim)
        self.attn1 = SelfAttention(embedding_dim, num_head, num_agent, masked=True)
        self.attn2 = SelfAttention(embedding_dim, num_head, num_agent, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(embedding_dim, 1 * embedding_dim), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * embedding_dim, embedding_dim))
        )

    def forward(self, x, rep_encoder):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_encoder + self.attn2(key=x, value=x, query=rep_encoder))
        x = self.ln3(x + self.mlp(x))
        return x

