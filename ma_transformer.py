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

    def __init__(self, embd_dim, head_num, agent_num, masked=False):
        super(SelfAttention, self).__init__()

        assert embd_dim % head_num == 0
        self.masked = masked
        self.head_num = head_num
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(embd_dim, embd_dim))
        self.query = init_(nn.Linear(embd_dim, embd_dim))
        self.value = init_(nn.Linear(embd_dim, embd_dim))
        # output projection
        self.proj = init_(nn.Linear(embd_dim, embd_dim))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(agent_num + 1, agent_num + 1))
                             .view(1, 1, agent_num + 1, agent_num + 1))

        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()
        # B: batch_size, L: sequence_length, D: embd_dim

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.head_num, D // self.head_num).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.head_num, D // self.head_num).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.head_num, D // self.head_num).transpose(1, 2)  # (B, nh, L, hs)

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

    def __init__(self, embd_dim, head_num, agent_num):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(embd_dim)
        self.ln2 = nn.LayerNorm(embd_dim)
        # self.attn = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn = SelfAttention(embd_dim, head_num, agent_num, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(embd_dim, 1 * embd_dim), activate=True),
            nn.GELU(),  # TODO
            init_(nn.Linear(1 * embd_dim, embd_dim))
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, embd_dim, head_num, agent_num):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(embd_dim)
        self.ln2 = nn.LayerNorm(embd_dim)
        self.ln3 = nn.LayerNorm(embd_dim)
        self.attn1 = SelfAttention(embd_dim, head_num, agent_num, masked=True)
        self.attn2 = SelfAttention(embd_dim, head_num, agent_num, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(embd_dim, 1 * embd_dim), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * embd_dim, embd_dim))
        )

    def forward(self, x, rep_encoder):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_encoder + self.attn2(key=x, value=x, query=rep_encoder))
        x = self.ln3(x + self.mlp(x))
        return x


class Encoder(nn.Module):

    def __init__(self, obs_dim, block_num, embd_dim, head_num, agent_num):
        super(Encoder, self).__init__()

        self.obs_dim = obs_dim
        self.embd_dim = embd_dim
        self.agent_num = agent_num

        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, embd_dim), activate=True), nn.GELU())
        self.ln = nn.LayerNorm(embd_dim)
        self.blocks = nn.Sequential(*[EncodeBlock(embd_dim, head_num, agent_num) for _ in range(block_num)])
        self.head = nn.Sequential(init_(nn.Linear(obs_dim, embd_dim), activate=True), nn.GELU(), nn.LayerNorm(embd_dim),
                                  init_(nn.Linear(obs_dim, 1)))

    def forward(self, obs):
        obs_embeddings = self.obs_encoder(obs)
        rep = self.blocks(self.ln(obs_embeddings))
        v_glob = self.head(rep)

        return v_glob, rep


class Decoder(nn.Module):

    def __init__(self, obs_dim, action_dim, embd_dim, block_num, head_num, agent_num, init_log_std, action_type):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.embd_dim = embd_dim
        self.action_type = action_type

        if action_type != 'Discrete':
            self.log_std = nn.Parameter(torch.zeros(action_dim, ) + init_log_std)

        if action_type == 'Discrete':
            self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim + 1, embd_dim, bias=False), activate=True), nn.GELU())
        else:
            self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, embd_dim), activate=True), nn.GELU())
        self.ln = nn.LayerNorm(embd_dim)
        self.blocks = nn.Sequential(*[DecodeBlock(embd_dim, head_num, agent_num) for _ in range(block_num)])
        self.head = nn.Sequential(init_(nn.Linear(embd_dim, embd_dim), activate=True), nn.GELU(),
                                  nn.LayerNorm(embd_dim),
                                  init_(nn.Linear(embd_dim, action_dim)))

    def forward(self, act, obs_rep):
        action_embeddings = self.action_encoder(act)
        x = self.ln(action_embeddings)
        for block in self.blocks:
            x = block(x, obs_rep)
        logit = self.head(x)

        return logit
