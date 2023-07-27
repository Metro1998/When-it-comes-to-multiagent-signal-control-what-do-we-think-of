import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torchviz import make_dot

from torch.nn import functional as F
from torch.distributions import Categorical, Normal
from utils.util import *

"""
reference: 
https://github.com/PKU-MARL/Multi-Agent-Transformer
"""


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


def init_aux_(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        nn.init.zeros_(m.bias)


class GRUs(nn.Module):

    def __init__(self, input_dim, hidden_dim, agent_num):
        super(GRUs, self).__init__()
        self.grus = nn.ModuleList([nn.GRU(input_dim, hidden_dim, batch_first=True) for _ in range(agent_num)])

    def forward(self, x):
        """

        :param x: x.shape = (batch_size, agent_num, seq_len, input_dim)
        :return: (batch_size, agent_num, hidden_dim)
        """

        ms = []
        for i in range(x.shape[1]):
            m = self.grus[i](x[:, i, :, :])[0]
            ms.append(m[:, -1, :])

        return torch.stack(ms, dim=1)


class MultiHeadAttention(nn.Module):

    def __init__(self, embd_dim, head_num, agent_num, masked=False):
        super(MultiHeadAttention, self).__init__()

        assert embd_dim % head_num == 0
        self.masked = masked
        self.head_num = head_num

        # key, query, value projections for all heads
        self.key_proj = init_(nn.Linear(embd_dim, embd_dim))
        self.query_proj = init_(nn.Linear(embd_dim, embd_dim))
        self.value_proj = init_(nn.Linear(embd_dim, embd_dim))
        # output projection
        self.out_proj = init_(nn.Linear(embd_dim, embd_dim))

        # if self.masked, causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(agent_num + 1, agent_num + 1))
                             .view(1, 1, agent_num + 1, agent_num + 1))

    def forward(self, key, value, query):
        # B: batch_size, L: seq_len, D: embd_dim
        B, L, D = query.size()

        # calculate query, key, value for all heads in batch and move head evaluate_actions to be the batch dim
        k = self.key_proj(key).view(B, L, self.head_num, D // self.head_num).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query_proj(query).view(B, L, self.head_num, D // self.head_num).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value_proj(value).view(B, L, self.head_num, D // self.head_num).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.out_proj(y)
        return y


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, embd_dim, head_num, agent_num, dropout=0.1):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(embd_dim)
        self.ln2 = nn.LayerNorm(embd_dim)
        # self.attn = MultiHeadAttention(n_embd, n_head, n_agent, masked=True)
        self.attn = MultiHeadAttention(embd_dim, head_num, agent_num, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(embd_dim, 2 * embd_dim), activate=True),
            nn.GELU(),
            init_(nn.Linear(2 * embd_dim, embd_dim))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention part
        attn_out = self.attn(x, x, x)
        x = self.ln1(x + self.dropout(attn_out))

        # MLP part
        linear_out = self.mlp(x)
        x = self.ln2(x + self.dropout(linear_out))
        return x


class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, embd_dim, head_num, agent_num, dropout=0.1):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(embd_dim)
        self.ln2 = nn.LayerNorm(embd_dim)
        self.ln3 = nn.LayerNorm(embd_dim)
        self.attn1 = MultiHeadAttention(embd_dim, head_num, agent_num, masked=True)
        self.attn2 = MultiHeadAttention(embd_dim, head_num, agent_num, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(embd_dim, 1 * embd_dim), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * embd_dim, embd_dim))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rep_encoder):
        x = self.ln1(x + self.dropout(self.attn1(x, x, x)))
        x = self.ln2(rep_encoder + self.dropout(self.attn2(key=x, value=x, query=rep_encoder)))
        x = self.ln3(x + self.dropout(self.mlp(x)))
        return x


class Encoder(nn.Module):

    def __init__(self, obs_dim, embd_dim, block_num, head_num, agent_num, dropout=0.1):
        """
        Encoder (critic) to approximate the value function
        :param obs_dim:
        :param embd_dim:
        :param block_num:
        :param head_num:
        :param agent_num:
        :param dropout:
        """
        super(Encoder, self).__init__()

        self.obs_dim = obs_dim
        self.embd_dim = embd_dim
        self.agent_num = agent_num

        self.GRU = GRUs(obs_dim, embd_dim, agent_num)  # we use GRU to encode the observation sequence

        self.obs_encoder = nn.Sequential(init_(nn.Linear(obs_dim, embd_dim), activate=True),
                                         nn.GELU())
        self.ln = nn.LayerNorm(embd_dim)
        self.blocks = nn.Sequential(*[EncodeBlock(embd_dim, head_num, agent_num, dropout) for _ in range(block_num)])
        self.head = nn.Sequential(init_(nn.Linear(embd_dim, embd_dim), activate=True),
                                  nn.GELU(),
                                  nn.LayerNorm(embd_dim),
                                  init_(nn.Linear(embd_dim, 1)))
        # (B, N, embd_dim) -> (B, N, 1) output the approximating state value_proj of each agent (token)

    def forward(self, obs):
        """

        :param obs: (B, N, seq_len, obs_dim)
        :return:
        """
        obs = obs[:, :, -1, :]  # (B, N, obs_dim)

        obs_embeddings = self.obs_encoder(obs)
        rep = self.blocks(self.ln(obs_embeddings))
        values = self.head(rep).squeeze()

        return values, rep


class Encoder_Decoder_dis(nn.Module):

    def __init__(self, action_dim, obs_dim, embd_dim, block_num, head_num, agent_num, dropout=0.1):
        """
        Like H-PPO we have separated the perception part of the critic and the actor,
        which means the actor has its own encoder.
        :param action_dim:
        :param obs_dim:
        :param embd_dim:
        :param block_num:
        :param head_num:
        :param agent_num:
        :param dropout:
        """
        super(Encoder_Decoder_dis, self).__init__()

        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.embd_dim = embd_dim
        self.agent_num = agent_num

        self.GRU = GRUs(obs_dim, embd_dim, agent_num)  # we use GRU to encode the observation sequence

        self.obs_embedding = nn.Sequential(init_(nn.Linear(obs_dim, embd_dim), activate=True),
                                           nn.GELU())
        self.ln1 = nn.LayerNorm(embd_dim)
        self.blocks_enc = nn.Sequential(*[EncodeBlock(embd_dim, head_num, agent_num, dropout) for _ in range(block_num)])

        self.action_embedding = nn.Sequential(init_(nn.Linear(action_dim + 2, embd_dim, bias=False), activate=True),
                                              nn.GELU())
        self.ln2 = nn.LayerNorm(embd_dim)
        self.blocks_dec = nn.Sequential(*[DecodeBlock(embd_dim, head_num, agent_num, dropout) for _ in range(block_num)])
        self.head = nn.Sequential(init_(nn.Linear(embd_dim, embd_dim), activate=True),
                                  nn.GELU(),
                                  nn.LayerNorm(embd_dim),
                                  init_(nn.Linear(embd_dim, action_dim)),
                                  nn.Softmax(dim=-1))

    def forward(self, obs, hybrid_action):

        obs = obs[:, :, -1, :]  # (B, N, obs_dim) gru is temporally deprecated
        obs_embeddings = self.obs_embedding(obs)
        obs_rep = self.blocks_enc(self.ln1(obs_embeddings))

        action_embeddings = self.action_embedding(hybrid_action)
        x = self.ln2(action_embeddings)
        for block in self.blocks_dec:
            x = block(x, obs_rep)
        logits = self.head(x)  # (B, N, action_dim)

        return logits


class Encoder_Decoder_con(nn.Module):

    def __init__(self, action_dim, obs_dim, embd_dim, block_num, head_num, agent_num, std_clips, dropout=0.1):
        """

        :param action_dim:
        :param obs_dim:
        :param embd_dim:
        :param block_num:
        :param head_num:
        :param agent_num:
        :param std_clips: the clipping range of std
        """
        super(Encoder_Decoder_con, self).__init__()

        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.embd_dim = embd_dim
        self.agent_num = agent_num
        self.std_clips = std_clips

        log_std = torch.zeros(action_dim) - 1.5
        self.log_std = nn.Parameter(log_std, requires_grad=True)

        self.obs_embedding = nn.Sequential(init_(nn.Linear(obs_dim, embd_dim), activate=True),
                                           nn.GELU())
        self.ln1 = nn.LayerNorm(embd_dim)
        self.blocks_enc = nn.Sequential(*[EncodeBlock(embd_dim, head_num, agent_num, dropout) for _ in range(block_num)])

        self.action_embedding = nn.Sequential(init_(nn.Linear(action_dim + 2, embd_dim, bias=False), activate=True),
                                              nn.GELU())
        self.ln2 = nn.LayerNorm(embd_dim)
        self.blocks_dec = nn.Sequential(*[DecodeBlock(embd_dim, head_num, agent_num, dropout) for _ in range(block_num)])
        self.head = nn.Sequential(init_(nn.Linear(embd_dim, embd_dim), activate=True),
                                  nn.GELU(),
                                  nn.LayerNorm(embd_dim),
                                  init_(nn.Linear(embd_dim, action_dim)))

    def forward(self, obs, hybrid_action):

        obs = obs[:, :, -1, :]  # (B, N, obs_dim) gru is temporally deprecated
        obs_embeddings = self.obs_embedding(obs)
        obs_rep = self.blocks_enc(self.ln1(obs_embeddings))

        action_embeddings = self.action_embedding(hybrid_action)
        x = self.ln2(action_embeddings)
        for block in self.blocks_dec:
            x = block(x, obs_rep)
        means = self.head(x)  # (B, N, action_dim)
        stds = torch.clamp(F.softplus(self.log_std), self.std_clips[0], self.std_clips[1]).repeat(means.shape[0], means.shape[1], 1)

        return means, stds


class MultiAgentTransformer(nn.Module):

    def __init__(self, obs_dim, action_dim, embd_dim, agent_num, block_num, head_num, std_clip, dropout, device):
        """

        :param obs_dim:
        :param action_dim:
        :param embd_dim:
        :param agent_num:
        :param block_num:
        :param head_num:
        :param std_clip:
        :param device:
        """
        super(MultiAgentTransformer, self).__init__()

        self.action_dim = action_dim
        self.agent_num = agent_num
        self.std_clip = std_clip
        self.device = device

        # In our original implementation of HPPO, the discrete and continuous actors are thought to be independent with
        # each other, so are they in MAT.
        self.encoder = Encoder(obs_dim, embd_dim, block_num, head_num, agent_num, dropout)
        self.decoder_con = Encoder_Decoder_con(action_dim, obs_dim, embd_dim, block_num, head_num, agent_num, std_clip, dropout)
        self.decoder_dis = Encoder_Decoder_dis(action_dim, obs_dim, embd_dim, block_num, head_num, agent_num, dropout)
        self.mapping = mapping(min_green=10, max_green=30)

        self.to(self.device)

    def evaluate_actions(self, obs, act_dis, act_con, agent_to_update, target_decoder_dis=None, target_decoder_con=None):
        """
        Get action logprobs / entropy for actor update.

        :param obs: (torch.Tensor) (batch_size, agent_num, obs_dim)  they will be preprocessed in the buffer, specifically from numpy to tensor.
        :param act_dis: (torch.Tensor) (batch_size, agent_num)
        :param act_con: (torch.Tensor) (batch_size, agent_num, action_dim)
        :param target_decoder_dis:
        :param target_decoder_con:
        :param agent_to_update:
        :return:
        """
        act_log_dis, entropy_dis, act_log_con, entropy_con = self.parallel_act(obs, act_dis, act_con, agent_to_update, target_decoder_dis, target_decoder_con)

        return act_log_dis, entropy_dis, act_log_con, entropy_con

    def get_values(self, obs):
        """
        Get value_proj function predictions.
        :param obs: (np.ndarray) (batch_size, agent_num, obs_dim)
        :return: (torch.Tensor) value_proj function predictions
        """
        obs = check(obs).to(self.device)
        values, _ = self.encoder(obs)

        return values

    def act(self, obs, act_dis_infer, act_con_infer, agent_to_update):
        """
        Compute stages and value_proj function predictions for the given inputs.

        :param obs:
        :param act_dis_infer: the last step's discrete actions, for masking the discrete action space if it time to act.
        :param act_con_infer: the last step's continuous actions, for deciding whether to infer(map2real(act_con_infer) < 0.1) on the current step.
        :param agent_to_update:
        :return:
        """
        obs = check(obs).to(self.device)
        act_dis_infer = check(act_dis_infer).to(self.device)
        act_con_infer = check(act_con_infer).to(self.device)
        agent_to_update = check(agent_to_update).to(self.device)

        with torch.no_grad():
            values, _ = self.encoder(obs)
            act_dis, logp_dis, act_con, logp_con = self.autoregressive_act(obs, act_dis_infer, act_con_infer, agent_to_update)

        return act_dis, logp_dis, act_con, logp_con, values

    def autoregressive_act(self, obs, act_dis_infer, act_con_infer, agent_to_update):
        hybrid_action = torch.zeros((obs.shape[0], self.agent_num, self.action_dim + 2), dtype=torch.float32,
                                    device=torch.device('cpu'))
        hybrid_action[:, 0, 0] = 1
        for i in range(self.agent_num):

            # For agent_i in the batch, there is at least one to update
            if agent_to_update[:, i].sum() > 0:
                with torch.no_grad():
                    logits = self.decoder_dis.forward(obs, hybrid_action)
                    means, stds = self.decoder_con.forward(obs, hybrid_action)
                    logit = logits[:, i]
                    mean = means[:, i]
                    std = stds[:, i]

                    agent_to_update_ = agent_to_update[:, i].bool()

                    dist_dis = Categorical(logits=logit)  # Batch discrete distributions
                    act_dis_ = torch.where(agent_to_update_, dist_dis.sample(), act_dis_infer[:, i])

                    mean_ = torch.gather(mean, 1, act_dis_.unsqueeze(-1)).squeeze()
                    std_ = torch.gather(std, 1, act_dis_.unsqueeze(-1)).squeeze()
                    dist_con = Normal(mean_, std_)
                    act_con_raw = dist_con.sample()
                    act_con_ = torch.where(agent_to_update_, self.mapping.norm(self.mapping.map2real(torch.tanh(act_con_raw))),
                                           act_con_infer[:, i].float())

                    act_logp_dis = torch.where(agent_to_update_, dist_dis.log_prob(act_dis_), torch.zeros_like(act_dis_,
                                               dtype=torch.float32, device=torch.device('cpu')))
                    act_logp_con = torch.where(agent_to_update_, dist_con.log_prob(act_con_raw), torch.zeros_like(act_con_raw,
                                               dtype=torch.float32, device=torch.device('cpu')))

            # For agent_i in the batch, there is no one need to update
            else:
                act_dis_ = act_dis_infer[:, i]
                act_con_ = act_con_infer[:, i].float()
                # Padding
                act_logp_dis = torch.zeros_like(act_dis_, dtype=torch.float32, device=torch.device('cpu'))
                act_logp_con = torch.zeros_like(act_con_, dtype=torch.float32, device=torch.device('cpu'))

            if i + 1 < self.agent_num:
                hybrid_action[:, i + 1, 1:-1].copy_(F.one_hot(act_dis_, num_classes=self.action_dim).float())
                hybrid_action[:, i + 1, -1] = act_con_

            if i == 0:
                output_act_dis = act_dis_.unsqueeze(0)
                output_act_con = act_con_.unsqueeze(0)
                output_logp_dis = act_logp_dis.unsqueeze(0)
                output_logp_con = act_logp_con.unsqueeze(0)
            else:
                output_act_dis = torch.cat((output_act_dis, act_dis_.unsqueeze(0)), dim=0)
                output_act_con = torch.cat((output_act_con, act_con_.unsqueeze(0)), dim=0)
                output_logp_dis = torch.cat((output_logp_dis, act_logp_dis.unsqueeze(0)), dim=0)
                output_logp_con = torch.cat((output_logp_con, act_logp_con.unsqueeze(0)), dim=0)

        return torch.t(output_act_dis).numpy(), torch.t(output_logp_dis).numpy(), torch.t(
            output_act_con).numpy(), torch.t(output_logp_con).numpy()

    def parallel_act(self, obs, act_dis_exec, act_con_exec, agent_to_update, target_decoder_dis=None, target_decoder_con=None):
        hybrid_action = torch.zeros((obs.shape[0], self.agent_num, self.action_dim + 2),
                                    device=torch.device('cuda'))
        hybrid_action[:, 0, 0] = 1
        hybrid_action[:, 1:, 1:-1].copy_(F.one_hot(act_dis_exec, num_classes=self.action_dim)[:, :-1, :])
        hybrid_action[:, 1:, -1] = act_con_exec[:, :-1]
        if target_decoder_dis is None:
            logits = self.decoder_dis.forward(obs, hybrid_action)
        else:
            logits = target_decoder_dis.forward(obs, hybrid_action)
        if target_decoder_con is None:
            means, stds = self.decoder_con.forward(obs, hybrid_action)
        else:
            means, stds = target_decoder_con.forward(obs, hybrid_action)

        dist_dis = Categorical(logits=logits)
        act_logp_dis = dist_dis.log_prob(act_dis_exec)[agent_to_update == 1]
        entropy_dis = dist_dis.entropy()[agent_to_update == 1]  # todo fix

        means = means.gather(-1, act_dis_exec.unsqueeze(-1)).squeeze()
        stds = stds.gather(-1, act_dis_exec.unsqueeze(-1)).squeeze()
        dist_con = Normal(means, stds)
        act_con_ = torch.where(agent_to_update.bool(), self.mapping.remap(self.mapping.anorm(act_con_exec)),
                               torch.zeros_like(act_con_exec, dtype=torch.float32, device=torch.device('cuda')))
        act_logp_con = dist_con.log_prob(act_con_)[agent_to_update == 1]
        entropy_con = dist_con.entropy()[agent_to_update == 1]

        return act_logp_dis, entropy_dis, act_logp_con, entropy_con
