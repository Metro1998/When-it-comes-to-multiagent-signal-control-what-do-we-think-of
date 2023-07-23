import copy
import math
import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.distributions import Categorical
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
        # B: batch_size, L: seq_len, D: embd_dim
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head evaluate_actions to be the batch dim
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
        # self.attn = MultiHeadAttention(n_embd, n_head, n_agent, masked=True)
        self.attn = MultiHeadAttention(embd_dim, head_num, agent_num, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(embd_dim, 1 * embd_dim), activate=True),
            nn.GELU(),
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
        self.attn1 = MultiHeadAttention(embd_dim, head_num, agent_num, masked=True)
        self.attn2 = MultiHeadAttention(embd_dim, head_num, agent_num, masked=True)
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

    def __init__(self, obs_dim, embd_dim, block_num, head_num, agent_num):
        super(Encoder, self).__init__()

        self.obs_dim = obs_dim
        self.embd_dim = embd_dim
        self.agent_num = agent_num

        self.GRU = GRUs(obs_dim, obs_dim, agent_num)  # we use GRU to encode the observation sequence

        self.obs_embedding = nn.Sequential(init_(nn.Linear(obs_dim, embd_dim), gain=5 / 3),
                                           nn.Tanh())  # TODO run statistics
        self.ln = nn.LayerNorm(embd_dim)  # todo
        self.blocks = nn.Sequential(*[EncodeBlock(embd_dim, head_num, agent_num) for _ in range(block_num)])

        # There are agent_num heads, because we approximate state value of each agent separately.
        # self.head = nn.Sequential(init_(nn.Linear(embd_dim, embd_dim), activate=True), nn.GELU(),
        #                           nn.LayerNorm(embd_dim),
        #                           init_(nn.Linear(embd_dim, 1)))
        self.head = nn.Sequential(init_(nn.Linear(embd_dim, embd_dim), gain=5 / 3),
                                  nn.Tanh(),
                                  init_(nn.Linear(embd_dim, 1)))
        # (B, N, embd_dim) -> (B, N, 1) output the approximating state value of each agent (token)

    def forward(self, obs):
        """

        :param obs: (B, N, seq_len, obs_dim)
        :return:
        """
        # print(obs.shape)
        # obs = self.GRU(obs)  # (B, N, embd_dim)
        obs = obs[:, :, -1, :]  # (B, N, obs_dim)
        obs_embeddings = self.obs_embedding(obs)
        rep = self.blocks(self.ln(obs_embeddings))
        values = self.head(rep).squeeze()

        return values, rep


class Decoder(nn.Module):

    def __init__(self, action_dim, embd_dim, block_num, head_num, agent_num, std_clip):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.embd_dim = embd_dim
        self.agent_num = agent_num

        log_std = torch.zeros(action_dim) - 1.5
        self.log_std = nn.Parameter(log_std)

        # action_dim + 2 = (action + 1) + 1, where action_dim + 1 means the one_hot encoding plus the start token,
        # and 1 indicates raw continuous parameter.
        self.action_embedding_dis = nn.Sequential(init_(nn.Linear(action_dim + 2, embd_dim, bias=False), gain=5 / 3),
                                                  nn.Tanh())
        self.action_embedding_con = nn.Sequential(init_(nn.Linear(action_dim + 2, embd_dim, bias=False), gain=5 / 3),
                                                  nn.Tanh())
        # self.ln = nn.LayerNorm(embd_dim)
        self.blocks_dis = nn.Sequential(*[DecodeBlock(embd_dim, head_num, agent_num) for _ in range(block_num)])
        self.blocks_con = nn.Sequential(*[DecodeBlock(embd_dim, head_num, agent_num) for _ in range(block_num)])

        self.head_dis = nn.Sequential(init_(nn.Linear(embd_dim, embd_dim), gain=5 / 3),
                                      nn.Tanh(),
                                      init_(nn.Linear(embd_dim, action_dim)),
                                      nn.Softmax(dim=-1))

        # self.head_con = nn.Sequential(init_(nn.Linear(embd_dim, embd_dim), activate=True), nn.GELU(),
        #                               nn.LayerNorm(embd_dim))
        # self.head_con = init_(nn.Linear(embd_dim, action_dim))
        # self.fc_std = init_(nn.Linear(embd_dim, action_dim))

        self.head_con = nn.Sequential(init_(nn.Linear(embd_dim, embd_dim), gain=5 / 3),
                                      nn.Tanh(),
                                      init_(nn.Linear(embd_dim, action_dim)))
        # self.fc_std = nn.Sequential(init_(nn.Linear(embd_dim, embd_dim), gain=5/3),
        #                             nn.Tanh(),
        #                             init_(nn.Linear(embd_dim, action_dim)))

        self.std_clip = std_clip

    def forward(self, hybrid_action, obs_rep):
        # x = self.ln(action_embeddings)

        x = self.action_embedding_dis(hybrid_action)
        for block in self.blocks_dis:
            x = block(x, obs_rep)
        logits = self.head_dis(x)  # (B, N, action_dim)

        y = self.action_embedding_con(hybrid_action)
        for block in self.blocks_con:
            y = block(y, obs_rep)
        means = self.head_con(y)  # (B, N, action_dim)

        B, N, _ = means.size()
        stds = torch.clamp(F.softplus(self.log_std), self.std_clip[0], self.std_clip[1]).repeat(B, N, 1)

        return logits, means, stds

    def autoregressive_act(self, obs_rep, act_dis_infer, act_con_infer, agent_to_update):
        hybrid_action = torch.zeros((obs_rep.shape[0], self.agent_num, self.action_dim + 2), dtype=torch.float32,
                                    device=torch.device('cpu'))
        hybrid_action[:, 0, 0] = 1
        for i in range(self.agent_num):

            # For agent_i in the batch, there is at least one to update
            if agent_to_update[:, i].sum() > 0:
                with torch.no_grad():
                    logits, means, stds = self.forward(hybrid_action, obs_rep)
                    logit = logits[:, i]
                    mean = means[:, i]
                    std = stds[:, i]

                    # print('logit', logit[0])
                    # print('mean', mean[0])
                    # print('std', std[0])

                    agent_to_update_ = agent_to_update[:, i].bool()

                    dist_dis = Categorical(logits=logit)  # Batch discrete distributions
                    act_dis_ = torch.where(agent_to_update_, dist_dis.sample(), act_dis_infer[:, i])

                    mean_ = torch.gather(mean, 1, act_dis_.unsqueeze(-1)).squeeze()
                    std_ = torch.gather(std, 1, act_dis_.unsqueeze(-1)).squeeze()
                    dist_con = Normal(mean_, std_)
                    act_con_raw = dist_con.sample()
                    act_con_ = torch.where(agent_to_update_, map2real(torch.tanh(act_con_raw)),
                                           act_con_infer[:, i].float())

                    act_logp_dis = torch.where(agent_to_update_, dist_dis.log_prob(act_dis_),
                                               torch.zeros_like(act_dis_, dtype=torch.float32,
                                                                device=torch.device('cpu')))
                    act_logp_con = torch.where(agent_to_update_, dist_con.log_prob(act_con_raw),
                                               torch.zeros_like(act_con_raw, dtype=torch.float32,
                                                                device=torch.device('cpu')))

                    # act_logp_dis = dist_dis.log_prob(act_dis_)
                    # act_logp_con = dist_con.log_prob(act_con_)

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
            output_act_con).numpy().astype(np.int32), torch.t(output_logp_con).numpy()

    def parallel_act(self, obs_rep, act_dis_exec, act_con_exec, act_dis_infer, act_con_infer, agent_to_update):

        hybrid_action = torch.zeros((obs_rep.shape[0], self.agent_num, self.action_dim + 2),
                                    device=torch.device('cuda'))
        action_dis = torch.where(agent_to_update.bool(), act_dis_exec, act_dis_infer)
        action_con = torch.where(agent_to_update.bool(), act_con_exec, act_con_infer)
        hybrid_action[:, 0, 0] = 1
        hybrid_action[:, 1:, 1:-1].copy_(F.one_hot(action_dis, num_classes=self.action_dim)[:, :-1, :])
        hybrid_action[:, 1:, -1] = action_con[:, :-1]
        logits, means, stds = self.forward(hybrid_action, obs_rep)

        dist_dis = Categorical(logits=logits)
        act_dis_ = torch.where(agent_to_update.bool(), act_dis_exec, act_dis_infer)
        act_logp_dis = dist_dis.log_prob(act_dis_)[agent_to_update == 1]
        entropy_dis = dist_dis.entropy()[agent_to_update == 1]  # todo fix

        means = means.gather(-1, act_dis_.unsqueeze(-1)).squeeze()
        stds = stds.gather(-1, act_dis_.unsqueeze(-1)).squeeze()
        dist_con = Normal(means, stds)
        act_con_ = torch.where(agent_to_update.bool(), remap(act_con_exec),
                               torch.zeros_like(act_con_infer, dtype=torch.float32, device=torch.device('cuda')))
        act_logp_con = dist_con.log_prob(act_con_)[agent_to_update == 1]
        entropy_con = dist_con.entropy()[agent_to_update == 1]

        return act_logp_dis, entropy_dis, act_logp_con, entropy_con


class MultiAgentTransformer(nn.Module):

    def __init__(self, obs_dim, action_dim, embd_dim, agent_num, block_num, head_num, std_clip, device):
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
        self.encoder = Encoder(obs_dim, embd_dim, block_num, head_num, agent_num)
        self.decoder = Decoder(action_dim, embd_dim, block_num, head_num, agent_num, std_clip)

        self.to(self.device)

    def evaluate_actions(self, obs, act_dis, act_con, act_dis_infer, act_con_infer, agent_to_update):
        """
        Get action logprobs / entropy for actor update.

        :param obs: (torch.Tensor) (batch_size, agent_num, obs_dim)  they will be preprocessed in the buffer, specifically from numpy to tensor.
        :param act_dis: (torch.Tensor) (batch_size, agent_num)
        :param act_con: (torch.Tensor) (batch_size, agent_num, action_dim)
        :return:
        """
        _, obs_rep = self.encoder(obs)

        act_log_dis, entropy_dis, act_log_con, entropy_con = self.decoder.parallel_act(obs_rep, act_dis, act_con,
                                                                                       act_dis_infer, act_con_infer,
                                                                                       agent_to_update)

        return act_log_dis, entropy_dis, act_log_con, entropy_con

    def get_values(self, obs):
        """
        Get value function predictions.
        :param obs: (np.ndarray) (batch_size, agent_num, obs_dim)
        :return: (torch.Tensor) value function predictions
        """
        obs = check(obs).to(self.device)
        values, _ = self.encoder(obs)

        return values

    def act(self, obs, act_dis_infer, act_con_infer, agent_to_update):
        """
        Compute stages and value function predictions for the given inputs.

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
            values, obs_rep = self.encoder(obs)

        act_dis, logp_dis, act_con, logp_con = self.decoder.autoregressive_act(obs_rep, act_dis_infer, act_con_infer,
                                                                               agent_to_update)

        return act_dis, logp_dis, act_con, logp_con, values
