"""
Include basic neural network, and specific implementation of HPPO
Author:Metro
date:2022.12.13
"""
import os

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical
from Utils.util import *


class Actor(nn.Module):
    def __init__(self,
                 obs_space,
                 obs_signal_space,
                 hidden_size,
                 action_space,
                 nonlinear,
                 init_log_std
                 ):
        """
        Distributed execution, that is, the agent_i output its hybrid action based on its own observation
        without other agents' policies and global observation representation.

        :param obs_space: the size of observation
        :param obs_signal_space: the size of signal observation
        :param hidden_size: the size of hidden layers [hidden_size[0], hidden_size[1], hidden_size[2]]
        :param action_space: the size of output (discrete) dimension
        :param nonlinear: the nonlinear activation
        """
        super().__init__()

        self.nonlinear = nn.ReLU() if nonlinear == 'relu' else nn.Tanh()
        self.log_std = nn.Parameter(torch.zeros(action_space, ) + init_log_std, requires_grad=True)

        # actor_con
        self.actor_con = nn.Sequential(
            nn.Linear(obs_space, hidden_size[0]),
            self.nonlinear,
            nn.Linear(hidden_size[0], hidden_size[1]),
            self.nonlinear,
            nn.Linear(hidden_size[1], hidden_size[2]),
            self.nonlinear,
            nn.Linear(hidden_size[2], action_space)
        )

        # actor_dis
        self.rnn = nn.LSTM(input_size=obs_signal_space, hidden_size=hidden_size[0] // 2)
        self.linear = nn.Linear(obs_space, hidden_size[0] // 2)
        self.actor_dis = nn.Sequential(
            nn.Linear(hidden_size[0], hidden_size[1]),
            self.nonlinear,
            nn.Linear(hidden_size[1], hidden_size[2]),
            self.nonlinear,
            nn.Linear(hidden_size[2], action_space),
            nn.Softmax(dim=-1)
        )

        # orthogonal initialization
        nn.init.orthogonal_(tensor=self.linear.weight, gain=np.sqrt(2))
        nn.init.zeros_(tensor=self.linear.bias)
        for m in self.actor_dis.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(tensor=m.weight, gain=np.sqrt(2))
                nn.init.zeros_(tensor=m.bias)
        for m in self.actor_con.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(tensor=m.weight, gain=np.sqrt(2))
                nn.init.zeros_(tensor=m.bias)
        # Also, Andrychowicz, et al.(2021) find centering the action distribution around 0 (i.e., initialize the policy
        # output layer weights with 0.01”) to be beneficial
        nn.init.orthogonal_(tensor=self.actor_dis[-2].weight, gain=0.01)
        nn.init.orthogonal_(tensor=self.actor_con[-1].weight, gain=0.01)

    def dis_forward(self, obs, obs_signal):
        """

        :param obs: the ordinary information (observation) of the agent (and its neighbors)
        :param obs_signal: the information of signal sequence  (sequence_length, state_space_signal)
        :return:
        """

        out_put, (hn, cn) = self.rnn(obs_signal)
        h = self.linear(obs)
        x = self.actor_dis(torch.cat((hn.squeeze(), h), dim=-1))

        return x

    def get_actions(self, obs, obs_signal):
        """

        :param obs: the ordinary information (observation) of the agent (and its neighbors)
        :param obs_signal: the information of signal sequence  (sequence_length, state_space_signal)
        :return:
        """

        action_probs = self.dis_forward(obs, obs_signal)
        dist_dis = Categorical(action_probs)
        action_dis = dist_dis.sample()
        logprob_dis = dist_dis.log_prob(action_dis)

        mean = self.actor_con(obs)
        std = torch.clamp(F.softplus(self.log_std), min=0.01, max=0.5)
        dist_con = Normal(mean, std)
        action_con = dist_con.sample()
        logprob_con = dist_con.log_prob(action_con)

        return action_dis, logprob_dis, action_con, logprob_con

    def evaluate_actions(self, obs, obs_signal, action_dis, action_con):
        """
        Compute log probability and entropy of given actions.
        :param obs: the ordinary information (observation) of the agent (and its neighbors)
        :param obs_signal: the information of signal sequence  (sequence_length, state_space_signal)
        :param action_dis: the discrete action
        :param action_con: the continuous action
        :return:
        """

        action_probs = self.dis_forward(obs, obs_signal)
        dist_dis = Categorical(action_probs)
        logprob_dis = dist_dis.log_prob(action_dis)
        dist_entropy_dis = dist_dis.entropy()

        mean = self.actor_con(obs)
        std = torch.clamp(F.softplus(self.log_std), min=0.01, max=0.5)
        dist_con = Normal(mean, std)
        logprob_con = dist_con.log_prob(action_con)
        dist_entropy_con = dist_con.entropy()

        return logprob_dis, dist_entropy_dis, logprob_con, dist_entropy_con


class Critic(nn.Module):
    def __init__(self,
                 cent_obs,
                 hidden_size,
                 nonlinear):
        """
        One benefit of applying Equation (9) is that agents only need to maintain a joint advantage estimator Aπ(s, a)
        rather than one centralised critic for each individual agent (e.g., unlike CTDE methods such as MADDPG).

        :param cent_obs: the size of global observation
        :param hidden_size: the size of hidden layers [hidden_size[0], hidden_size[1], hidden_size[2]]
        :param nonlinear: the nonlinear activation
        """
        super().__init__()

        self.nonlinear = nn.ReLU() if nonlinear == 'relu' else nn.Tanh()

        self.critic = nn.Sequential(
            nn.Linear(cent_obs, hidden_size[0]),
            nn.Tanh(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.Tanh(),
            nn.Linear(hidden_size[1], hidden_size[2]),
            nn.Tanh(),
            nn.Linear(hidden_size[2], 1)
        )

        # orthogonal initialization
        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(tensor=m.weight, gain=np.sqrt(2))
                nn.init.zeros_(tensor=m.bias)

    def get_values(self, cent_obs):
        """

        :param cent_obs: the size of global observation
        :return:
        """

        values = self.critic(cent_obs)

        return values


class HAPPO:
    def __init__(self, buffer, actors, critic, num_agents, rollout_buffer, epochs, mini_batch, clip_ratio, gamma, lam,
                 max_norm, coeff_entropy, random_seed, lr_actor_con, lr_actor_dis, lr_std, lr_critic, lr_decay_rate,
                 target_kl_dis, target_kl_con, init_log_std, minus_inf, obs_dim, sequence_dim, act_dim, device):

        self.buffer = rollout_buffer
        self.actors = [a.to(device) for a in actors]
        self.actors_old = [a.to(device) for a in actors]
        self.critic = critic.to(device)
        self.random_seed = random_seed
        self.mini_batch = mini_batch
        self.epochs = epochs
        self.gamma = gamma
        self.lam = lam
        self.minus_inf = minus_inf
        self.obs_dim = obs_dim
        self.sequence_dim = sequence_dim
        self.act_dim = act_dim

        # to offer a random permutation
        self.permutation = np.arange(num_agents)
        np.random.shuffle(self.permutation)

        self.optimizer_actor_con = [torch.optim.Adam([
            {'params': a.actor_con.parameters(), 'lr': lr_actor_con},
            {'params': a.log_std, 'lr': lr_std}
        ]) for a in self.actors]
        self.optimizer_actor_dis = [torch.optim.Adam([
            {'params': a.actor_dis.parameters(), 'lr': lr_actor_dis}
        ]) for a in self.actors]
        self.optimizer_critic = torch.optim.Adam(critic.parameters(), lr=lr_critic)

        self.lr_scheduler_actor_con = [torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=lr_decay_rate)
                                       for optim in self.optimizer_actor_con]
        self.lr_scheduler_actor_dis = [torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=lr_decay_rate)
                                       for optim in self.optimizer_actor_dis]
        self.lr_scheduler_critic = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_critic, lr_decay_rate)

        self.target_kl_dis = target_kl_dis
        self.target_kl_con = target_kl_con

    def set_random_seeds(self):
        """
        Sets all possible random seeds to results can be reproduces.
        :return:
        """
        os.environ['PYTHONHASHSEED'] = str(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
            torch.cuda.manual_seed(self.random_seed)

    def update(self):

        cent_obs_buf, obs_agent_buf, obs_sequence_agent_buf, act_dis_buf, act_con_buf, logp_dis_buf, logp_con_buf, rew_buf, rew_agent_buf, flag_buf = self.buffer.get()

        observation_agent, observation_sequence_agent, action_dis, action_con, old_logp_dis, old_logp_con = \
            self.preprocess(obs_agent_buf, obs_sequence_agent_buf, act_con_buf, act_dis_buf, logp_dis_buf, logp_con_buf)
        for i in self.epochs:
            # Recompute values at the beginning of each epoch
            advantage, reward_to_go = self.recompute(cent_obs_buf, rew_agent_buf, flag_buf)
            for j in self.permutation:
                # mini_batch

    def recompute(self, cent_obs_buf, rew_agent_buf, flag_buf):
        val_buf = self.critic(torch.from_numpy(cent_obs_buf)).sequence().detach().numpy()  # (num_envs, num_steps+1, cent_obs_dim) --> (num_envs, num_steps+1)
        advantage, reward_to_go = {}, {}
        for i in rew_agent_buf.shape[0]: # num_agents
            advantage[str(i)], reward_to_go[str(i)] = np.array([]), np.array([])
            for j in rew_agent_buf.shape[1]: # num_envs
                flag = flag_buf[i][j]
                val = val_buf[j][flag]
                rew = rew_agent_buf[i][j][rew_agent_buf[i][j]>self.minus_inf]
                rew = np.append(rew, values=0, axis=0) # Just fill up the space
                delta = rew[:-1] + self.gamma * val[1:] - val[:-1]
                advantage[str(i)] = np.append(arr=advantage[str(i)], values=discount_cumsum(delta, self.gamma * self.lam))
                reward_to_go[str(i)] = np.append(arr=reward_to_go[str(i)], values=discount_cumsum(rew, self.gamma)[:-1])

        # Transfer from numpy.ndarry to torch.Tensor
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in advantage.items()}, \
               {k: torch.as_tensor(v, dtype=torch.float32) for k,v in reward_to_go.items()}

    def preprocess(self, obs_agent_buf, obs_sequence_agent_buf, act_dis_buf, act_con_buf, logp_dis_buf, logp_con_buf):
        """

        :param obs_agent_buf:
        :param obs_sequence_agent_buf:
        :param act_dis_buf:
        :param act_con_buf:
        :param logp_dis_buf:
        :param logp_con_buf:
        :return:
        """
        observation_agent, observation_sequence_agent, action_dis, action_con, old_logp_dis, old_logp_con = {}, {}, {}, {}, {}, {}
        for i in obs_agent_buf.shape[0]: # num_agent
            # flags
            flag_obs = obs_agent_buf[i] > self.minus_inf
            flag_sequence_obs = obs_sequence_agent_buf[i] > self.minus_inf
            flag_act_con = action_con[i] > self.minus_inf
            flag = action_dis[i] > self.minus_inf

            observation_agent[str(i)] = obs_agent_buf[i][flag_obs].reshape(-1, self.obs_dim)
            observation_sequence_agent[str(i)] = obs_sequence_agent_buf[i][flag_sequence_obs].reshape(-1, self.sequence_dim)
            action_dis[str(i)] = act_dis_buf[i][flag]
            action_con[str(i)]= act_con_buf[i][flag_act_con].reshape(-1, self.act_dim)
            old_logp_dis[str(i)] = logp_dis_buf[i][flag]
            old_logp_con[str(i)] = logp_con_buf[i][flag]

        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in observation_agent.items()}, \
               {k: torch.as_tensor(v, dtype=torch.float32) for k,v in observation_sequence_agent.items()}, \
               {k: torch.as_tensor(v, dtype=torch.int32) for k,v in action_dis.items()}, \
               {k: torch.as_tensor(v, dtype=torch.float32) for k,v in action_con.items()}, \
               {k: torch.as_tensor(v, dtype=torch.float32) for k,v in old_logp_dis.items()}, \
               {k: torch.as_tensor(v, dtype=torch.float32) for k,v in old_logp_con.items()}















    @staticmethod
    def cal_value_loss(values, return_batch):
        """
        Calculate value function loss.

        :param values:
        :param return_batch:
        :return:
        """

        value_loss = 0.5 * ((return_batch - values) ** 2).mean()

        return value_loss


if __name__ == "__main__":
    actor = Actor(obs_space=8,
                  obs_signal_space=8,
                  hidden_size=[64, 32, 16],
                  action_space=8,
                  nonlinear='tanh',
                  init_log_std=-0.4)
    print(actor.actor_con)

    for param in actor.actor_con.parameters():
        print(param)