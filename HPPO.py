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
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
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
    def __init__(self, buffer, actors, critic, num_agents, rollout_buffer, epochs, batch_size, clip_ratio, gamma, lam,
                 max_norm, coeff_entropy, random_seed, lr_actor_con, lr_actor_dis, lr_std, lr_critic, lr_decay_rate,
                 target_kl_dis, target_kl_con, init_log_std, minus_inf, obs_dim, sequence_dim, act_dim, device):

        self.buffer = rollout_buffer
        self.actors = [a.to(device) for a in actors]
        self.actors_old = [a.to(device) for a in actors]
        self.critic = critic.to(device)
        self.random_seed = random_seed
        self.clip_ratio = clip_ratio
        self.batch_size = batch_size
        self.epochs = epochs
        self.gamma = gamma
        self.lam = lam
        self.max_norm = max_norm
        self.minus_inf = minus_inf
        self.obs_dim = obs_dim
        self.sequence_dim = sequence_dim
        self.act_dim = act_dim
        self.device = device
        self.num_agents = num_agents

        # to offer a random permutation
        self.permutation = np.arange(num_agents)
        np.random.shuffle(self.permutation)  # TODO

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

        self.loss_func = nn.SmoothL1Loss(reduction='mean')

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

        cent_obs_buf, rew_buf, obs_agent_buf, obs_sequence_agent_buf, act_dis_buf, act_con_buf, logp_dis_buf, logp_con_buf = self.buffer.get()

        obs_agent, obs_sequence_agent, act_dis, act_con, old_logp_dis, old_logp_con, cent_obs, ret = \
            self.preprocess(obs_agent_buf, obs_sequence_agent_buf, act_con_buf, act_dis_buf, logp_dis_buf, logp_con_buf, cent_obs_buf, rew_buf)

        for i in self.epochs:
            # Recompute values at the beginning of each epoch
            advantage = self.recompute(cent_obs_buf, rew_buf)
            # Normalization
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # Update global critic
            sampler_critic = list(BatchSampler(
                    sampler=SubsetRandomSampler(cent_obs.shape[0]),
                    batch_size=self.batch_size,
                    drop_last=True))
            for indices in sampler_critic:
                cent_obs_batch = torch.as_tensor(cent_obs[indices], dtype=torch.float32, device=self.device)
                ret_batch = torch.as_tensor(ret[indices], dtype=torch.float32, device=self.device)
                critic_loss = self.compute_critic_loss(cent_obs_batch, ret_batch)

                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), norm_type=2, max_norm=self.max_norm)
                self.optimizer_critic.step()

            # Update actors

            # Retrieve the mini_batch_num, since the number of available batchs of each agent will be different
            sampler = {}
            for j in self.permutation:
                sampler[str(j)] = list(BatchSampler(
                    sampler=SubsetRandomSampler(advantage[str(j)].shape[-1]),
                    batch_size=self.batch_size,
                    drop_last=True))
            mini_batch_num = min([len(v) for k, v in sampler.items()])

            for k in range(mini_batch_num):
                for j in self.permutation:
                    obs_batch = torch.as_tensor(obs_agent[str(j)][sampler[str(j)][k]], dtype=torch.float32, device=self.device)
                    obs_sequence_batch = torch.as_tensor(advantage[str(j)][sampler[str(j)][k]], dtype=torch.float32, device=self.device)
                    act_dis_batch = torch.as_tensor(act_dis[str(j)][sampler[str(j)][k]], dtype=torch.int64, device=self.device)
                    act_con_batch = torch.as_tensor(act_con[str(j)][sampler[str(j)][k]], dtype=torch.float32, device=self.device)
                    logp_dis_batch = torch.as_tensor(old_logp_dis[str(j)][sampler[str(j)][k]], dtype=torch.float32, device=self.device)
                    logp_con_batch = torch.as_tensor(old_logp_con[str(j)][sampler[str(j)][k]], dtype=torch.float32, device=self.device)

    def recompute(self, cent_obs_buf, rew_buf):
        """
        Compute advantage function A(s, a) based on global V-value network with GAE, where a represent joint action

        :param cent_obs_buf:
        :param rew_buf:
        :return:
        """

        # (num_envs, num_steps, cent_obs_dim) --> (num_envs, num_steps)
        val_buf = self.critic(torch.from_numpy(cent_obs_buf)).sequence().detach().numpy()
        # (num_envs, num_steps, num_agents) --> (num_envs, num_steps)
        rew_buf = np.sum(rew_buf, axis=-1)

        advantage, reward_to_go = np.array([]), np.array([])
        for i in val_buf.shape[0]:  # num_envs
            val = val_buf[i]
            rew = rew_buf[i]

            # the next two lines implement GAE-Lambda advantage calculation
            delta = rew[:-1] + self.gamma * val[1:] - val[:-1]
            advantage = np.append(advantage, discount_cumsum(delta, self.gamma * self.lam))

        return advantage

    def preprocess(self, obs_agent_buf, obs_sequence_agent_buf, act_dis_buf, act_con_buf, logp_dis_buf, logp_con_buf, cent_obs_buf, rew_buf):
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
        for i in obs_agent_buf.shape[0]:  # num_agent
            # flags
            flag_obs = obs_agent_buf[i] > self.minus_inf
            flag_sequence_obs = obs_sequence_agent_buf[i] > self.minus_inf
            flag_act_con = action_con[i] > self.minus_inf
            flag = action_dis[i] > self.minus_inf

            observation_agent[str(i)] = obs_agent_buf[i][flag_obs].reshape(-1, self.obs_dim)
            observation_sequence_agent[str(i)] = obs_sequence_agent_buf[i][flag_sequence_obs].reshape(-1, self.sequence_dim)
            action_dis[str(i)] = act_dis_buf[i][flag]
            action_con[str(i)] = act_con_buf[i][flag_act_con].reshape(-1, self.act_dim)
            old_logp_dis[str(i)] = logp_dis_buf[i][flag]
            old_logp_con[str(i)] = logp_con_buf[i][flag]

        # (num_envs, num_steps, num_agents) --> (num_envs, num_steps)
        rew_buf = np.sum(rew_buf, axis=-1)
        centralized_observation, reward_to_go = np.array([]), np.array([])
        for i in rew_buf.shape[0]:  # num_envs
            cent_obs = centralized_observation[i]
            rew = rew_buf[i]

            centralized_observation = np.append(centralized_observation, cent_obs[:-1])
            # the next line computes rewards-to-go, to be targets for the value function
            reward_to_go = np.append(reward_to_go, discount_cumsum(rew, self.gamma)[:-1])

        return observation_agent, observation_sequence_agent, action_dis, action_con, old_logp_dis, old_logp_con, centralized_observation, reward_to_go

    def compute_critic_loss(self, cent_obs_batch, ret_batch):
        """

        :param cent_obs_batch:
        :param ret_batch:
        :return:
        """
        state_values = self.critic(cent_obs_batch)

        return self.loss_func(state_values, ret_batch)

    def compute_actor_loss(self, obs_batch, obs_sequence_batch, act_dis_batch, act_con_batch, old_logp_dis, old_logp_con, actor, adv_targ):
        """

        :param obs_batch:
        :param obs_sequence_batch:
        :param act_dis_batch:
        :param act_con_batch:
        :param old_logp_dis:
        :param old_logp_con:
        :return:
        """
        logp_dis, entropy_dis, logp_con, entropy_con = actor.evaluate_actions(obs_batch, obs_sequence_batch, act_dis_batch, act_con_batch)
        logp_con = logp_con.gather(1, act_dis_batch.view(-1, 1)).squeeze()
        imp_weights_dis = torch.prod(torch.exp(logp_dis - old_logp_dis), dim=-1, keepdim=True)
        imp_weights_con = torch.prod(torch.exp(logp_con - old_logp_con), dim=-1, keepdim=True)
        surr1_dis = imp_weights_dis * adv_targ
        surr2_dis = torch.clamp(imp_weights_dis, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_targ
        surr1_dis = imp_weights_con * adv_targ
        surr2_dis = torch.clamp(imp_weights_con, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_targ

        # 基本思路要改变，也就是说在每个step上 obs 和 adv都是有的，那么现在的目标是在这些step上面batch 然后针对某个step（肯定有智能体行动）进行更新， 不过大概率只有一个智能体，这时候不就退化成IPPO？




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
