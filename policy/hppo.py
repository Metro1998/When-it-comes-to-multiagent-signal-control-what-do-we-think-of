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
from utils.util import *


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

    def evaluate_action_dis(self, obs, obs_signal, action_dis):
        """
        Compute log probability and entropy of given stages.
        :param obs: the ordinary information (observation) of the agent (and its neighbors)
        :param obs_signal: the information of signal sequence  (sequence_length, state_space_signal)
        :param action_dis: the discrete action
        :return:
        """
        action_probs = self.dis_forward(obs, obs_signal)
        dist_dis = Categorical(action_probs)
        logprob_dis = dist_dis.log_prob(action_dis)
        dist_entropy_dis = dist_dis.entropy()

        return logprob_dis, dist_entropy_dis

    def evaluate_action_con(self, obs, action_con):
        """
        Compute log probability and entropy of given stages.
        :param obs: the ordinary information (observation) of the agent (and its neighbors)
        :param action_con: the continuous action
        :return:
        """
        mean = self.actor_con(obs)
        std = torch.clamp(F.softplus(self.log_std), min=0.01, max=0.5)
        dist_con = Normal(mean, std)
        logprob_con = dist_con.log_prob(action_con)
        dist_entropy_con = dist_con.entropy()

        return logprob_con, dist_entropy_con


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
                 max_norm, coeff_entropy_dis, coeff_entropy_con, random_seed, lr_actor_con, lr_actor_dis, lr_std, lr_critic, lr_decay_rate,
                 target_kl_dis, target_kl_con, queue_dim, signal_dim, act_dim, device):

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
        self.queue_dim = queue_dim
        self.signal_dim = signal_dim
        self.act_dim = act_dim
        self.device = device
        self.num_agents = num_agents
        self.coeff_entropy_dis = coeff_entropy_dis
        self.coeff_entropy_con = coeff_entropy_con

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

        cent_obs_buf, rew_buf, obs_queue_buf, obs_signal_buf, act_dis_buf, act_con_buf, logp_dis_buf, logp_con_buf = self.buffer.get()

        obs_queue_dic, obs_signal_dic, act_dis_dic, act_con_dic, old_logp_dis_dic, old_logp_con_dic, state, ret = \
            self.preprocess(obs_queue_buf, obs_signal_buf, act_con_buf, act_dis_buf, logp_dis_buf, logp_con_buf,
                            cent_obs_buf, rew_buf)

        update_dis_actor, update_con_actor = 1, 1

        for i in self.epochs:
            # Recompute values at the beginning of each epoch
            advantage = self.recompute(cent_obs_buf, rew_buf)
            # Normalization
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # Update global critic
            sampler_critic = list(BatchSampler(
                sampler=SubsetRandomSampler(state.shape[0]),
                batch_size=self.batch_size,
                drop_last=True))
            for indices in sampler_critic:
                state_batch = torch.as_tensor(state[indices], dtype=torch.float32, device=self.device)
                ret_batch = torch.as_tensor(ret[indices], dtype=torch.float32, device=self.device)

                # Update gloable critic
                self.optimizer_critic.zero_grad()
                critic_loss = self.compute_critic_loss(state_batch, ret_batch)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), norm_type=2, max_norm=self.max_norm)
                self.optimizer_critic.step()

                adv_batch_dis = torch.as_tensor(advantage[indices], dtype=torch.float32, device=self.device)
                adv_batch_con = adv_batch_dis
                # Update actors
                if update_dis_actor:
                    for j in self.permutation:
                        obs_queue_batch = torch.as_tensor(obs_queue_dic[str(j)], dtype=torch.float32, device=self.device)
                        obs_signal_batch = torch.as_tensor(obs_signal_dic[str(j)], dtype=torch.float32, device=self.device)
                        act_dis_batch = torch.as_tensor(act_dis_dic[str(j)], dtype=torch.int64, device=self.device)
                        old_logp_dis_batch = torch.as_tensor(old_logp_dis_dic[str(j)], dtype=torch.float32, device=self.device)

                        actor_loss_dis, adv_batch_dis, approx_kl_dis = \
                            self.compute_actor_dis_loss(obs_queue_batch, obs_signal_batch, act_dis_batch, old_logp_dis_batch, self.actors[j], adv_batch_dis)

                        self.optimizer_actor_dis[j].zero_grad()
                        actor_loss_dis.backward()
                        torch.nn.utils.clip_grad_norm_(self.optimizer_actor_dis[j], norm_type=2, max_norm=self.max_norm)
                        self.optimizer_actor_dis[j].step()

                if update_con_actor:
                    for j in self.permutation:
                        obs_queue_batch = torch.as_tensor(obs_queue_dic[str(j)], dtype=torch.float32, device=self.device)
                        act_dis_batch = torch.as_tensor(act_dis_dic[str(j)], dtype=torch.int64, device=self.device)
                        act_con_batch = torch.as_tensor(act_con_dic[str(j)], dtype=torch.float32, device=self.device)
                        old_logp_con_batch = torch.as_tensor(old_logp_con_dic[str(j)], dtype=torch.float32, device=self.device)

                        actor_loss_con, adv_batch_con, approx_kl_con = \
                            self.compute_actor_con_loss(obs_queue_batch, act_dis_batch, act_con_batch, old_logp_con_batch, self.actors[j], adv_batch_con)

                        self.optimizer_actor_con[j].zero_grad()
                        actor_loss_con.backward()
                        torch.nn.utils.clip_grad_norm_(self.optimizer_actor_con[j], norm_type=2,
                                                       max_norm=self.max_norm)
                        self.optimizer_actor_con[j].step()

            if approx_kl_dis > self.target_kl_dis:
                update_dis_actor = 0
            if approx_kl_con > self.target_kl_con:
                update_con_actor = 0

    def recompute(self, cent_obs_buf, rew_buf):
        """
        Compute advantage function A(s, a) based on global V-value_proj network with GAE, where a represent joint action

        :param cent_obs_buf:
        :param rew_buf:
        :return:
        """

        # (num_envs, num_steps, cent_obs_dim) --> (num_envs, num_steps)
        val_buf = self.critic(torch.from_numpy(cent_obs_buf)).sequence().detach().numpy()
        # (num_envs, num_steps, num_agents) --> (num_envs, num_steps)
        rew_buf = np.sum(rew_buf, axis=-1)

        advantage, reward_to_go = np.array([])
        for i in val_buf.shape[0]:  # num_envs
            val = val_buf[i]
            rew = rew_buf[i]

            # the next two lines implement GAE-Lambda advantage calculation
            delta = rew[:-1] + self.gamma * val[1:] - val[:-1]
            advantage = np.append(advantage, discount_cumsum(delta, self.gamma * self.lam))

        return advantage

    def preprocess(self, obs_queue_buf, obs_signal_buf, act_dis_buf, act_con_buf, logp_dis_buf, logp_con_buf,
                   cent_obs_buf, rew_buf):
        """

        :param obs_queue_buf:
        :param obs_signal_buf:
        :param act_dis_buf:
        :param act_con_buf:
        :param logp_dis_buf:
        :param logp_con_buf:
        :param cent_obs_buf:
        :param rew_buf:
        :return:
        """
        obs_queue_dic, obs_signal_dic, act_dis_dic, act_con_dic, old_logp_dis_dic, old_logp_con_dic = {}, {}, {}, {}, {}, {}
        for i in obs_queue_buf.shape[0]: # num_agents
            obs_queue_dic[str(i)] = obs_queue_buf[i][:, :-1].flatten().reshape(-1, self.queue_dim)
            obs_signal_dic[str(i)] = obs_signal_buf[i][:, :-1].flatten().reshape(-1, self.signal_dim)
            act_dis_dic[str(i)] = act_dis_buf[i][:, :-1].flatten()
            act_con_dic[str(i)] = act_con_buf[i][:, :-1].flatten().reshape(-1, self.act_dim)
            old_logp_dis_dic[str(i)] = logp_dis_buf[i][:, :-1].flatten()
            old_logp_con_dic[str(i)] = logp_con_buf[i][:, :-1].flatten()

        rew_buf = np.sum(rew_buf, axis=-1)
        state, ret = np.array([]), np.array([])
        for i in rew_buf.shape[0]:  # num_envs
            cent_obs = cent_obs_buf[i]
            rew = rew_buf[i]

            state = np.append(state, cent_obs[:-1])
            # the next line computes rewards-to-go, to be targets for the value_proj function
            ret = np.append(ret, discount_cumsum(rew, self.gamma)[:-1])

        return obs_queue_dic, obs_signal_dic, act_dis_dic, act_con_dic, old_logp_dis_dic, old_logp_con_dic, state, ret

    def compute_critic_loss(self, state_batch, ret_batch):
        """

        :param state_batch:
        :param ret_batch:
        :return:
        """
        state_values = self.critic(state_batch)
        return self.loss_func(state_values, ret_batch)

    def compute_actor_dis_loss(self, obs_queue_batch, obs_signal_batch, act_dis_batch, old_logp_dis_batch, actor, adv_target_dis):
        """

        :param obs_queue_batch:
        :param obs_signal_batch:
        :param act_dis_batch:
        :param old_logp_dis_batch:
        :param actor:
        :param adv_target_dis:
        :return:
        """
        logp_dis, entropy_dis = actor.evaluate_action_dis(obs_queue_batch, obs_signal_batch, act_dis_batch)
        imp_weights_dis = torch.exp(logp_dis - old_logp_dis_batch)
        surr1_dis = imp_weights_dis * adv_target_dis
        surr2_dis = torch.clamp(imp_weights_dis, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_target_dis
        loss_pi_dis = - (torch.min(surr1_dis, surr2_dis) + self.coeff_entropy_dis * entropy_dis).mean()

        adv_target_dis *= imp_weights_dis
        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            approx_kl_dis = ((imp_weights_dis - 1) - (logp_dis - old_logp_dis_batch)).mean()

        return loss_pi_dis, adv_target_dis, approx_kl_dis

    def compute_actor_con_loss(self, obs_queue_batch, act_dis_batch, act_con_batch, old_logp_con_batch, actor, adv_target_con):
        """

        :param obs_queue_batch:
        :param act_dis_batch:
        :param act_con_batch:
        :param old_logp_con_batch:
        :param actor:
        :param adv_target_con:
        :return:
        """
        logp_con, entropy_con = actor.evaluate_action_dis(obs_queue_batch, act_con_batch)
        logp_con = logp_con.gather(1, act_dis_batch.view(-1, 1)).squeeze()
        imp_weights_con = torch.exp(logp_con - old_logp_con_batch)
        surr1_con = imp_weights_con * adv_target_con
        surr2_con = torch.clamp(imp_weights_con, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_target_con

        # Andrychowicz, et al. (2021) overall find no evidence that the entropy term improves performance on
        # continuous control environments.
        loss_pi_con = - torch.min(surr1_con, surr2_con).mean()

        adv_target_con *= imp_weights_con
        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            approx_kl_con = ((imp_weights_con - 1) - (logp_con - old_logp_con_batch)).mean()

        return loss_pi_con, adv_target_con, approx_kl_con


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
