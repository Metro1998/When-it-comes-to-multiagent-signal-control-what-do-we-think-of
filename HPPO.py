"""
Include basic neural network, and specific implementation of HPPO
Author:Metro
date:2022.12.13
"""
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self,
                 observation_space,
                 observation_space_signal,
                 hidden_size,
                 action_space,
                 nonlinear,
                 init_log_std
                 ):
        """
        Distributed execution, that is, the agent_i output its hybrid action based on its own observation
        without other agents' policies and global state representation.

        :param observation_space: the size of state
        :param observation_space_signal: the size of state_signal
        :param hidden_size: the size of hidden layers [hidden_size[0], hidden_size[1], hidden_size[2]]
        :param action_space: the size of output (discrete) dimension
        :param nonlinear: the nonlinear activation
        """
        super().__init__()

        self.nonlinear = nn.ReLU() if nonlinear == 'relu' else nn.Tanh()
        self.log_std = nn.Parameter(torch.zeros(action_space, ) + init_log_std, requires_grad=True)

        # actor_con
        self.actor_con = nn.Sequential(
            nn.Linear(observation_space, hidden_size[0]),
            self.nonlinear,
            nn.Linear(hidden_size[0], hidden_size[1]),
            self.nonlinear,
            nn.Linear(hidden_size[1], hidden_size[2]),
            self.nonlinear,
            nn.Linear(hidden_size[2], action_space)
        )

        # actor_dis
        self.rnn = nn.LSTM(input_size=observation_space_signal, hidden_size=hidden_size[0] // 2)
        self.linear = nn.Linear(observation_space, hidden_size[0] // 2)
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

    def dis_forward(self, signal_sequence, state):
        """

        :param signal_sequence: the information of signal sequence  (sequence_length, state_space_signal)
        :param state: the ordinary information (observation) of the agent (and its neighbors)
        :return:
        """

        out_put, (hn, cn) = self.rnn(signal_sequence)
        h = self.linear(state)
        x = self.actor_dis(torch.cat((hn.squeeze(), h), dim=-1))

        return x

    def act(self, signal_sequence, state):
        """

        :param signal_sequence: the information of signal sequence  (sequence_length, state_space_signal)
        :param state: the ordinary information (observation) of the agent (and its neighbors)
        :return:
        """

        action_probs = self.dis_forward(signal_sequence, state)
        dist_dis = Categorical(action_probs)
        action_dis = dist_dis.sample()
        logprob_dis = dist_dis.log_prob(action_dis)

        mean = self.actor_con(state)
        std = torch.clamp(F.softplus(self.log_std), min=0.01, max=0.5)
        dist_con = Normal(mean, std)
        action_con = dist_con.sample()
        logprob_con = dist_con.log_prob(action_con)

        return action_dis, logprob_dis, action_con, logprob_con

    def evluate_actions(self, signal_sequence, state, action_dis, action_con):
        """
        Compute log probability and entropy of given actions.
        :param signal_sequence: the information of signal sequence  (sequence_length, state_space_signal)
        :param state: the ordinary information (observation) of the agent (and its neighbors)
        :param action_dis: the discrete action
        :param action_con: the continuous action
        :return:
        """

        action_probs = self.dis_forward(signal_sequence, state)
        dist_dis = Categorical(action_probs)
        logprob_dis = dist_dis.log_prob(action_dis)
        dist_entropy_dis = dist_dis.entropy()

        mean = self.actor_con(state)
        std = torch.clamp(F.softplus(self.log_std), min=0.01, max=0.5)
        dist_con = Normal(mean, std)
        logprob_con = dist_con.log_prob(action_con)
        dist_entropy_con = dist_con.entropy()

        return logprob_dis, dist_entropy_dis, logprob_con, dist_entropy_con


class Critic(nn.Module):
    def __init__(self,
                 state_sapce,
                 hidden_size,
                 nonlinear):
        """
        One benefit of applying Equation (9) is that agents only need to maintain a joint advantage estimator Aπ(s, a)
        rather than one centralised critic for each individual agent (e.g., unlike CTDE methods such as MADDPG).

        :param state_sapce: the size of state
        :param hidden_size: the size of hidden layers [hidden_size[0], hidden_size[1], hidden_size[2]]
        :param nonlinear: the nonlinear activation
        """
        super().__init__()

        self.nonlinear = nn.ReLU() if nonlinear == 'relu' else nn.Tanh()

        self.critic = nn.Sequential(
            nn.Linear(state_sapce, hidden_size[0]),
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


class PPO:
    def __init__(self, actors, critic, rollout_buffer, epochs, mini_batch, clip_ratio, max_norm, coeff_entropy,
                 random_seed, lr_actor_con, lr_actor_dis, lr_std, lr_critic, lr_decay_rate, target_kl_dis,
                 target_kl_con, init_log_std, device):

        self.random_seed = random_seed
        self.actors = [a.to(device) for a in actors]
        self.actors_old = [a.to(device) for a in actors]
        self.critic = critic.to(device)

        self.optimizer_actor_con = [torch.optim.Adam([
            {'params': a.actor_con.parameters(), 'lr': lr_actor_con},
            {'params': a.log_std, 'lr': lr_std}
        ]) for a in self.actors]
        self.optimizer_actor_dis = [torch.optim.Adam([
            {'params': a.actor_dis.parameters(), 'lr': lr_actor_dis}
        ]) for a in self.actors]
        self.optimizer_critic = torch.optim.Adam(critic.parameters(), lr=lr_critic)

        for _ in self.optimizer_actor_con:
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
        :param random_seed:
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

if __name__ == "__main__":
    actor = Actor(observation_space=8,
                  observation_space_signal=8,
                  hidden_size=[64, 32, 16],
                  action_space=8,
                  nonlinear='tanh',
                  init_log_std=-0.4)
    print(actor.actor_con)

    for param in actor.actor_con.parameters():
        print(param)