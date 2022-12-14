"""
Include basic neural network, buffer, and specific implementation of HPPO
Author:Metro
date:2022.12.13
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self,
                 state_sapce,
                 state_space_signal,
                 hidden_size,
                 action_space,
                 nonlinear,
                 init_log_std
                 ):
        """
        Centeralized execution, that is, the agent_i output its hybrid action based on the global
        state representation, but without other agents' policies.
        Mask is considered to narrow the state space of actor, it is suggested agents far from the
        target agent seldomly influence the latter one.

        :param state_sapce: the size of state
        :param state_space_signal: the size of state_signal
        :param hidden_size: the size of hidden layers [hidden_size[0], hidden_size[1], hidden_size[2]]
        :param action_space: the size of output (discrete) dimension
        :param nonlinear: the nonlinear activation
        """
        super().__init__()

        self.nonlinear = nn.ReLU() if nonlinear == 'relu' else nn.Tanh()
        self.log_std = nn.Parameter(torch.zeros(action_space, ) + init_log_std, requires_grad=True)

        # actor_con
        self.actor_con = nn.Sequential(
            nn.Linear(state_sapce, hidden_size[0]),
            self.nonlinear,
            nn.Linear(hidden_size[0], hidden_size[1]),
            self.nonlinear,
            nn.Linear(hidden_size[1], hidden_size[2]),
            self.nonlinear,
            nn.Linear(hidden_size[2], action_space)
        )

        # actor_dis
        self.rnn = nn.LSTM(input_size=state_space_signal, hidden_size=hidden_size[0] // 2)
        self.linear = nn.Linear(state_sapce, hidden_size[0] // 2)
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
        # output layer weights with 0.01???) to be beneficial
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
        One benefit of applying Equation (9) is that agents only need to maintain a joint advantage estimator A??(s, a)
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


if __name__ == "__main__":
    actor = Actor(state_sapce=8,
                  state_space_signal=8,
                  hidden_size=[64, 32, 16],
                  action_space=8,
                  nonlinear='tanh',
                  init_log_std=-0.4)
    print(actor.actor_con)

    for param in actor.actor_con.parameters():
        print(param)
