"""
Include basic neural network, buffer, and specific implementation of HPPO
Author:Metro
date:2022.12.13
"""
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self,
                 input_size,
                 input_size_signal,
                 hidden_size,
                 output_size,
                 nonlinear
                 ):
        """

        :param input_size: the size of state
        :param input_size_signal: the size of state_signal
        :param hidden_size: the size of hidden layers [hidden_size[0], hidden_size[1], hidden_size[2]]
        :param output_size: the size of output dimension
        :param nonlinear: the nonlinear activation
        """
        super().__init__()

        self.nonlinear = nn.ReLU() if nonlinear == 'relu' else nn.Tanh()

        # actor_con
        self.actor_con = nn.Sequential(
            nn.Linear(input_size, hidden_size[0]),
            self.nonlinear,
            nn.Linear(hidden_size[0], hidden_size[1]),
            self.nonlinear,
            nn.Linear(hidden_size[1], hidden_size[2]),
            self.nonlinear,
            nn.Linear(hidden_size[2], 1)
        )

        # actor_dis
        self.rnn = nn.LSTM(input_size=input_size_signal, hidden_size=hidden_size[0] / 2)
        self.linear = nn.Linear(in_features=input_size, out_features=hidden_size[0] / 2)
        self.actor_dis = nn.Sequential(
            nn.Linear(hidden_size[0], hidden_size[1]),
            self.nonlinear,
            nn.Linear(hidden_size[1], hidden_size[2]),
            self.nonlinear,
            nn.Linear(hidden_size[2], output_size),
            nn.Softmax(dim=-1)
        )

        # critic # TODO

    def dis_forward(self, signal_sequence, state):
        """

        :param signal_sequence: the information of signal sequence  (sequence_length, input_size_signal)
        :param state: the ordinary information (observation) of the agent (and its neighbors)
        :return:
        """

        out_put, (hn, cn) = self.rnn(signal_sequence)
        h = self.linear(state)
        x = self.actor_dis(torch.cat((hn.squeeze(), h), dim=-1))

        return x




