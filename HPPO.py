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
            nn.Linear(hidden_size[2], output_size)
        )


    def forward(self): # TODO


