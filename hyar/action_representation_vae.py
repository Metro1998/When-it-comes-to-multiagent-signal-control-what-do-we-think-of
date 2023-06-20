"""
reference: 
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class VAE(nn.Module):

    def __init__(self, state_dim, action_dim, action_embedding_dim, parameter_action_dim, latent_dim, max_action,
                 hidden_size):
        super(VAE, self).__init__()

        # embedding table
        # TODO: be careful with the initialization
        init_tensor = torch.randn((action_dim, action_embedding_dim), dtype=torch.float32)
        self.embeddings = torch.nn.Parameter(init_tensor, requires_grad=True)

        # Encoder
        self.e0_dis = nn.Linear(state_dim + action_embedding_dim, hidden_size)
        self.e0_con = nn.Linear(parameter_action_dim, hidden_size)
        self.e1 = nn.Linear(hidden_size, hidden_size)
        self.e2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, latent_dim)
        self.log_std = nn.Linear(hidden_size, latent_dim)

        # Decoder
        self.d0_dis = nn.Linear(state_dim + action_embedding_dim, hidden_size)
        self.d0_con = nn.Linear(latent_dim, hidden_size)
        self.d1 = nn.Linear(hidden_size, hidden_size)
        self.d2 = nn.Linear(hidden_size, hidden_size)

        self.parameter_action_output = nn.Linear(hidden_size, parameter_action_dim)

        # cascaded head for dynamic predictive representation
        self.d3 = nn.Linear(hidden_size, hidden_size)
        self.delta_state_output = nn.Linear(hidden_size, state_dim)

        self.max_acton = max_action
        self.latent_dim = latent_dim

    def forward(self, state, action, action_parameter):
        """

        :param state:
        :param action:
        :param action_parameter:
        :return:
        """

        z, mean, std = self.encode(state, action, action_parameter)
        u, s, mean, std = self.decode(state, z, action)

        return u, s, mean, std

    def encode(self, state, action, action_parameter):
        """
        the encoder parameterized by φ takes s and the embedding eζ,k as condition, and maps xk into the latent variable z ∈ Rd2
        :param state:
        :param action:
        :param action_parameter:
        :return:
        """
        z_dis = F.relu(self.e0_dis(torch.cat([state, action], 1)))
        z_con = F.relu(self.e0_con(action_parameter))

        z = z_dis * z_con

        z = F.relu(self.e1(z))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # the recommended trick in What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study

        log_std = self.log_std(z)
        std = torch.clamp(F.softplus(log_std), min=0.01, max=0.6)

        z = mean + std * torch.rand_like(std)

        return z, mean, std

    def decode(self, state, z=None, action=None, raw=False):
        """

        :param state:
        :param z:
        :param action:
        :param raw:
        :return:
        """

        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        # if z is None:
        #     z = torch.randn((state.shape[0], self.latent_dim)).to(device)
        #     if clip is not None:
        #         z = z.clamp(-clip, clip)
        v_dis = F.relu(self.d0_dis(torch.cat([state, action], 1)))
        v_con = F.relu(self.d0_con(z))

        v = v_dis * v_con
        v = F.relu(self.d1(v))
        v = F.relu(self.d2(v))

        parameter_action = self.parameter_action_output(v)

        v = F.relu(self.d3(v))
        delta_s = self.delta_state_output(v)

        if raw:
            return parameter_action, delta_s
        else:
            return self.max_acton * torch.tanh(parameter_action), torch.tanh(delta_s)


class Action_Representation(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 parameter_action_dim,
                 reduced_action_dim=2,
                 reduced_parameter_action_dim=2,
                 embed_lr=1e-4,
                 ):
        super(Action_Representation, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.parameter_action_dim = parameter_action_dim
        self.reduced_action_dim = reduced_action_dim
        self.reduced_parameter_action_dim = reduced_parameter_action_dim

        self.latent_dim = self.reduced_parameter_action_dim
        self.embed_lr = embed_lr
        self.vae = VAE(state_dim=state_dim, action_dim=action_dim, action_embedding_dim=reduced_action_dim,
                       parameter_action_dim=parameter_action_dim, latent_dim=self.latent_dim, max_action=1, hidden_size=256)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.embed_lr)



