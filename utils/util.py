import scipy.signal
import torch
import numpy as np
from torch.distributions import Categorical, Normal
from torch.nn import functional as F


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


def autoregressive_act(decoder_dis, decoder_con, obs_rep, batch_size, agent_num, action_dim, device, available_actions=None):
    """
    In the rollout phase, it will infer the first action a_1 with a start signal a_0, then insert the back into the
    input and infer the a_2 with [a_0, a_1], and so on till a_n is inferred, aka, autoregressive.
    :param decoder_dis:
    :param decoder_con:
    :param obs_rep: (torch.Tensor) (batch_size, agent_num, embd_dim) the representation of the obs, which has been masked
    :param batch_size:
    :param agent_num:
    :param action_dim:
    :param device:
    :param available_actions: available discrete stages that an agent could select, and we still approximate the whole
    action space of continuous stages, since it will be reselect by discrete stages.
    :return:
    """

    shifted_action_dis = torch.zeros((batch_size, agent_num, action_dim + 1)).to(device)
    # Note that the start signal is [1, 0, 0, ... 0], that‘s why the 3rd dimension of action_dim + 1
    shifted_action_dis[: 0, 0] = 1
    output_action_dis = torch.zeros((batch_size, agent_num, 1), dtype=torch.long)
    output_action_log_dis = torch.zeros_like(output_action_dis, dtype=torch.float32)

    shifted_action_con = torch.zeros((batch_size, agent_num, action_dim)).to(device)
    output_action_con = torch.zeros((batch_size, agent_num, action_dim), dtype=torch.float32)
    output_action_log_con = torch.zeros_like(output_action_con, dtype=torch.float32)

    for i in range(agent_num):
        logit = decoder_dis(shifted_action_dis,obs_rep)[:, i, :]
        if available_actions is not None:
            logit[available_actions[:, i, :] == 0] = -1e10
        dist_dis = Categorical(logits=logit)
        act_dis = dist_dis.sample()
        act_log_dis = dist_dis.log_prob(act_dis)

        mean = decoder_con(shifted_action_con, obs_rep)[:, i, :]
        std = torch.clamp(F.softplus(decoder_con.log_std), min=0.01, max=0.5)
        dist_con = Normal(mean, std)
        act_con = dist_con.sample()
        act_log_con = dist_con.log_prob(act_con)

        output_action_dis[:, i, :] = act_dis.unsqueeze(-1)
        output_action_log_dis[:, i, :] = act_log_dis.unsqueeze(-1)
        output_action_con[:, i, :] = act_con
        output_action_log_con[:, i, :] = act_log_con
        if i + 1 < agent_num:
            shifted_action_dis[:, i + 1, 1:] = F.one_hot(act_dis, num_classes=action_dim)
            shifted_action_con[:, i + 1, :] = act_con

    return output_action_dis, output_action_log_dis, output_action_con, output_action_log_con


def parallel_act(decoder_dis, decoder_con, obs_rep, batch_size, agent_num, action_dim, act_dis, act_con, device, available_actions=None):
    """
    In the training process, this function get input of sequence [a_0, a_1, ..., a_n-1] and predict [a_1, a_2, ..., a_n]
    simultaneously in one evaluate_actions process, aka, teaching force.
    :param decoder_dis:
    :param decoder_con:
    :param obs_rep: (torch.Tensor) (batch_size, agent_num, embd_dim) the representation of the obs, which has been masked
    :param batch_size:
    :param agent_num:
    :param action_dim:
    :param act_dis: (torch.Tensor) (batch_size, agent_num, 1) the final discrete selection not the logits
    :param act_con: (torch.Tensor) (batch_size, agent_num, action_dim) all of the continuous heads
    :param device:
    :param available_actions: available discrete stages that an agent could select, and we still approximate the whole
    action space of continuous stages, since it will be reselect by discrete stages.
    :return:
    """
    one_hot_action = F.one_hot(act_dis.squeeze(-1), num_classes=action_dim)  # (batch_size, agent_num, action_dim)
    shifted_action_dis = torch.zeros((batch_size, agent_num, action_dim + 1)).to(device)
    shifted_action_dis[:, 0, 0] = 1
    shifted_action_dis[:, 1:, 1:] = one_hot_action[:, :-1, :]
    logit = decoder_dis(shifted_action_dis, obs_rep)
    if available_actions is not None:
        logit[available_actions == 0] = -1e10
    dist_dis = Categorical(logits=logit)
    act_log_dis = dist_dis.log_prob(act_dis.squeeze(-1)).unsqueeze(-1)
    entropy_dis = dist_dis.entropy().unsqueeze(-1)

    shifted_action_con = torch.zeros((batch_size, agent_num, action_dim)).to(device)
    shifted_action_con[:, 1:, :] = act_con[:, :-1, :]
    mean = decoder_con(shifted_action_con, obs_rep)
    std = torch.clamp(F.softplus(decoder_con.log_std), min=0.01, max=0.5)
    dist_con = Normal(mean, std)
    act_log_con = dist_con.log_prob(act_con)
    entropy_con = dist_con.entropy()

    return act_log_dis, entropy_dis, act_log_con, entropy_con
    

def batchify_obs(obs, device):
    """
    Converts dic style observations to batch of torch arrays.
    :param obs:
    :param device:
    :return:
    """
    obs = np.stack([obs[o] for o in obs], axis=0)
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)

    
    
