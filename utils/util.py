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


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


def remap(time_remaining, max_green):
    """
    Remap the remaining time to its original range.
    :param time_remaining:
    :param max_green:
    :return:
    """
    return -1 + 2 * time_remaining / max_green


def map2real(raw_con, max_green):
    """
    Map the raw continuous parameter to the range of [0, max_green]
    :param raw_con:
    :param max_green:
    :return:
    """
    return (raw_con + 1) * max_green / 2


def autoregressive_act(decoder, obs_rep, batch_size, agent_num, action_dim, action_dis, action_con, agent_to_update, device):
    hybrid_action = torch.zeros((batch_size, agent_num, action_dim + 2), dtype=torch.float32, device=device)
    hybrid_action[:, 0, 0] = 1

    for i in range(agent_num):
        with torch.no_grad():
            logits, means, stds = decoder(hybrid_action, obs_rep)
            logits = logits[:, i]
            means = means[:, i]
            stds = stds[:, i]

            if agent_to_update[:, i]:
                if action_dis[:, i] >= 0:
                    logits[action_dis[:, i]] = float('-inf')
                dist_dis = Categorical(logits=logits)
                act_dis = dist_dis.sample()

                dist_con = Normal(means[act_dis], stds[act_dis])
                act_con = dist_con.sample()

            else:
                dist_dis = Categorical(logits=logits)
                act_dis = action_dis[:, i]

                dist_con = Normal(means[action_dis], stds[action_dis])
                act_con = action_con[:, i]

            act_logp_dis = dist_dis.log_prob(act_dis)
            act_logp_con = dist_con.log_prob(act_con)

            if i + 1 < agent_num:
                hybrid_action[:, i + 1, 1:-1].copy_(F.one_hot(act_dis, num_classes=action_dim).float())
                hybrid_action[:, i + 1, -1] = act_con

        if i == 0:
            output_act_dis = act_dis
            output_act_con = act_con
            output_logp_dis = act_logp_dis
            output_logp_con = act_logp_con
        else:
            output_act_dis = torch.cat((output_act_dis, act_dis), dim=1)
            output_act_con = torch.cat((output_act_con, act_con), dim=1)
            output_logp_dis = torch.cat((output_logp_dis, act_logp_dis), dim=1)
            output_logp_con = torch.cat((output_logp_con, act_logp_con), dim=1)

    return output_act_dis, output_logp_dis, output_act_con, output_logp_con


def parallel_act(decoder, obs_rep, batch_size, agent_num, action_dim, action_dis, action_con, decision_flag, device):
    hybrid_action = torch.zeros((batch_size, agent_num, action_dim + 2), device=device)
    hybrid_action[:, 0, 0] = 1
    hybrid_action[:, 1:, 1:-1].copy_(F.one_hot(action_dis, num_classes=action_dim)[:, :-1, :])
    hybrid_action[:, 1:, -1] = action_con.unsqueeze(-1)[:, :-1, :]
    logits, means, stds = decoder(hybrid_action, obs_rep)

    action_dis_one_hot = F.one_hot(action_dis, num_classes=action_dim)
    mask = decision_flag.unsqueeze(-1).float() * action_dis_one_hot
    logits.masked_fill_(mask, float('-inf'))
    dist_dis = Categorical(logits=logits)
    act_logp_dis = dist_dis.log_prob(action_dis)
    entropy_dis = dist_dis.entropy()

    means = means.gather(-1, action_dis.unsqueeze(-1)).squeeze(-1)
    stds = stds.gather(-1, action_dis.unsqueeze(-1)).squeeze(-1)
    dist_con = Normal(means, stds)
    act_logp_con = dist_con.log_prob(action_con)
    entropy_con = dist_con.entropy()

    return act_logp_dis, entropy_dis, act_logp_con, entropy_con


# def batchify_obs(obs, device):
#     """
#     Converts dic style observations to batch of torch arrays.
#     :param obs:
#     :param device:
#     :return:
#     """
#     obs = np.stack([obs[o] for o in obs], axis=0)
#     obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
