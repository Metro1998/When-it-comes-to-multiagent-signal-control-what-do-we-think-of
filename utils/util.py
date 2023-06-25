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


def autoregressive_act(decoder, obs_rep, env_num, agent_num, action_dim, action_dis, action_con, max_green, device):
    """
    In the rollout phase, it will infer the first action a_1 with a start signal a_0, then insert the back into the
    input and infer the a_2 with [a_0, a_1], and so on till a_n is inferred, aka, autoregressive.

    :param decoder:
    :param obs_rep:
    :param env_num:
    :param agent_num:
    :param action_dim:
    :param action_dis:
    :param action_con: (torch.Tensor) [env_num, agent_num]
    :param max_green:
    :param device:
    :return:
    """

    # Note that the start signal is [1, 0, 0, ... 0] with the length of action_dim + 1,
    # that‘s why the 3rd dimension of hybrid_action is action_dim (one hot) + 1 + 1 (raw continuous parameter)
    hybrid_action = torch.zeros((env_num, agent_num, action_dim + 2), dtype=torch.float32, device=device)
    hybrid_action[:, 0, 0] = 1
    output_action = torch.zeros((env_num, agent_num, 2), dtype=torch.float32, device=device)
    output_action_logp = torch.zeros_like(output_action, dtype=torch.float32, device=device)

    for i in range(agent_num):
        # Get the DISTRIBUTION according to the previous actions
        logits, means, stds = decoder(hybrid_action, obs_rep)
        logits = logits[:, i, :]  # TODO check the dimension
        means = means[:, i, :]
        stds = stds[:, i, :]

        # If it's time to make a decision rather than to infer, just sample the action from the distribution
        if map2real(action_con[i], max_green) < 0.1:
            # Mask the last discrete choice, which means the phase is not repeatable
            logits[action_dis[i]] = float('-inf')
            dist_dis = Categorical(logits=logits)
            act_dis = dist_dis.sample()
            act_logp_dis = dist_dis.log_prob(act_dis)

            dist_con = Normal(means[act_dis], stds[act_dis])
            act_con = dist_con.sample()
            act_logp_con = dist_con.log_prob(act_con)

        else:
            # If it's time to infer, just use the original discrete action and continuous parameter
            dist_dis = Categorical(logits=logits)
            act_dis = action_dis[i]
            act_logp_dis = dist_dis.log_prob(act_dis)

            dist_con = Normal(means[action_dis], stds[action_dis])
            act_con = action_con[i]
            act_logp_con = dist_con.log_prob(act_con)

        if i + 1 < agent_num:
            hybrid_action[i + 1, 1:-1] = F.one_hot(act_dis, num_classes=action_dim).float()
            hybrid_action[i + 1, -1] = act_con

        output_action[i, :] = torch.cat((act_dis.float(), act_con), -1)
        output_action_logp[i, :] = torch.cat((act_logp_dis, act_logp_con), -1)

    return output_action, output_action_logp  # [env_num, agent_num, 2]


def parallel_act(decoder, obs_rep, batch_size, agent_num, action_dim, action_dis, action_con, decision_flag, device):
    """
    In the training process, this function get input of sequence [a_0, a_1, ..., a_n-1] and predict [a_1, a_2, ..., a_n]
    simultaneously in one evaluate_actions process, aka, teaching force.

    :param decoder:
    :param obs_rep:
    :param batch_size:
    :param agent_num:
    :param action_dim:
    :param action_dis: (torch.int64) [batch_size, agent_num]
    :param action_con: [batch_size, agent_num]
    :param decision_flag: [batch_size, agent_num] to indicate whether it's time to make a decision or to infer,
    if it's time to make a decision, mask the last discrete choice
    :param device:
    :return:
    """
    hybrid_action = torch.zeros((batch_size, agent_num, action_dim + 2)).to(device)
    hybrid_action[:, 0, 0] = 1
    hybrid_action[:, 1:, 1:-1] = F.one_hot(action_dis, num_classes=action_dim)[:, :-1, :]
    hybrid_action[:, 1:, -1] = action_con.unsqueeze(-1)[:, :-1, :]
    logits, means, stds = decoder(hybrid_action, obs_rep)  # logits: [batch_size, agent_num, action_dim]

    # mask, if the action_dis is selected before, it shouldn't be selected at this time.
    action_dis_one_hot = F.one_hot(action_dis, num_classes=action_dim)
    mask = torch.where(decision_flag.unsqueeze(-1), action_dis_one_hot, torch.zeros_like(action_dis_one_hot))
    logits.masked_fill_(mask, float('-inf'))
    dist_dis = Categorical(logits=logits)
    act_logp_dis = dist_dis.log_prob(action_dis)
    entropy_dis = dist_dis.entropy()

    means = means.gather(-1, action_dis.unsqueeze(-1)).squeeze(-1)  # [batch_size, agent_num]
    stds = stds.gather(-1, action_dis.unsqueeze(-1)).squeeze(-1)  # [batch_size, agent_num]
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
