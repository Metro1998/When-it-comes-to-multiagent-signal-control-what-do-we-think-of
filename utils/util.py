import time

import scipy.signal
import torch
import numpy as np
import xml.etree.ElementTree as ET
from torch.distributions import Categorical, Normal
from torch.nn import functional as F


# def convert_array(array):
#     """
#     Convert the numpy array to torch tensor.
#     :param array:
#     :return:
#     """
#     for i in range(len(array)):
#         array[i] = torch.from_numpy(array[i])


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
    return ((raw_con + 1) * max_green / 2).astype(np.int32)


def autoregressive_act(decoder, obs_rep, batch_size, agent_num, action_dim, last_action_dis, last_action_con,
                       agent_to_update,
                       device):
    print(agent_to_update)
    hybrid_action = torch.zeros((batch_size, agent_num, action_dim + 2), dtype=torch.float32, device=device)
    hybrid_action[:, 0, 0] = 1
    for i in range(agent_num):
        with torch.no_grad():

            logits, means, stds = decoder(hybrid_action, obs_rep)

            logit = logits[:, i]
            mean = means[:, i]
            std = stds[:, i]
            act_dis_ori = last_action_dis[:, i]
            act_con_ori = last_action_con[:, i]
            agent_to_update_ = agent_to_update[:, i].bool()

            dist_dis = Categorical(logits=logit)  # the discrete distribution of all the agents
            act_dis_sam = dist_dis.sample()
            act_dis_ = torch.where(agent_to_update_, act_dis_sam, act_dis_ori)

            mean_ = torch.gather(mean, 1, act_dis_.unsqueeze(-1)).squeeze()
            std_ = torch.gather(std, 1, act_dis_.unsqueeze(-1)).squeeze()
            dist_con = Normal(mean_, std_)
            act_con_ = torch.where(agent_to_update_, torch.tanh(dist_con.sample()), act_con_ori.float())

            act_logp_dis = dist_dis.log_prob(act_dis_)
            act_logp_con = dist_con.log_prob(act_con_)

            # if agent_to_update[:, i]:
            #     dist_dis = Categorical(logits=logit)
            #     act_dis = dist_dis.sample()
            #
            #     dist_con = Normal(mean[act_dis], std[act_dis])
            #     act_con = dist_con.sample()
            #
            # else:
            #     dist_dis = Categorical(logits=logit)
            #     act_dis = action_dis[:, i]
            #
            #     dist_con = Normal(mean[action_dis], std[action_dis])
            #     act_con = action_con[:, i]
            #
            # act_logp_dis = dist_dis.log_prob(act_dis)
            # act_logp_con = dist_con.log_prob(act_con)

            if i + 1 < agent_num:
                hybrid_action[:, i + 1, 1:-1].copy_(F.one_hot(act_dis_, num_classes=action_dim).float())
                hybrid_action[:, i + 1, -1] = act_con_

        if i == 0:
            output_act_dis = act_dis_.unsqueeze(0)
            output_act_con = act_con_.unsqueeze(0)
            output_logp_dis = act_logp_dis.unsqueeze(0)
            output_logp_con = act_logp_con.unsqueeze(0)
        else:
            output_act_dis = torch.cat((output_act_dis, act_dis_.unsqueeze(0)), dim=0)
            output_act_con = torch.cat((output_act_con, act_con_.unsqueeze(0)), dim=0)
            output_logp_dis = torch.cat((output_logp_dis, act_logp_dis.unsqueeze(0)), dim=0)
            output_logp_con = torch.cat((output_logp_con, act_logp_con.unsqueeze(0)), dim=0)
    return torch.transpose(output_act_dis, 0, 1), torch.transpose(output_logp_dis, 0, 1), torch.transpose(
        output_act_con, 0, 1), torch.transpose(output_logp_con, 0, 1)


def parallel_act(decoder, obs_rep, batch_size, agent_num, action_dim, action_dis, action_con, last_action_dis,
                 last_action_con, agent_to_update, device):
    hybrid_action = torch.zeros((batch_size, agent_num, action_dim + 2), device=device)
    hybrid_action[:, 0, 0] = 1
    hybrid_action[:, 1:, 1:-1].copy_(F.one_hot(action_dis, num_classes=action_dim)[:, :-1, :])
    hybrid_action[:, 1:, -1] = action_con[:, :-1]
    logits, means, stds = decoder(hybrid_action, obs_rep)

    dist_dis = Categorical(logits=logits)
    act_dis_sam = dist_dis.sample()
    act_dis_ori = last_action_dis
    act_dis_ = torch.where(agent_to_update.bool(), act_dis_sam, act_dis_ori)
    act_logp_dis = dist_dis.log_prob(act_dis_)
    entropy_dis = dist_dis.entropy()

    means = means.gather(-1, act_dis_.unsqueeze(-1)).squeeze()
    stds = stds.gather(-1, act_dis_.unsqueeze(-1)).squeeze()
    dist_con = Normal(means, stds)
    act_con_ = torch.where(agent_to_update.bool(), torch.tanh(dist_con.sample()), last_action_con.float())
    act_logp_con = dist_con.log_prob(act_con_)
    entropy_con = dist_con.entropy()

    return act_logp_dis, entropy_dis, act_logp_con, entropy_con


def retrieve_eval_results(loc='data/eval_results.pkl'):
    # todo 文件写入错误
    tree = ET.parse(loc)
    root = tree.getroot()

    tripinfos = {'ids': [], 'depart': [], 'departLane': [], 'departPos': []}

    for tripinfo in root.findall('tripinfo'):
        ids = tripinfo.get('ids')
        depart = float(tripinfo.get('depart'))
        departLane = tripinfo.get('departLane')
        departPos = float(tripinfo.get('departPos'))

        tripinfos['ids'].append(ids)
        tripinfos['depart'].append(depart)
        tripinfos['departLane'].append(departLane)
        tripinfos['departPos'].append(departPos)

    return tripinfos


def autoregressive_act_v2(decoder, obs_rep, batch_size, agent_num, action_dim, act_dis_infer, act_con_infer,
                          agent_to_update, device):
    hybrid_action = torch.zeros((batch_size, agent_num, action_dim + 2), dtype=torch.float32, device=device)
    hybrid_action[:, 0, 0] = 1
    for i in range(agent_num):

        # For agent_i in the batch, there is at least one to update
        if agent_to_update[:, i].sum() > 0:
            with torch.no_grad():
                logits, means, stds = decoder(hybrid_action, obs_rep)
                logit = logits[:, i]
                mean = means[:, i]
                std = stds[:, i]
                agent_to_update_ = agent_to_update[:, i].bool()

                dist_dis = Categorical(logits=logit)  # Batch discrete distributions
                act_dis_ = torch.where(agent_to_update_, dist_dis.sample(), act_dis_infer[:, i])

                mean_ = torch.gather(mean, 1, act_dis_.unsqueeze(-1)).squeeze()
                std_ = torch.gather(std, 1, act_dis_.unsqueeze(-1)).squeeze()
                dist_con = Normal(mean_, std_)
                act_con_ = torch.where(agent_to_update_, torch.tanh(dist_con.sample()), act_con_infer[:, i])

                act_logp_dis = dist_dis.log_prob(act_dis_)
                act_logp_con = dist_con.log_prob(act_con_)

        # For agent_i in the batch, there is no one need to update
        else:
            act_dis_ = act_dis_infer[:, i]
            act_con_ = act_con_infer[:, i]
            # Padding
            act_logp_dis = torch.zeros_like(act_dis_, dtype=torch.float32, device=device)
            act_logp_con = torch.zeros_like(act_con_, dtype=torch.float32, device=device)

        if i + 1 < agent_num:
            hybrid_action[:, i + 1, 1:-1].copy_(F.one_hot(act_dis_, num_classes=action_dim).float())
            hybrid_action[:, i + 1, -1] = act_con_

        if i == 0:
            output_act_dis = act_dis_.unsqueeze(0)
            output_act_con = act_con_.unsqueeze(0)
            output_logp_dis = act_logp_dis.unsqueeze(0)
            output_logp_con = act_logp_con.unsqueeze(0)
        else:
            output_act_dis = torch.cat((output_act_dis, act_dis_.unsqueeze(0)), dim=0)
            output_act_con = torch.cat((output_act_con, act_con_.unsqueeze(0)), dim=0)
            output_logp_dis = torch.cat((output_logp_dis, act_logp_dis.unsqueeze(0)), dim=0)
            output_logp_con = torch.cat((output_logp_con, act_logp_con.unsqueeze(0)), dim=0)
    return torch.transpose(output_act_dis, 0, 1), torch.transpose(output_logp_dis, 0, 1), torch.transpose(
        output_act_con, 0, 1), torch.transpose(output_logp_con, 0, 1)


def parallel_act_v2(decoder, obs_rep, batch_size, agent_num, action_dim, act_dis_exe, act_con_exe, act_dis_infer,
                    act_con_infer, agent_to_update, device):
    hybrid_action = torch.zeros((batch_size, agent_num, action_dim + 2), device=device)
    action_dis = torch.where(agent_to_update.bool(), act_dis_exe, act_dis_infer)
    action_con = torch.where(agent_to_update.bool(), act_con_exe, act_con_infer)
    hybrid_action[:, 0, 0] = 1
    hybrid_action[:, 1:, 1:-1].copy_(F.one_hot(action_dis, num_classes=action_dim)[:, :-1, :])
    hybrid_action[:, 1:, -1] = action_con[:, :-1]
    logits, means, stds = decoder(hybrid_action, obs_rep)

    dist_dis = Categorical(logits=logits)
    act_dis_ = torch.where(agent_to_update.bool(), dist_dis.sample(), act_dis_infer)
    act_logp_dis = dist_dis.log_prob(act_dis_)[agent_to_update == 1]
    entropy_dis = dist_dis.entropy()  # todo fix

    means = means.gather(-1, act_dis_.unsqueeze(-1)).squeeze()
    stds = stds.gather(-1, act_dis_.unsqueeze(-1)).squeeze()
    dist_con = Normal(means, stds)
    act_con_ = torch.where(agent_to_update.bool(), torch.tanh(dist_con.sample()), act_con_infer)
    act_logp_con = dist_con.log_prob(act_con_)[agent_to_update == 1]
    entropy_con = dist_con.entropy()

    return act_logp_dis, entropy_dis, act_logp_con, entropy_con
