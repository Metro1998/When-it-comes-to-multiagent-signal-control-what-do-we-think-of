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


def remap(time_remaining):
    """
    Remap the remaining time to its original range.
    :param time_remaining:
    :param max_green:
    :return:
    """
    return torch.atanh(2 * (time_remaining - 10) / (30 - 10) - 1)


def map2real(raw_con):
    """
    Map the raw continuous parameter to the range of [0, max_green]
    :param raw_con:
    :param min_green:
    :param max_green:
    :return:
    """
    return 10 + (raw_con + 1) * (30 - 10) / 2


def run_stat(stat, stat_ptrs):
    """
    Calculate the mean and std of the observation.
    :param stat:
    :param stat_ptrs:
    :return:
    """
    for i in range(len(stat_ptrs)):
        if i == 0:
            stats = stat[i][:stat_ptrs[i]]
        else:
            stats = np.append(stats, stat[i][:stat_ptrs[i]], axis=0)
    mean = np.mean(stats, axis=(0, 2, 3))
    std = np.std(stats, axis=(0, 2, 3))
    return mean, std
