import scipy.signal
import torch
import numpy as np
import torch.optim as optim


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


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class mapping:

    def __init__(self, min_green, max_green):
        self.min_green = min_green
        self.max_green = max_green

    def norm(self, time_left):
        """
        Remap the remaining time to [-1, 1] linearly.
        :param time_left:
        :return:
        """
        return 2 * time_left / self.max_green - 1

    def anorm(self, normed_time_left):
        """
        Remap the remaining time to [-1, 1] linearly.
        :param normed_time_left
        :return:
        """
        return (normed_time_left + 1) * self.max_green / 2

    def map2real(self, raw_con):
        """
        Map the raw continuous parameter to the range of [0, max_green]
        :param raw_con:
        :return:
        """
        return self.min_green + (raw_con + 1) * (self.max_green - self.min_green) / 2

    def remap(self, time_left):
        """
        Remap the remaining time to its original range.
        :param time_left:
        :return:
        """
        return torch.atanh(2 * (time_left - self.min_green) / (self.max_green - self.min_green) - 1)
