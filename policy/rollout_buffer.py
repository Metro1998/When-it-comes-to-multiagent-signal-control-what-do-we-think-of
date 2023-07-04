"""
reference:
https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py

Author:Metro
date:2023.1.10
"""
import sys

import numpy as np
from typing import List
from utils.util import *


class PPOBuffer:
    """
    A buffer for storing trajectories generated by (H)PPO agents interacting with
    parallel sub-environments, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of obs-action pairs.
    """

    def __init__(self, num_steps, num_envs, num_agents, obs_dim, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((num_steps, num_envs, num_agents * obs_dim), dtype=np.float32)
        self.rew_buf = np.zeros((num_steps, num_envs, num_agents), dtype=np.float32)
        self.val_buf = np.zeros((num_steps, num_envs, num_agents), dtype=np.float32)
        self.adv_buf = np.zeros((num_steps, num_envs, num_agents), dtype=np.float32)
        self.ret_buf = np.zeros((num_steps, num_envs, num_agents), dtype=np.float32)
        self.act_con_buf = np.zeros((num_steps, num_envs, num_agents), dtype=np.float32)
        self.act_dis_buf = np.zeros((num_steps, num_envs, num_agents), dtype=np.int64)
        self.logp_con_buf = np.zeros((num_steps, num_envs, num_agents), dtype=np.float32)
        self.logp_dis_buf = np.zeros((num_steps, num_envs, num_agents), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, num_steps
        self.end_idx = np.array([0])

    def finish_path(self, critical_step_idx, last_val=0):
        """
        Call this at the end of a trajectory.
        首先我们会等一个truncated 或者 terminated 这个表示我们已经采样到了足够数量的样本/该episode已经结束，但是我们还会让SUMOEnv继续运行直到所有
        的智能体都能够make a decision, 这样做的目的是使得truncated那一步能计算到其对应的reward
        同时我们需要对reward进行处理，因为该状态的reward其实不是两个相邻之间的step之间的，而是该step到下一个make a decision 的step；
        所以我们需要将reward往后cumsum， 至于end_idx我们会在仿真的时候进行记录，处理完reward之后我们实际上已经拿到了所有（可行的）state 所对应的reward
        紧接着我会在make a decision 的step上 进行gae的计算.
        在计算每个state所对应的advantage之后，我们会计算lambda return using G_lambda =  GAE + V(s)
        :param critical_step_idx: if it is time to decide and episode_simple > 0, then append episode_sample to critical_step_idx.
        :return:
        """

        path_slice = slice(self.path_start_idx, self.ptr)

        end_idx = sys.maxsize  # The end_idx of the available fixed trajectory segment.
        reward = self.rew_buf[path_slice].transpose(1, 2, 0)  # [num_envs, num_agents, num_steps] [s_0, ... , s_n-1]
        value = np.append(self.val_buf[path_slice].transpose(1, 2, 0), last_val)  # [num_envs, num_agents, num_steps + 1]  [s_0, ... , s_n]
        reward_ = np.zeros_like(reward)  # For the rebuilt reward.
        cum_end_idx = np.zeros_like(reward)

        # reward[i][j]              np.array([2, 9, 1, -5, 13, 12, 7, 3])
        # critical_step_idx[i][j]                      [3,              8]
        # cum_end_idx[i][j]         np.array([3, 3, 3, 8, 8, 8, 8, 8])
        # idx                                [0, 1, 2, 3, 4, 5, 6, 7]
        # reward_[i][j]             np.array([12, 10, 1, 30, 25, 22, 10, 3])

        num_envs = len(critical_step_idx)
        num_agents = len(critical_step_idx[0])

        # Calculate the cum_end_idx.
        for i in range(num_envs):
            for j in range(num_agents):
                end_idx = min(end_idx, critical_step_idx[i][j][-1])  # 最短那段轨迹，其critical_step的idx，注意是idx 也可以看成可用轨迹的长度 不包括最后一个critical_step
                arr = np.array(critical_step_idx[i][j])
                arr_diff = np.diff(np.insert(arr, 0, 0))
                cum_end_idx[i][j] = np.repeat(arr, arr_diff)

        # Transverse the time step.
        for i in range(end_idx):
            reward_ = np.cumsum(reward[:, :, i: cum_end_idx[:, :, i]], axis=2)
        # Finally, we will utilize the left part to calculate the advantage.
        reward_ = reward_[:, :, :end_idx]

        for i in range(num_envs):
            for j in range(num_agents):
                arr = np.array(critical_step_idx[i][j])
                mask_rew = (arr < end_idx)
                mask_val = (arr <= end_idx)
                for k in range(end_idx):
                    rew_tra = np.insert(reward_[i][j][mask_rew & (arr > k)], 0, reward_[i][j][k])
                    val_tra = np.insert(value[i][j][mask_val & (arr > k)], 0, value[i][j][k])
                    delta = rew_tra + self.gamma * val_tra[1:] - val_tra[:-1]
                    self.adv_buf[i][j][self.path_start_idx + k] = discount_cumsum(delta, self.gamma * self.lam)[0]
                    self.ret_buf[i][j][self.path_start_idx + k] = self.adv_buf[i][j][self.path_start_idx + k] + value[i][j][self.path_start_idx + k]

        self.path_start_idx = end_idx  # 因为我们的计算都是到end_idx(不包含)为止

    def store_trajectories(self, obs, rew, act_con, act_dis, logp_con, logp_dis):
        """
`       Append one timestep of agent-environment interaction to the buffer.
        ### Inputs are batch of num_envs * num_agents ###
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.rew_buf[self.ptr] = rew
        self.act_con_buf[self.ptr] = act_con
        self.act_dis_buf[self.ptr] = act_dis
        self.logp_con_buf[self.ptr] = logp_con
        self.logp_dis_buf[self.ptr] = logp_dis
        self.ptr += 1

    def get(self):
        """
        Call this at the end of a rollout round to retrieve the full information.
        :return:
        """
        assert self.ptr == self.max_size

        obs = self.obs_buf[:self.ptr]
        rew = self.rew_buf[:self.ptr]
        act_con = self.act_con_buf[:self.ptr]
        act_dis = self.act_dis_buf[:self.ptr]
        logp_con = self.logp_con_buf[:self.ptr]
        logp_dis = self.logp_dis_buf[:self.ptr]

        return obs, rew, act_con, act_dis, logp_con, logp_dis, self.end_idx

    def clear(self):
        self.ptr = 0
        self.end_idx = np.array([0])
