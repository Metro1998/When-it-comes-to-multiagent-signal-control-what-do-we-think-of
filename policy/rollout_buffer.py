"""
reference:
https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py

Author:Metro
date:2023.1.10
"""
import sys

import numpy as np
from typing import List

import torch

from utils.util import *


class PPOBuffer:
    """
    A buffer for storing trajectories generated by (H)PPO agents interacting with
    parallel sub-environments, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of obs-action pairs.
    """

    def __init__(self, num_steps, num_envs, num_agents, obs_dim, len_his, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((num_steps, num_envs, num_agents, len_his, obs_dim), dtype=np.float32)
        self.rew_buf = np.zeros((num_steps, num_envs, num_agents), dtype=np.float32)
        self.val_buf = np.zeros((num_steps, num_envs, num_agents), dtype=np.float32)
        self.adv_buf = np.zeros((num_steps, num_envs, num_agents), dtype=np.float32)
        self.ret_buf = np.zeros((num_steps, num_envs, num_agents), dtype=np.float32)
        self.act_con_buf = np.zeros((num_steps, num_envs, num_agents), dtype=np.float32)
        self.act_dis_buf = np.zeros((num_steps, num_envs, num_agents), dtype=np.int64)
        self.act_con_infer_buf = np.zeros((num_steps, num_envs, num_agents), dtype=np.float32)
        self.act_dis_infer_buf = np.zeros((num_steps, num_envs, num_agents), dtype=np.int64)
        self.logp_con_buf = np.zeros((num_steps, num_envs, num_agents), dtype=np.float32)
        self.logp_dis_buf = np.zeros((num_steps, num_envs, num_agents), dtype=np.float32)
        self.agent_to_update = np.zeros((num_steps, num_envs, num_agents), dtype=np.int64)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, num_steps
        self.end_idx = np.array([0])

        self.num_steps = num_steps
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.len_his = len_his

    def finish_path(self, critical_step_idx):
        """
        Call this at the end of a trajectory.
        首先我们会等一个truncated 或者 terminated 这个表示我们已经采样到了足够数量的样本/该episode已经结束，但是我们还会让SUMOEnv继续运行直到所有
        的智能体都能够make a decision, 这样做的目的是使得truncated那一步能计算到其对应的reward
        同时我们需要对reward进行处理，因为该状态的reward其实不是两个相邻之间的step之间的，而是该step到下一个make a decision 的step；
        所以我们需要将reward往后cumsum， 至于end_idx我们会在仿真的时候进行记录，处理完reward之后我们实际上已经拿到了所有（可行的）state 所对应的reward
        紧接着我会在make a decision 的step上 进行gae的计算.
        在计算每个state所对应的advantage之后，我们会计算lambda return using G_lambda =  GAE + V(s)
        :param last_val:
        :param critical_step_idx: if it is time to decide and episode_simple > 0, then append episode_sample to critical_step_idx.
        :return:
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        end_idx = sys.maxsize
        reward = self.rew_buf[path_slice].transpose(1, 2, 0)  # [num_envs, num_agents, num_steps(s_0, ... , s_n-1)]
        reward_ = np.zeros_like(reward)  # reorganized reward.
        cum_end_idx = np.zeros_like(reward,
                                    dtype=np.int64)  # The cum_end_idx of the available fixed trajectory segment.

        # Calculate the cum_end_idx.
        for i in range(self.num_envs):
            for j in range(self.num_agents):
                end_idx = min(end_idx, critical_step_idx[i][j][
                    -1])  # 最短那段轨迹，其critical_step的idx，注意是idx 也可以看成可用轨迹的长度 不包括最后一个critical_step
                arr = np.array(critical_step_idx[i][j])
                arr_diff = np.diff(np.insert(arr, 0, 0))
                tmp = np.repeat(arr, arr_diff)
                cum_end_idx[i][j][:len(tmp)] = tmp

        for i in range(self.num_envs):
            for j in range(self.num_agents):
                for k in range(end_idx):
                    reward_[i][j][k] = np.sum(reward[i][j][k: cum_end_idx[i][j][k]])
        reward_ = reward_[:, :, :end_idx]
        value_ = self.val_buf[path_slice].transpose(1, 2, 0)

        for i in range(self.num_envs):
            for j in range(self.num_agents):
                arr = np.array(critical_step_idx[i][j])
                for k in range(end_idx):
                    rew_idx = arr[(arr > k) & (arr < end_idx)]
                    val_idx = arr[arr > k][:len(rew_idx) + 1]
                    rew_tra = np.insert(reward_[i][j][rew_idx], 0, reward_[i][j][k])
                    val_tra = np.insert(value_[i][j][val_idx], 0, value_[i][j][k])
                    delta = rew_tra + self.gamma * val_tra[1:] - val_tra[:-1]
                    self.adv_buf[self.path_start_idx + k][i][j] = discount_cumsum(delta, self.gamma * self.lam)[0]
                    self.ret_buf[self.path_start_idx + k][i][j] = self.adv_buf[self.path_start_idx + k][i][j] + \
                                                                  value_[i][j][k]
        print('SUCCESS!')
        self.path_start_idx += end_idx
        self.ptr = self.path_start_idx

    def store_trajectories(self, obs, act_dis_infer, act_con_infer, agent_to_update, act_dis, act_con, logp_dis, logp_con, val, rew):
        """
`       Append one timestep of agent-environment interaction to the buffer.
        ### Inputs are batch of num_envs * num_agents ###
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_dis_infer_buf[self.ptr] = act_dis_infer
        self.act_con_infer_buf[self.ptr] = act_con_infer
        self.agent_to_update[self.ptr] = agent_to_update
        self.act_dis_buf[self.ptr] = act_dis
        self.act_con_buf[self.ptr] = act_con
        self.logp_dis_buf[self.ptr] = logp_dis
        self.logp_con_buf[self.ptr] = logp_con
        self.val_buf[self.ptr] = val
        self.rew_buf[self.ptr] = rew

        self.ptr += 1

    def get(self):
        """
        Call this at the end of a rollout round to retrieve the full information.
        :return:
        """
        data = dict(
            obs=self.obs_buf[:self.ptr].reshape(-1, self.num_agents, self.len_his, self.obs_dim),
            act_dis_infer=self.act_dis_infer_buf[:self.ptr].reshape(-1, self.num_agents),
            act_con_infer=self.act_con_infer_buf[:self.ptr].reshape(-1, self.num_agents),
            act_dis=self.act_dis_buf[:self.ptr].reshape(-1, self.num_agents),
            act_con=self.act_con_buf[:self.ptr].reshape(-1, self.num_agents),
            logp_dis=self.logp_dis_buf[:self.ptr].reshape(-1, self.num_agents),
            logp_con=self.logp_con_buf[:self.ptr].reshape(-1, self.num_agents),
            adv=self.adv_buf[:self.ptr].reshape(-1, self.num_agents),
            ret=self.ret_buf[:self.ptr].reshape(-1, self.num_agents),
            agent=self.agent_to_update[:self.ptr].reshape(-1, self.num_agents)
        )

        self.path_start_idx, self.ptr = 0, 0

        return {
            k: torch.as_tensor(v, dtype=torch.int64, device=torch.device('cuda')) if k in ['act_dis', 'act_dis_infer', 'agent']
            else torch.as_tensor(v, dtype=torch.float32, device=torch.device('cuda')) for k, v in data.items()
        }
