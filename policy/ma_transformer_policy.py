"""
reference: https://github.com/PKU-MARL/Multi-Agent-Transformer/blob/main/mat/algorithms/mat/algorithm/transformer_policy.py
"""
import torch.nn as nn
from utils.util import *


class PPOTrainer:

    def __init__(self, args, buffer, policy):
        """
        Trainer class for MAT(in hybrid action space) to update policies,
        actually it's a standard trainer based PPO.
        :param args:
        :param buffer:
        :param policy:
        """

        self.buffer = buffer
        self.policy = policy

        self.random_seed = args.random_seed
        self.agents_num = args.agents_num
        self.device = args.device
        self.clip_ratio = args.clip_ratio
        self.ppo_epoch = args.ppo_epoch
        self.batch_size = args.batch_size
        self.entropy_coef_dis = args.entropy_coef_dis
        self.entropy_coef_con = args.entropy_coef_con
        self.max_grad_norm = args.max_grad_norm
        self.target_kl_dis = args.target_kl_dis
        self.target_kl_con = args.target_kl_con
        self.gamma = args.gamma
        self.lam = args.lam

        if args.action_type == 'Continuous':
            self.optimizer_actor_con = torch.optim.Adam([
                {'params': policy.decoder_con.parameters(), 'lr': args.lr_actor_con},
                {'params': policy.decoder_con.log_std, 'lr': args.lr_std}])
            self.lr_scheduler_actor_con = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_actor_con, gamma=args.lr_decay_rate)
        elif args.action_type == 'Discrete':
            self.optimizer_actor_dis = torch.optim.Adam([
                {'params': policy.decoder_dis.parameters(), 'lr': args.lr_actor_dis}])
            self.lr_scheduler_actor_dis = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_actor_dis, gamma=args.lr_decay_rate)
        else:  # Hybrid
            self.optimizer_actor_con = torch.optim.Adam([
                {'params': policy.decoder_con.parameters(), 'lr': args.lr_actor_con},
                {'params': policy.decoder_con.log_std, 'lr': args.lr_std}])
            self.optimizer_actor_dis = torch.optim.Adam([
                {'params': policy.decoder_dis.parameters(), 'lr': args.lr_actor_dis}])
            self.lr_scheduler_actor_con = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_actor_con, gamma=args.lr_decay_rate)
            self.lr_scheduler_actor_dis = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_actor_dis, gamma=args.lr_decay_rate)

        self.optimizer_critic = torch.optim.Adam([{'params': policy.encoder.parameters(), 'lr': args.lr_critic}])
        self.lr_scheduler_critic = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_critic, gamma=args.lr_decay_rate)
        self.loss_func = nn.SmoothL1Loss(reduction='mean')

    def recompute(self, obs_batch, rew_batch):
        """
        Compute advantage function A(s, a) based on global V-value network with GAE, where a represent joint action
        :param obs_batch:
        :param rew_batch:
        :return:
        """

        # (num_envs, num_steps, num_agents, obs_dim) --> (num_envs, num_steps)
        val_batch = self.policy.encoder(check(obs_batch)).squeeze().detach().numpy()
        # (num_envs, num_steps, num_agents) --> (num_envs, num_steps)
        rew_batch = np.sum(rew_batch, axis=-1)

        advantage, reward_to_go = np.array([]), np.array([])
        for i in val_batch.shape[0]:  # num_envs
            val = val_batch[i]
            rew = rew_batch[i]

            # the next two lines implement GAE-Lambda advantage calculation
            delta = rew[:-1] + self.gamma * val[1:] - val[:-1]
            advantage = np.append(advantage, discount_cumsum(delta, self.gamma * self.lam))

        return advantage

    def cal_critic_loss(self, obs_batch, ret_batch):
        """
        Calculate the loss of critic
        :param obs_batch: (np.ndarray) (batch_size, agent_num, obs_dim)
        :param ret_batch: (torch.Tensor) (batch_size, 1)
        :return:
        """
        predicted_values = self.policy.get_values(obs_batch)
        critic_loss = self.loss_func(predicted_values, ret_batch)
        return critic_loss

    def save(self, save_dir, episode):
        torch.save(self.policy.state_dict(), str(save_dir) + "/MAT_" + str(episode) + ".pt")

    def load(self, model_dir):
        self.policy.load_state_dict(torch.load(model_dir))


