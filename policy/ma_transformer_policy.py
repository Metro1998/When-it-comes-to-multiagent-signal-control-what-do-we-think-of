"""
reference: https://github.com/PKU-MARL/Multi-Agent-Transformer/blob/main/mat/algorithms/mat/algorithm/transformer_policy.py
"""
import torch
import torch.nn as nn
from utils.util import *
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


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
        self.epochs = args.epochs

        self.optimizer_actor_con = torch.optim.Adam([
            {'params': policy.decoder_con.parameters(), 'lr': args.lr_actor_con},
            {'params': policy.decoder_con.log_std, 'lr': args.lr_std}])
        self.optimizer_actor_dis = torch.optim.Adam([
            {'params': policy.decoder_dis.parameters(), 'lr': args.lr_actor_dis}])
        self.lr_scheduler_actor_con = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_actor_con,
                                                                             gamma=args.lr_decay_rate)
        self.lr_scheduler_actor_dis = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_actor_dis,
                                                                             gamma=args.lr_decay_rate)

        self.optimizer_critic = torch.optim.Adam([{'params': policy.encoder.parameters(), 'lr': args.lr_critic}])
        self.lr_scheduler_critic = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_critic,
                                                                          gamma=args.lr_decay_rate)
        self.loss_func = nn.SmoothL1Loss(reduction='mean')

    def update(self):

        obs, rew, act_con, act_dis, logp_con, logp_dis, end_idx = self.buffer.get()

        update_dis_actor, update_con_actor = 1, 1

        for i in self.epochs:
            # Recompute values at the beginning of each epoch
            advantage, return_ = self.recompute(obs, rew, end_idx)

            sampler_critic = list(BatchSampler(
                sampler=SubsetRandomSampler(obs.shape[0]),
                batch_size=self.batch_size,
                drop_last=True))
            for indices in sampler_critic:
                ret_batch = torch.as_tensor(return_[indices], dtype=torch.float32, device=self.device)
                adv_batch = torch.as_tensor(advantage[indices], dtype=torch.float32, device=self.device)
                obs_batch = torch.as_tensor(obs[indices], dtype=torch.float32, device=self.device)
                act_con_batch = torch.as_tensor(act_con[indices], dtype=torch.float32, device=self.device)
                act_dis_batch = torch.as_tensor(act_dis[indices], dtype=torch.int64, device=self.device)
                old_logp_con_batch = torch.as_tensor(logp_con[indices], dtype=torch.float32, device=self.device)
                old_logp_dis_batch = torch.as_tensor(logp_dis[indices], dtype=torch.float32, device=self.device)

                # Advantage normalization
                # In particular, this normalization happens at the minibatch level instead of the whole batch level!
                adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

                ### Update encoder ###
                self.optimizer_critic.zero_grad()
                critic_loss = self.cal_critic_loss(obs_batch, ret_batch)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.encoder.parameters(), norm_type=2,
                                               max_norm=self.max_grad_norm)
                self.optimizer_critic.step()

                ### Update decoders ###
                new_logp_dis_batch, entropy_dis, new_logp_con_batch, entropy_con = self.policy.evaluate_actions(obs_batch,
                                                                                                                act_dis_batch,
                                                                                                                act_con_batch)
                if update_dis_actor:
                    loss_pi_dis, approx_kl_dis = self.cal_actor_dis_loss(old_logp_dis_batch, new_logp_dis_batch,
                                                                         entropy_dis, adv_batch)
                    self.optimizer_actor_dis.zero_grad()
                    loss_pi_dis.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.decoder_dis.parameters(), norm_type=2, max_norm=self.max_grad_norm)
                    self.optimizer_actor_dis.step()

                if update_con_actor:
                    loss_pi_con, approx_kl_con = self.cal_actor_con_loss(act_dis_batch, old_logp_con_batch, new_logp_con_batch, adv_batch)

                    self.optimizer_actor_con.zero_grad()
                    loss_pi_con.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.decoder_con.parameters(), norm_type=2, max_norm=self.max_grad_norm)
                    self.optimizer_actor_con.step()

            if approx_kl_dis > self.target_kl_dis:
                update_dis_actor = 0
            if approx_kl_con > self.target_kl_con:
                update_con_actor = 0

    def recompute(self, observation, reward, end_idx):
        """
        Trick[0], recompute the value prediction when calculate the advantage
        Compute advantage function A(s, a) based on global V-value network with GAE, where a represents joint action

        :param observation:
        :param reward:
        :param end_idx:
        :return:
        """
        # (num_steps, num_agents, obs_dim) --> (num_steps)
        value = self.policy.encoder(check(observation)).squeeze().detach().numpy()
        # (num_steps, num_agents) --> (num_steps)
        reward = np.sum(reward, axis=-1)

        advantage = np.array([])
        return_ = np.array([])
        for j in range(len(end_idx) - 1):
            val = np.array(value[end_idx[j], end_idx[j + 1]] + [0])
            rew = np.array(reward[end_idx[j], end_idx[j + 1]] + [0])

            # the next two lines implement GAE-Lambda advantage calculation
            delta = rew[:-1] + self.gamma * val[1:] - val[:-1]
            advantage = np.append(advantage, discount_cumsum(delta, self.gamma * self.lam))
            # The return is repeatedly calculated actually
            return_ = np.append(return_, discount_cumsum(rew, self.gamma)[:-1])

        return advantage, return_

    def cal_critic_loss(self, obs_batch, ret_batch):
        """
        Calculate the loss of critic

        :param obs_batch:
        :param ret_batch:
        :return:
        """
        predicted_values = self.policy.get_values(obs_batch)
        critic_loss = self.loss_func(predicted_values, ret_batch)
        return critic_loss

    def cal_actor_dis_loss(self, old_logp_dis_batch, new_logp_dis_batch, entropy_dis, adv_batch):
        """

        :param old_logp_dis_batch:
        :param new_logp_dis_batch:
        :param entropy_dis:
        :param adv_batch:
        :return:
        """
        imp_weights = torch.exp(new_logp_dis_batch - old_logp_dis_batch)
        surr1 = imp_weights * adv_batch
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_batch
        loss_pi_dis = - (torch.min(surr1, surr2) + self.entropy_coef_dis * entropy_dis).mean()

        with torch.no_grad():
            # Trick, calculate approx_kl http://joschu.net/blog/kl-approx.html
            approx_kl_dis = ((imp_weights - 1) - (new_logp_dis_batch - old_logp_dis_batch)).mean()

        return loss_pi_dis, approx_kl_dis

    def cal_actor_con_loss(self, act_dis_batch, old_logp_con_batch, new_logp_con_batch, adv_batch):
        """

        :param act_dis_batch:
        :param old_logp_con_batch:
        :param new_logp_con_batch:
        :param adv_batch:
        :return:
        """
        old_logp_con_batch = old_logp_con_batch.gather(1, act_dis_batch.view(-1, 1)).squeeze()
        new_logp_con_batch = new_logp_con_batch.gather(1, act_dis_batch.view(-1, 1)).squeeze()

        imp_weights = torch.exp(new_logp_con_batch - old_logp_con_batch)
        surr1 = imp_weights * adv_batch
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_batch

        # Andrychowicz, et al. (2021) overall find no evidence that the entropy term improves performance on
        # continuous control environments.
        loss_pi_con = - torch.min(surr1, surr2).mean()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            approx_kl_con = ((imp_weights - 1) - (new_logp_con_batch - old_logp_con_batch)).mean()

        return loss_pi_con, approx_kl_con

    def get_action(self, obs):
        """

        :param obs: (torch.Tensor)
        :return:
        """
        act_con, act_dis, logp_con, logp_dis, _ = self.policy.act(obs)
        return act_con, act_dis, logp_con, logp_dis

    def save(self, save_dir, episode):
        torch.save(self.policy.state_dict(), str(save_dir) + "/MAT_" + str(episode) + ".pt")

    def load(self, model_dir):
        self.policy.load_state_dict(torch.load(model_dir))
