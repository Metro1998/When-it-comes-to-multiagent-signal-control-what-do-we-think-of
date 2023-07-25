"""
reference: https://github.com/PKU-MARL/Multi-Agent-Transformer/blob/main/mat/algorithms/mat/algorithm/transformer_policy.py
"""
import torch
import torch.nn as nn
from typing import List

from utils.util import *
from policy.ma_transformer import MultiAgentTransformer
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from policy.rollout_buffer import PPOBuffer
from torch.utils.tensorboard import SummaryWriter


class PPOTrainer:

    def __init__(self, writer, args):
        """

        :param args:
        """

        self.policy_gpu = MultiAgentTransformer(args.obs_dim, args.action_dim, args.embd_dim, args.agent_num, args.block_num, args.head_num, args.std_clip, device='cuda:0')
        self.policy_cpu = MultiAgentTransformer(args.obs_dim, args.action_dim, args.embd_dim, args.agent_num, args.block_num, args.head_num, args.std_clip, device='cpu')
        self.copy_parameter()
        self.buffer = PPOBuffer(3600, args.env_num, args.agent_num, args.obs_dim, args.history_len, args.gamma, args.lam)
        self.random_seed = args.random_seed
        self.agent_num = args.agent_num
        self.clip_ratio = args.clip_ratio
        self.batch_size = args.batch_size
        self.entropy_coef_dis = args.entropy_coef_dis
        self.entropy_coef_con = args.entropy_coef_con
        self.max_grad_norm = args.max_grad_norm  # gradient clip value, is set to be 0.5 in MAT
        self.target_kl_dis = args.target_kl_dis
        self.target_kl_con = args.target_kl_con
        self.gamma = args.gamma
        self.lam = args.lam
        self.epochs = args.epochs
        self.writer = writer
        self.global_step = 0

        # args.adam_eps is set to be 1e-5, recommended by "The 37 Implementation Details of Proximal Policy Optimization"
        # self.parameters_con = [
        #     {'params': self.policy_gpu.decoder.action_embedding_con.parameters()},
        #     {'params': self.policy_gpu.decoder.blocks_con.parameters()},
        #     {'params': self.policy_gpu.decoder.head_con.parameters()},
        #     {'params': self.policy_gpu.decoder.log_std}
        # ]
        self.optimizer_actor = torch.optim.Adam([
            {'params': self.policy_gpu.decoder.action_embedding_con.parameters(), 'lr': args.lr_actor_con, 'eps': args.adam_eps},
            {'params': self.policy_gpu.decoder.blocks_con.parameters(), 'lr': args.lr_actor_con, 'eps': args.adam_eps},
            {'params': self.policy_gpu.decoder.head_con.parameters(), 'lr': args.lr_actor_con, 'eps': args.adam_eps},
            {'params': self.policy_gpu.decoder.log_std, 'lr': args.lr_std, 'eps': args.adam_eps},
            {'params': self.policy_gpu.decoder.action_embedding_dis.parameters(), 'lr': args.lr_actor_dis, 'eps': args.adam_eps},
            {'params': self.policy_gpu.decoder.blocks_dis.parameters(), 'lr': args.lr_actor_dis, 'eps': args.adam_eps},
            {'params': self.policy_gpu.decoder.head_dis.parameters(), 'lr': args.lr_actor_dis, 'eps': args.adam_eps},
        ])
        # self.parameters_dis = [
        #     {'params': self.policy_gpu.decoder.action_embedding_dis.parameters()},
        #     {'params': self.policy_gpu.decoder.blocks_dis.parameters()},
        #     {'params': self.policy_gpu.decoder.head_dis.parameters()}
        # ]
        # self.optimizer_actor_dis = torch.optim.Adam(self.parameters_dis, lr=args.lr_actor_con, eps=args.adam_eps)

        self.optimizer_critic = torch.optim.Adam(self.policy_gpu.encoder.parameters(), lr=args.lr_critic, eps=args.adam_eps)

        # self.lr_scheduler_actor_con = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_actor, gamma=args.lr_decay_rate)
        # self.lr_scheduler_actor_dis = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_actor_dis, gamma=args.lr_decay_rate)
        self.lr_scheduler_critic = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_critic, gamma=args.lr_decay_rate)
        self.loss_func = nn.SmoothL1Loss(reduction='mean')

    def update(self):

        sample_dic = self.buffer.get()
        total_len = sample_dic['obs'].size()[0]

        for _ in range(self.epochs):
            # Recompute values at the beginning of each epoch
            # advantage, return_ = self.recompute(obs, rew, end_idx)

            sampler = list(BatchSampler(
                sampler=SubsetRandomSampler(range(total_len)),
                batch_size=self.batch_size,
                drop_last=True))
            for indices in sampler:
                obs_batch = sample_dic['obs'][indices]
                agent_batch = sample_dic['agent'][indices]
                act_dis_batch = sample_dic['act_dis'][indices]
                act_con_batch = sample_dic['act_con'][indices]
                old_logp_dis_batch = sample_dic['logp_dis'][indices][agent_batch == 1]
                old_logp_con_batch = sample_dic['logp_con'][indices][agent_batch == 1]
                adv_batch = sample_dic['adv'][indices]
                ret_batch = sample_dic['ret'][indices]

                # Advantage normalization
                # In particular, this normalization happens at the minibatch level instead of the whole batch level!
                joint_adv_batch = torch.mean(adv_batch, dim=-1, keepdim=True)
                joint_adv_batch = (joint_adv_batch - joint_adv_batch.mean()) / (joint_adv_batch.std() + 1e-8)
                joint_adv_batch = joint_adv_batch.expand(-1, self.agent_num)[agent_batch == 1]

                ### Update encoder ###
                ## Calculate the gradient of critic ##
                predicted_values = self.policy_gpu.get_values(obs_batch)
                critic_loss = self.loss_func(predicted_values, ret_batch)
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_gpu.encoder.parameters(), norm_type=2, max_norm=self.max_grad_norm)
                self.optimizer_critic.step()

                ### Update decoders ###
                new_logp_dis_batch, entropy_dis, new_logp_con_batch, entropy_con = self.policy_gpu.evaluate_actions(
                    obs_batch, act_dis_batch, act_con_batch, agent_batch)

                self.optimizer_actor.zero_grad()

                ## Calculate the gradient of discrete actor ##
                imp_weights_dis = torch.exp(new_logp_dis_batch - old_logp_dis_batch)
                surr1_dis = imp_weights_dis * joint_adv_batch
                surr2_dis = torch.clamp(imp_weights_dis, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * joint_adv_batch
                # loss_pi_dis = - (torch.min(surr1_dis, surr2_dis) + self.entropy_coef_dis * entropy_dis).mean()
                loss_pi_dis = - torch.min(surr1_dis, surr2_dis).mean()
                with torch.no_grad():
                    # Trick, calculate approx_kl http://joschu.net/blog/kl-approx.html
                    approx_kl_dis = ((imp_weights_dis - 1) - (new_logp_dis_batch - old_logp_dis_batch)).mean()

                imp_weights_con = torch.exp(new_logp_con_batch - old_logp_con_batch)
                surr1_con = imp_weights_con * joint_adv_batch
                surr2_con = torch.clamp(imp_weights_con, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * joint_adv_batch

                # Andrychowicz, et al. (2021) overall find no evidence that the entropy term improves performance on
                # continuous control environments.
                loss_pi_con = - torch.min(surr1_con, surr2_con).mean()
                with torch.no_grad():
                    approx_kl_con = ((imp_weights_con - 1) - (new_logp_con_batch - old_logp_con_batch)).mean()
                if approx_kl_dis > self.target_kl_dis or approx_kl_con > self.target_kl_con:
                    print('Early stopping at step {} due to reaching max kl.'.format(_))
                else:
                    loss_pi = loss_pi_dis + loss_pi_con
                    loss_pi.backward()
                    # [torch.nn.utils.clip_grad_norm_(_['params'], norm_type=2, max_norm=self.max_grad_norm) for _ in self.parameters_dis]
                    self.optimizer_actor.step()

                self.writer.add_scalar('loss/critic', critic_loss, self.global_step)
                self.writer.add_scalars('loss/actor', {'discrete_head': loss_pi_dis, 'continuous_head': loss_pi_con}, self.global_step)
                self.writer.add_scalars('entropy', {'discrete_head': entropy_dis.mean(), 'continuous_head': entropy_con.mean()}, self.global_step)
                self.writer.add_scalars('approx_kl', {'discrete_head': approx_kl_dis, 'continuous_head': approx_kl_con}, self.global_step)
                self.writer.flush()
                self.global_step += 1

        # self.lr_scheduler_actor_con.step()
        # self.lr_scheduler_actor_dis.step()
        # self.lr_scheduler_critic.step()
        self.copy_parameter()

    # def recompute(self, observation, reward, end_idx):
    #     """
    #     Trick[0], recompute the value prediction when calculate the advantage
    #     Compute advantage function A(s, a) based on global V-value network with GAE, where a represents joint action
    #
    #     :param observation:
    #     :param reward:
    #     :param end_idx:
    #     :return:
    #     """
    #     # TODO: policy_gpu
    #     # (num_steps, num_agents, obs_dim) --> (num_steps)
    #     value = self.policy.encoder(check(observation)).squeeze().detach().numpy()
    #     # (num_steps, num_agents) --> (num_steps)
    #     reward = np.sum(reward, axis=-1)
    #
    #     advantage = np.array([])
    #     return_ = np.array([])
    #     for j in range(len(end_idx) - 1):
    #         val = np.array(value[end_idx[j], end_idx[j + 1]] + [0])
    #         rew = np.array(reward[end_idx[j], end_idx[j + 1]] + [0])
    #
    #         # the next two lines implement GAE-Lambda advantage calculation
    #         delta = rew[:-1] + self.gamma * val[1:] - val[:-1]
    #         advantage = np.append(advantage, discount_cumsum(delta, self.gamma * self.lam))
    #         # The return is repeatedly calculated actually
    #         return_ = np.append(return_, discount_cumsum(rew, self.gamma)[:-1])
    #
    #     return advantage, return_

    def save(self, save_dir, episode):
        torch.save(self.policy_gpu.state_dict(), str(save_dir) + "/MAT_GPU_" + str(episode) + ".pt")
        torch.save(self.policy_cpu.satet_dict(), str(save_dir) + "/MAT_CPU_" + str(episode) + ".pt")

    def load(self, model_dir_gpu, model_dir_cpu):
        self.policy_gpu.load_state_dict(torch.load(model_dir_gpu))
        self.policy_cpu.load_state_dict(torch.load(model_dir_cpu))

    def copy_parameter(self):
        source_params = torch.nn.utils.parameters_to_vector(self.policy_gpu.parameters())
        torch.nn.utils.vector_to_parameters(source_params.cpu(), self.policy_cpu.parameters())


