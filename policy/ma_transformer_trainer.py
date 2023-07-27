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

        self.policy_gpu = MultiAgentTransformer(args.obs_dim, args.action_dim, args.embd_dim, args.agent_num, args.block_num, args.head_num, args.std_clip, args.dropout, device='cuda:0')
        self.policy_gpu_target = MultiAgentTransformer(args.obs_dim, args.action_dim, args.embd_dim, args.agent_num, args.block_num, args.head_num, args.std_clip, args.dropout, device='cuda:0')
        self.policy_cpu = MultiAgentTransformer(args.obs_dim, args.action_dim, args.embd_dim, args.agent_num, args.block_num, args.head_num, args.std_clip, args.dropout, device='cpu')
        self.copy_parameter()
        self.buffer = PPOBuffer(3600, args.env_num, args.agent_num, args.obs_dim, args.history_len, args.gamma, args.lam)
        self.random_seed = args.random_seed
        self.agent_num = args.agent_num
        self.clip_ratio = args.clip_ratio
        self.batch_size = args.batch_size
        self.entropy_coef_dis = args.entropy_coef_dis
        self.entropy_coef_con = args.entropy_coef_con
        self.max_grad_norm = args.max_grad_norm  # gradient clip value_proj, is set to be 0.5 in MAT
        self.target_kl_dis = args.target_kl_dis
        self.target_kl_con = args.target_kl_con
        self.gamma = args.gamma
        self.lam = args.lam
        self.epochs = args.epochs
        self.writer = writer
        self.global_step = 0

        self.optimizer_actor_dis = torch.optim.Adam(self.policy_gpu.decoder_dis.parameters(), lr=args.lr_actor_dis, eps=args.adam_eps)
        self.optimizer_actor_con = torch.optim.Adam(self.policy_gpu.decoder_con.parameters(), lr=args.lr_actor_con, eps=args.adam_eps)
        self.optimizer_critic = torch.optim.Adam(self.policy_gpu.encoder.parameters(), lr=args.lr_critic, eps=args.adam_eps)
        self.lr_scheduler_actor_dis = CosineWarmupScheduler(optimizer=self.optimizer_actor_dis, warmup=(10^4) / 2, max_iters=10^5)
        self.lr_scheduler_actor_con = CosineWarmupScheduler(optimizer=self.optimizer_actor_con, warmup=(10^4) / 2, max_iters=10^5)
        self.lr_scheduler_critic = CosineWarmupScheduler(optimizer=self.optimizer_critic, warmup=(10^4) / 2, max_iters=10^5)
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
                joint_adv_batch = torch.zeros(size=(self.batch_size, 1), dtype=torch.float32, device='cuda:0')
                for i in range(self.batch_size):
                    joint_adv_batch[i] = torch.mean(adv_batch[i][agent_batch[i] == 1], dim=-1, keepdim=True)
                # joint_adv_batch = torch.mean(adv_batch, dim=-1, keepdim=True)
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

                ### Update decoder_dis ###
                self.optimizer_actor_dis.zero_grad()
                new_logp_dis_batch, entropy_dis, new_logp_con_batch, entropy_con = self.policy_gpu.evaluate_actions(
                    obs_batch, act_dis_batch, act_con_batch, agent_batch, None, self.policy_gpu_target.decoder_con)

                imp_weights_dis = torch.exp(new_logp_dis_batch - old_logp_dis_batch)
                surr1_dis = imp_weights_dis * joint_adv_batch
                surr2_dis = torch.clamp(imp_weights_dis, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * joint_adv_batch
                loss_pi_dis = - (torch.min(surr1_dis, surr2_dis) + self.entropy_coef_dis * entropy_dis).mean()
                # loss_pi_dis = - torch.min(surr1_dis, surr2_dis).mean()
                with torch.no_grad():
                    # Trick, calculate approx_kl http://joschu.net/blog/kl-approx.html
                    approx_kl_dis = ((imp_weights_dis - 1) - (new_logp_dis_batch - old_logp_dis_batch)).mean()
                if approx_kl_dis > self.target_kl_dis:
                    print('Early stopping at step {} due to reaching max kl_dis.'.format(self.global_step))
                else:
                    loss_pi_dis.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_gpu.decoder_dis.parameters(), norm_type=2, max_norm=self.max_grad_norm)
                    self.optimizer_actor_dis.step()

                ### Update decoder_con ###
                self.optimizer_actor_con.zero_grad()
                new_logp_dis_batch, entropy_dis, new_logp_con_batch, entropy_con = self.policy_gpu.evaluate_actions(
                    obs_batch, act_dis_batch, act_con_batch, agent_batch, self.policy_gpu_target.decoder_dis, None)
                imp_weights_con = torch.exp(new_logp_con_batch - old_logp_con_batch)
                surr1_con = imp_weights_con * joint_adv_batch
                surr2_con = torch.clamp(imp_weights_con, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * joint_adv_batch

                # Andrychowicz, et al. (2021) overall find no evidence that the entropy term improves performance on
                # continuous control environments.
                loss_pi_con = - torch.min(surr1_con, surr2_con).mean()
                with torch.no_grad():
                    approx_kl_con = ((imp_weights_con - 1) - (new_logp_con_batch - old_logp_con_batch)).mean()
                if approx_kl_con > self.target_kl_con:
                    print('Early stopping at step {} due to reaching max kl_con.'.format(self.global_step))
                else:
                    loss_pi_con.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_gpu.decoder_con.parameters(), norm_type=2, max_norm=self.max_grad_norm)
                    self.optimizer_actor_con.step()

                self.writer.add_scalar('loss/critic', critic_loss, self.global_step)
                self.writer.add_scalars('loss/actor', {'discrete_head': loss_pi_dis, 'continuous_head': loss_pi_con}, self.global_step)
                self.writer.add_scalars('entropy', {'discrete_head': entropy_dis.mean(), 'continuous_head': entropy_con.mean()}, self.global_step)
                self.writer.add_scalars('approx_kl', {'discrete_head': approx_kl_dis, 'continuous_head': approx_kl_con}, self.global_step)
                self.writer.flush()
                self.global_step += 1

                self.lr_scheduler_actor_con.step()
                self.lr_scheduler_actor_dis.step()
                self.lr_scheduler_critic.step()
        self.copy_parameter()

    # def recompute(self, observation, reward, end_idx):
    #     """
    #     Trick[0], recompute the value_proj prediction when calculate the advantage
    #     Compute advantage function A(s, a) based on global V-value_proj network with GAE, where a represents joint action
    #
    #     :param observation:
    #     :param reward:
    #     :param end_idx:
    #     :return:
    #     """
    #     # TODO: policy_gpu
    #     # (num_steps, num_agents, obs_dim) --> (num_steps)
    #     value_proj = self.policy.encoder(check(observation)).squeeze().detach().numpy()
    #     # (num_steps, num_agents) --> (num_steps)
    #     reward = np.sum(reward, axis=-1)
    #
    #     advantage = np.array([])
    #     return_ = np.array([])
    #     for j in range(len(end_idx) - 1):
    #         val = np.array(value_proj[end_idx[j], end_idx[j + 1]] + [0])
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
        source_params = torch.nn.utils.parameters_to_vector(self.policy_gpu.parameters())
        torch.nn.utils.vector_to_parameters(source_params, self.policy_gpu_target.parameters())


