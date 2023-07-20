from collections import deque
import argparse
import json
import gymnasium as gym
import numpy as np
import time
from utils.util import *
from policy.ma_transformer import MultiAgentTransformer
from policy.ma_transformer_trainer import PPOTrainer
from policy.rollout_buffer import PPOBuffer
from torch.utils.tensorboard import SummaryWriter

# from policy.rollout_buffer import

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_dim', type=int, default=8, help='Observation dimension.')  # 观测空间维度
    parser.add_argument('--action_dim', type=int, default=8, help='Action dimension.')  # 动作空间维度
    parser.add_argument('--embd_dim', type=int, default=64, help='Embedding dimension.')  # 嵌入维度
    parser.add_argument('--agent_num', type=int, default=20, help='Number of agents.')  # 代理数量
    parser.add_argument('--block_num', type=int, default=1, help='Number of transformer blocks.')  # Transformer块数量
    parser.add_argument('--head_num', type=int, default=8, help='Number of attention heads.')  # 注意力头数量
    parser.add_argument('--std_clip', type=float, default=[0.1, 0.4], help='Standard deviation clip value.')  # 标准差剪裁值
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--clip_ratio', type=float, default=0.2, help='Clip ratio.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--entropy_coef_dis', type=float, default=0.005, help='Entropy coefficient for discrete action.')
    parser.add_argument('--entropy_coef_con', type=float, default=0.005, help='Entropy coefficient for continuous action.')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Maximum gradient norm.')
    parser.add_argument('--target_kl_dis', type=float, default=0.025, help='Target KL divergence for discrete action.')
    parser.add_argument('--target_kl_con', type=float, default=0.05, help='Target KL divergence for continuous action.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor.')
    parser.add_argument('--lam', type=float, default=0.9, help='Lambda parameter for GAE.')
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs.')
    parser.add_argument('--comment', type=str, default='_test', help='Comment for tensorboard.')

    # 添加Adam优化器参数
    parser.add_argument('--lr_actor_con', type=float, default=0.0003, help='Learning rate for continuous actor.')
    parser.add_argument('--lr_std', type=float, default=0.02, help='Learning rate for std deviation.')
    parser.add_argument('--lr_actor_dis', type=float, default=0.0003, help='Learning rate for discrete actor.')
    parser.add_argument('--adam_eps', type=float, default=1e-5, help='Epsilon value for Adam optimizer.')
    parser.add_argument('--lr_decay_rate', type=float, default=0.005, help='Learning rate decay rate.')
    parser.add_argument('--lr_critic', type=float, default=0.001, help='Learning rate for critic.')

    # 算法参数
    parser.add_argument('--total_episodes', type=int, default=500, help='Total number of episodes.')
    parser.add_argument('--history_len', type=int, default=5, help='Length of history for the observation.')

    # 环境设置
    parser.add_argument('--yellow', type=int, default=3, help='Duration of yellow phase in seconds.')
    parser.add_argument('--stage_num', type=int, default=8, help='Number of stages in the traffic signal.')
    parser.add_argument('--env_num', type=int, default=6, help='Number of parallel environments.')
    parser.add_argument('--local_net_file', type=str, default='envs/roadnet.net.xml', help='Path to local net file.')
    parser.add_argument('--local_route_file', type=str, default='envs/roadnet.rou.xml', help='Path to local route file.')
    parser.add_argument('--local_addition_file', type=str, default='envs/roadnet.add.xml', help='Path to local addition file.')
    # parser.add_argument('--max_episode_step', type=int, default=4800, help='Maximum steps per episode.')
    # parser.add_argument('--max_sample_step', type=int, default=3600, help='Maximum steps per sample.')

    args = parser.parse_args()
    with open('runs/args/' + args.comment + '.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    """ ALGORITHM PARAMETERS """
    gamma = 0.99
    batch_size = 256
    agent_num = 20
    total_episodes = 500
    history_len = 5
    obs_dim = 8

    """ AGENT SETUP """
    writer = SummaryWriter(comment=args.comment)
    trainer = PPOTrainer(writer, args)

    """ ENVIRONMENT SETUP """
    yellow = 3
    stage_num = 8
    env_num = 6
    local_net_file = 'envs/roadnet.net.xml'
    local_route_file = 'envs/roadnet.rou.xml'
    local_addition_file = 'envs/roadnet.add.xml'
    max_episode_step = 4800
    max_sample_step = 600
    pattern = 'queue'
    env = gym.vector.AsyncVectorEnv([
        lambda i=i: gym.make('sumo-rl-v1',
                             yellow=[yellow] * agent_num,
                             num_agent=agent_num,
                             use_gui=False,
                             net_file=local_net_file,
                             route_file=local_route_file,
                             addition_file=local_addition_file,
                             pattern=pattern,
                             max_episode_step=max_episode_step,
                             max_sample_step=max_sample_step,
                             comment=args.comment,
                             ) for i in range(env_num)

    ])
    # st = time.time()
    # _, _ = env.reset()
    # cnt = 0
    # while 1:
    #     action = env.action_space.sample()
    #     _, _, _, _, info = env.step(action)
    #     cnt += 1
    #     if info['termi']:
    #         break
    # print(cnt)
    # print('---------------------------------------Env step time: ', time.time() - st)

    # """ TRAINING LOGIC """
    for episode in range(total_episodes):
        test_st = time.time()
        # rollout phase
        obs_history = np.zeros((max_episode_step * 2, env_num, agent_num, obs_dim), dtype=np.float32)
        next_obs, info = env.reset()

        act_dis = None
        history_ptr = 0

        while True:
            # We only retrieve the queue information, and then push it into the observation history (for GRU).
            obs = np.reshape(next_obs['queue'], (env_num, agent_num, -1))
            obs_history[history_ptr + history_len - 1] = obs
            obs_rnn = obs_history[history_ptr: history_ptr + history_len].transpose((1, 2, 0, 3))
            history_ptr += 1

            act_dis_infer = np.zeros((env_num, agent_num), dtype=np.int64) if act_dis is None else act_dis
            act_con_infer = np.array([info['left_time'][i] for i in range(env_num)])
            agent_to_update = np.array([info['agents_to_update'][i] for i in range(env_num)])

            # Get action from the agent
            # st = time.time()
            act_dis, logp_dis, act_con, logp_con, value = trainer.policy_cpu.act(obs_rnn, act_dis_infer=act_dis_infer, act_con_infer=remap(act_con_infer, 40), agent_to_update=agent_to_update)
            # print('act time: ', time.time() - st)
            # Execute the environment and log data
            action = {'duration': map2real(act_con, 40), 'stage': act_dis}
            # st = time.time()
            next_obs, reward, _, _, info = env.step(action)
            # print('env step time: ', time.time() - st)
            trainer.buffer.store_trajectories(obs_rnn, act_dis_infer, remap(act_con_infer, 40), agent_to_update,
                                              act_dis, act_con, logp_dis, logp_con, value, reward)

            trunc = np.array([info['trunc'][i] for i in range(env_num)])
            termi = np.array([info['termi'][i] for i in range(env_num)])
            # todo
            if trunc.all():
                critical_step_idx = [info['critical_step_idx'][i] for i in range(env_num)]
                trainer.buffer.finish_path(critical_step_idx=critical_step_idx)
                trainer.update()
            if termi.all():
                print('---------------------------------------Test time: ', time.time() - test_st)
                break

        # update phase
        # TODO 自动对齐 以及 类型转换 以及 reset buffer 以及 update 以及 util 简化计算
        # 读出数据


