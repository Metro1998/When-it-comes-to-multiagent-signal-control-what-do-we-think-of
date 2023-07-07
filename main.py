from collections import deque
import argparse
import gymnasium as gym
import numpy as np
import time
from utils.util import *
from policy.ma_transformer import MultiAgentTransformer
from policy.ma_transformer_trainer import PPOTrainer
from policy.rollout_buffer import PPOBuffer

# from policy.rollout_buffer import

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_dim', type=int, default=8, help='Observation dimension.')  # 观测空间维度
    parser.add_argument('--action_dim', type=int, default=8, help='Action dimension.')  # 动作空间维度
    parser.add_argument('--embd_dim', type=int, default=64, help='Embedding dimension.')  # 嵌入维度
    parser.add_argument('--agent_num', type=int, default=20, help='Number of agents.')  # 代理数量
    parser.add_argument('--block_num', type=int, default=1, help='Number of transformer blocks.')  # Transformer块数量
    parser.add_argument('--head_num', type=int, default=4, help='Number of attention heads.')  # 注意力头数量
    parser.add_argument('--std_clip', type=float, default=[0.1, 0.8], help='Standard deviation clip value.')  # 标准差剪裁值
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--clip_ratio', type=float, default=0.2, help='Clip ratio.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--entropy_coef_dis', type=float, default=0.005, help='Entropy coefficient for discrete action.')
    parser.add_argument('--entropy_coef_con', type=float, default=0.005, help='Entropy coefficient for continuous action.')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Maximum gradient norm.')
    parser.add_argument('--target_kl_dis', type=float, default=0.01, help='Target KL divergence for discrete action.')
    parser.add_argument('--target_kl_con', type=float, default=0.01, help='Target KL divergence for continuous action.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor.')
    parser.add_argument('--lam', type=float, default=0.9, help='Lambda parameter for GAE.')
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs.')

    # 添加Adam优化器参数
    parser.add_argument('--lr_actor_con', type=float, default=0.003, help='Learning rate for continuous actor.')
    parser.add_argument('--lr_std', type=float, default=0.02, help='Learning rate for std deviation.')
    parser.add_argument('--lr_actor_dis', type=float, default=0.003, help='Learning rate for discrete actor.')
    parser.add_argument('--adam_eps', type=float, default=1e-5, help='Epsilon value for Adam optimizer.')
    parser.add_argument('--lr_decay_rate', type=float, default=0.005, help='Learning rate decay rate.')
    parser.add_argument('--lr_critic', type=float, default=0.001, help='Learning rate for critic.')

    # 算法参数
    parser.add_argument('--total_episodes', type=int, default=500, help='Total number of episodes.')
    parser.add_argument('--history_length', type=int, default=5, help='Length of history for the observation.')

    # 环境设置
    parser.add_argument('--yellow', type=int, default=3, help='Duration of yellow phase in seconds.')
    parser.add_argument('--stage_num', type=int, default=8, help='Number of stages in the traffic signal.')
    parser.add_argument('--env_num', type=int, default=2, help='Number of parallel environments.')
    parser.add_argument('--local_net_file', type=str, default='envs/roadnet.net.xml', help='Path to local net file.')
    parser.add_argument('--local_route_file', type=str, default='envs/roadnet.rou.xml', help='Path to local route file.')
    parser.add_argument('--local_addition_file', type=str, default='envs/roadnet.add.xml', help='Path to local addition file.')
    parser.add_argument('--max_episode_step', type=int, default=3600 * 6, help='Maximum steps per episode.')
    parser.add_argument('--max_sample_step', type=int, default=3600, help='Maximum steps per sample.')

    args = parser.parse_args()

    """ ALGORITHM PARAMETERS """
    gamma = 0.99
    batch_size = 256
    agent_num = 20
    total_episodes = 500
    history_length = 5
    obs_dim = 8

    """ AGENT SETUP """
    trainer = PPOTrainer(args)


    """ ENVIRONMENT SETUP """
    yellow = 3
    stage_num = 8
    env_num = 2
    local_net_file = 'envs/roadnet.net.xml'
    local_route_file = 'envs/roadnet.rou.xml'
    local_addition_file = 'envs/roadnet.add.xml'
    max_episode_step = 3600 * 6
    max_sample_step = 3600
    pattern = 'queue'
    st = time.time()
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
                             ) for i in range(env_num)

    ])
    buffer = PPOBuffer(100000, env_num, agent_num, obs_dim, history_length, gamma=0.99, lam=0.95)

    # env = gym.make('sumo-rl-v1',
    #                yellow=[yellow] * agent_num,
    #                num_agent=agent_num,
    #                use_gui=False,
    #                net_file=local_net_file,
    #                route_file=local_route_file,
    #                addition_file=local_addition_file,
    #                pattern=pattern
    #                )
    # state, _ = env.reset()  # (✔)
    #
    # while True:
    #     random_numbers = np.random.randint(low=[0, 10], high=[8, 41], size=(12, 20, 2))
    #     action = {'duration': random_numbers[:, :, 1], 'stage': random_numbers[:, :, 0]}
    #     obs, reward, t, _, info = env.step(action)
    #     print("obs_queue", obs['queue'].shape)
    #     print("obs_stage", obs['stage'].shape)
    #     print(info['critical_step_idx'])
    #     if True in t:
    #         print(t)
    #         print('----------------------------------------------------------------------------------------')
    #
    # """ TRAINING LOGIC """
    for episode in range(total_episodes):

        # rollout phase
        obs_history = np.zeros((max_episode_step, env_num, agent_num, obs_dim), dtype=np.float32)
        next_obs, info = env.reset()
        obs_history_ptr = 0
        act_dis = np.zeros((env_num, agent_num), dtype=np.int64)
        act_con = np.zeros((env_num, agent_num), dtype=np.float32)

        while True:
            # We only retrieve the queue information, and then push it into the observation history (for GRU).
            obs = np.reshape(next_obs['queue'], (env_num, agent_num, -1))
            obs_history[obs_history_ptr + history_length - 1] = obs
            obs_rnn = obs_history[obs_history_ptr: obs_history_ptr + history_length].transpose((1, 2, 0, 3))
            agent_to_update = next_obs['agents_to_update']

            # Get action from the agent
            act_dis, logp_dis, act_con, logp_con, value = trainer.policy_cpu.act(obs_rnn, last_act_dis=act_dis, last_act_con=act_con, agent_to_update=agent_to_update)
            action = {'duration': map2real(act_con, 40), 'stage': act_dis}

            # Execute the environment and log data
            next_obs, reward, termi, trunc, info = env.step(action)
            buffer.store_trajectories(obs_rnn, reward, value, act_con, act_dis, logp_con, logp_dis)

            obs_history_ptr += 1
            print(termi, trunc)
            if termi.all() or trunc.all():
                buffer.finish_path(info[reward_idx])
                if not termi:
                    env.reset_truncated()
                    continue
                break

