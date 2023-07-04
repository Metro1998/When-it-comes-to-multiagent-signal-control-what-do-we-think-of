from collections import deque

import gymnasium as gym
import numpy as np
import time
from utils.util import *
from policy.ma_transformer import MultiAgentTransformer
from policy.ma_transformer_trainer import PPOTrainer

# from policy.rollout_buffer import

if __name__ == '__main__':

    """ ALGORITHM PARAMETERS """
    gamma = 0.99
    batch_size = 256
    agent_num = 20
    total_episodes = 500
    history_length = 5

    """ AGENT SETUP """
    # agent_gpu = MultiAgentTransformer(obs_dim=200, action_dim=8, embd_dim=64, agent_num=20, block_num=1, head_num=8,
    #                                   std_clip=[0.1, 0.9], max_green=40, device='cuda:0')
    # agent_cpu = MultiAgentTransformer(obs_dim=200, action_dim=8, embd_dim=64, agent_num=20, block_num=1, head_num=8,
    #                                   std_clip=[0.1, 0.9], max_green=40, device='cpu')
    # buffer =
    # agent_trainer = PPOTrainer(agent_gpu, agent_cpu, gamma=gamma, batch_size=batch_size,  agent_num=agent_num)

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

    # env = gym.make('sumo-rl-v1',
    #                yellow=[yellow] * agent_num,
    #                num_agent=agent_num,
    #                use_gui=False,
    #                net_file=local_net_file,
    #                route_file=local_route_file,
    #                addition_file=local_addition_file,
    #                pattern=pattern
    #                )
    state, _ = env.reset()  # (✔)

    while True:
        random_numbers = np.random.randint(low=[0, 10], high=[8, 41], size=(12, 20, 2))
        action = {'duration': random_numbers[:, :, 1], 'stage': random_numbers[:, :, 0]}
        obs, reward, t, _, info = env.step(action)
        print("obs_queue", obs['queue'].shape)
        print("obs_stage", obs['stage'].shape)
        print(info['critical_step_idx'])
        if True in t:
            print(t)
            print('----------------------------------------------------------------------------------------')

    """ TRAINING LOGIC """
    for episode in range(total_episodes):

        # rollout phase
        obs_history = np.zeros((max_episode_step, env_num, agent_num, np.float32))
        next_obs, info = env.reset()
        obs_history_ptr = 0
        act_dis = -np.ones((env_num, agent_num), dtype=np.int64)
        act_con = np.zeros((env_num, agent_num), dtype=np.float32)

        while True:
            # We only retrieve the queue information, and then push it into the observation history (for GRU).
            obs = np.reshape(next_obs['queue'], (env_num, agent_num, -1))
            obs_history[obs_history_ptr + obs_history - 1] = obs
            obs_rnn = obs_history[obs_history_ptr: obs_history_ptr + history_length]

            # Get action from the agent
            act_dis, logp_dis, act_con, logp_con, values = trainer.agent_cpu.act(obs_rnn, last_act_dis=act_dis, last_act_con=act_con, agent_to_update=info['agent_to_update'])

            # Execute the environment and log data
            next_obs, reward, termi, trunc, info = env.step()

            buffer.store_trajectories(obs_rnn, reward, value, act_con, act_dis, logp_con, logp_dis)

            obs_history_ptr += 1
            if termi or trunc:
                buffer.finish_path(info[reward_idx])
                if not termi:
                    env.reset_truncated()
                    continue
                break

