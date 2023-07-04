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
    env_num = 6
    local_net_file = 'envs/roadnet.net.xml'
    local_route_file = 'envs/roadnet.rou.xml'
    local_addition_file = 'envs/roadnet.add.xml'
    max_step_round = 3600
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
                             max_step_episode=500,

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
        print(info['critical_step_idx'])
        if True in t:
            print(t)
            print('----------------------------------------------------------------------------------------')

    """ TRAINING LOGIC """
    for episode in range(total_episodes):

        # rollout phase
        next_obs, _ = env.reset()

        while True:
            # Retrieve the current observation, and transfer it to GRU. Utilize the deque implementation
            obs = batchify_obs(next_obs, device=self.device)

            # Get action from the agent
            act_con, act_dis, logp_dis, logp_con = agent.get_action(obs)

            # Execute the environment and log data
            next_obs, reward, termi, trunc, info = env.step()

            buffer.store_trajectories(obs, rewards, act_con, act_dis, logp_con, logp_dis)

            if termi or trunc:
                buffer.finish_path(info[reward_idx])
                if not termi:
                    continue
                break

