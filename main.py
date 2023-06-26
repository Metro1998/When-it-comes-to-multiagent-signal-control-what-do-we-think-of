import gymnasium as gym
import numpy as np
from utils.util import *

if __name__ == '__main__':

    """ ALGORITHM PARAMETERS """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gamma = 0.99
    batch_size = 256
    agent_num = 20

    """ ENVIRONMENT SETUP """
    yellow = 3
    stage_num = 8
    env_num = 2
    local_net_file = 'envs/roadnet.net.xml'
    local_route_file = 'envs/roadnet.rou.xml'
    local_addition_file = 'envs/roadnet.add.xml'
    max_step_round = 3600
    pattern = 'queue'
    env = gym.vector.AsyncVectorEnv([
        lambda i=i: gym.make('sumo-rl-v1',
                             yellow=[yellow] * agent_num,
                             num_stage=stage_num,  # Be careful with the continuous action space pattern!!!
                             num_agent=agent_num,
                             use_gui=False,
                             net_file=local_net_file,
                             route_file=local_route_file,
                             addition_file=local_addition_file,
                             max_step_round=max_step_round,
                             pattern=pattern
                             ) for i in range(env_num)

    ])
    state, _ = env.reset()  # (✔)
    print(state['queue'].shape)

    print(env.observation_space)
    print(env.action_space)
    """ TRAINING LOGIC """
    for episode in range(args.total_episodes):

        # rollout phase
        next_obs = env.reset()
        while True:
            obs = batchify_obs(next_obs, device=self.device)

            # Get action from the agent
            act_con, act_dis, logp_dis, logp_con = agent.get_action(obs)

            # Execute the environment and log data
            next_obs, rewards, terms, truncs, infos = env.step(

            )

            agent.buffer.store_trajectories(obs, rewards, act_con, act_dis, logp_con, logp_dis)
