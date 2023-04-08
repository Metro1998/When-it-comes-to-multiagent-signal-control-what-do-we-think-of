import gymnasium as gym
import numpy as np
import traci
import sumolib

from gymnasium import spaces
from collections import deque
from typing import Callable, Optional, Tuple, Union, List

LIBSUMO = False


class TrafficSignal:
    """
    This class represents a Traffic Signal controlling an intersection.
    It is responsible for retrieving information and changing the traffic phase using the Traci API.

    IMPORTANT!!! NOTE THAT
    Our reward is defined as the change in vehicle number of one specific junction.
    Our state is defined as the pressure between the inlanes and outlanes.
    """

    def __init__(self, tl_id, yellow, sumo):

        self.id = tl_id
        self.yellow = yellow
        self.sumo = sumo

        # The schedule is responsible for the automatic timing for the incoming green stage.
        # | 0 | 0 | 0 | 16 |
        # | yellow len| when 16 is dequeued, the stage is automatically transferred to the green stage and 16 is for duration.
        self.schedule = deque()
        self.duration = None

        # Links is relative with connections defined in the rou.xml, what's more the connection definition should be
        # relative with traffic state definition. Therefore, there is no restriction that the connection should start
        # at north and step clockwise then.
        all_lanes = self.sumo.trafficlight.getControlledLinks(self.id)
        self.in_lanes = [conn[0][0] for conn in all_lanes]
        # Delete the right turn movement.
        del self.in_lanes[0::3]
        self.out_lanes = [conn[0][1] for conn in all_lanes]
        del self.out_lanes[0::3]

        self.subscribe()

        self.inlane_halting_vehicle_number = None
        self.inlane_halting_vehicle_number_old = None
        self.inlane_waiting_time = None
        self.outlane_halting_vehicle_number = None
        self.outlane_waiting_time = None

    def set_stage_duration(self, stage: str, duration: int):
        """
        Call this at the beginning the of one stage, which includes the switching yellow light between two different
        green light.
        In add.xml the stage is defined as the yellow stage then next green stage, therefore the yellow stage is first
        implemented, and after self.yellow seconds, it will automatically transfer to green stage, through a schedule to
        set the incoming green stage's duration.
        :return:
        """
        self.sumo.trafficlight.setPhase('universal_program', stage)
        self.sumo.trafficlight.setPhaseDuration(self.yellow)
        for i in range(self.yellow):
            self.schedule.append(0)
        self.duration = duration
        self.schedule.append(duration)

    def check(self):
        """
        Check whether the yellow stage is over and automatically extend the green light.
        # | 0 | 0 | 0 | 16 |  --->  | 0 | 0 | 0 | 0 | ... | 0 | -1 |
        #                                       {     16X     } where -1 indicates that the agent should get a new action
        :return:
        """
        if self.schedule[0] > 0:
            self.sumo.trafficlight.setPhaseDuration(self.schedule[0])
            for i in range(self.schedule[0] - 1):
                self.schedule.append(0)
            self.schedule.popleft()
            self.schedule.append(-1)

        return self.schedule[0]

    def pop(self):
        self.schedule.popleft()

    def subscribe(self):
        """
        Pre subscribe the information we interest, so as to accelerate the information retrieval.
        See https://sumo.dlr.de/docs/TraCI.html "Performance" for more detailed explanation.
        :return:
        """

        for lane_id in self.in_lanes:
            self.sumo.lane.subscribe(lane_id, [traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER,
                                               traci.constants.VAR_WAITING_TIME])

        for lane_id in self.out_lanes:
            self.sumo.lane.subscribe(lane_id, [traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER,
                                               traci.constants.VAR_WAITING_TIME])

    def get_subscription_result(self):
        self.inlane_halting_vehicle_number = np.array(
            [list(self.sumo.lane.getSubscriptionResults(lane_id).values())[0] for lane_id in self.in_lanes])
        # self.inlane_waiting_time = [list(self.sumo.lane.getSubscriptionResults(lane_id).values())[1] for lane_id in self.in_lanes]
        self.outlane_halting_vehicle_number = np.array(
            [list(self.sumo.lane.getSubscriptionResults(lane_id).values())[0] for lane_id in self.out_lanes])
        # self.outlane_waiting_time = [list(self.sumo.lane.getSubscriptionResults(lane_id).values())[1] for lane_id in self.out_lanes]

    def compute_reward(self):
        if not self.inlane_halting_vehicle_number_old:
            reward = -sum(self.inlane_halting_vehicle_number)
        else:
            reward = sum(self.inlane_halting_vehicle_number_old) - sum(self.inlane_halting_vehicle_number)
        self.inlane_halting_vehicle_number_old = self.inlane_halting_vehicle_number

        return reward

    def compute_observation(self):
        observation = self.inlane_halting_vehicle_number - self.outlane_halting_vehicle_number

        return observation


class SUMOEnv(gym.Env):
    CONNECTION_LABEL = 0  # For traci multi-client support

    def __init__(self,
                 yellow,
                 num_stage: int,
                 num_agent: int,
                 use_gui: bool,
                 net_file: str,
                 route_file: str,
                 addition_file: str,
                 min_green: int = 0,
                 max_green: int = 40,
                 sumo_seed: Union[str, int] = "random",
                 max_depart_delay: int = -1,
                 waiting_time_memory: int = 1000,
                 time_to_teleport: int = -1,
                 max_step_round: int = 3600,
                 max_step_sample: int = 2000,
                 hybrid: bool = True
                 ):

        # self.action_space = spaces.Tuple((spaces.Discrete(num_stage), spaces.Box(low=min_green, high=max_green, shape=(1,), dtype=np.int64))) * num_agent
        # self.observation_space = spaces.Box(low=-100, high=100, shape=(num_agent, num_stage), dtype=np.int64)

        self.yellow = yellow
        self.use_gui = use_gui
        self.net = net_file
        self.route = route_file
        self.addition = addition_file
        self.sumo_seed = sumo_seed
        # Whether the agent reaches the terminal state as defined under the MDP of the task.
        self.num_stage = num_stage
        self.num_agent = num_agent
        self.min_green = min_green
        self.max_green = max_green
        self.max_step_episode = max_step_round
        # Whether the truncation condition outside the scope of the MDP is satisfied.
        self.max_step_sample = max_step_sample

        if self.use_gui or self.render_mode is not None:
            self.sumo_binary = sumolib.checkBinary("sumo-gui")
        else:
            self.sumo_binary = sumolib.checkBinary("sumo")
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.waiting_time_memory = waiting_time_memory  # Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime)
        self.time_to_teleport = time_to_teleport
        self.label = str(SUMOEnv.CONNECTION_LABEL)
        SUMOEnv.CONNECTION_LABEL += 1  # Increments itself when an instance is initialized

        self.step_round = None
        self.step_sample = None
        self.episode = 0
        self.sumo = None
        self.tl_ids = None
        self.tls = None
        self.observations = None
        self.rewards = None
        self.terminated = None
        self.truncated = None

    @property
    def observation_space(self):
        """Return the observation space of a traffic signal.
        Only used in case of single-agent environment.
        """
        return spaces.Box(low=-100, high=100, shape=(self.num_agent, self.num_stage), dtype=np.int64)

    @property
    def action_space(self):
        """Return the action space of a traffic signal.
        Only used in case of single-agent environment.
        """
        return spaces.Tuple((spaces.MultiDiscrete(np.array([self.num_stage] * self.num_agent)),
                             spaces.Box(low=self.min_green, high=self.max_green, shape=(self.num_agent,), dtype=np.int64)))

    def step(self, action):
        """

        :param action:
        :return:
        """
        for k, v in enumerate(action[0]):
            # We use s == self.num_stage to indicate that the agent doesn't need to execute at this step.
            if v != self.num_stage:
                # transfer_matrix
                self.tls[k].set_stage_duration(action[0][k], action[1][k])

        while True:
            # Just step the simulation.
            self.sumo.simulationStep()
            # Pop the most left element of the schedule.
            [tl.pop() for tl in self.tls]
            self.step_round += 1
            if self.step_round >= self.max_step_episode:
                self.terminated = True
            # Automatically execute the transition from the yellow stage to green stage, and simultaneously set the end indicator -1.
            # Moreover, check() will return the front of the schedule.
            checks = [tl.check() for tl in self.tls]
            # ids are agents who should act right now.
            if -1 in checks or self.terminated:
                ids = [k for k, v in enumerate(checks) if v == -1]
                info = {'incoming': ids}
                break

        self.step_sample += 1
        if self.step_round >= self.max_step_sample:
            self.truncated = True

        [tl.get_subscription_result() for tl in self.tls]
        observation = np.array([tl.compute_observation() for tl in self.tls])
        reward = np.array([tl.compute_reward() for tl in self.tls])

        return observation, reward, self.terminated, self.truncated, info

    def start_simulation(self):
        """
        Start the sumo simulation according to the sumo commend.
        :return:
        """
        sumo_cmd = [
            self.sumo_binary,
            "-n",
            self.net,
            "-r",
            self.route,
            "-a",
            self.addition,
            "--max-depart-delay",
            str(self.max_depart_delay),
            "--waiting-time-memory",
            str(self.waiting_time_memory),
            "--time-to-teleport",
            str(self.time_to_teleport),
        ]

        if self.sumo_seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])

        if self.use_gui or self.render_mode is not None:
            sumo_cmd.extend(["--start", "--quit-on-end"])

        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)

        if self.use_gui or self.render_mode is not None:
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

        self.tl_ids = list(self.sumo.trafficlight.getIDList())
        self.tls = [TrafficSignal(tl_id, yellow, self.sumo) for tl_id, yellow in zip(self.tl_ids, self.yellow)]

    def reset(self, seed: Optional[int] = None, **kwargs):
        """

        :param seed:
        :param kwargs:
        :return:
        """
        super(SUMOEnv, self).reset(seed=seed, **kwargs)
        if self.step_round != 0:
            self.close()
        self.terminated = False
        self.step_round = 0

        if seed is not None:
            self.sumo_seed = seed
        self.start_simulation()

        [tl.get_subscription_result() for tl in self.tls]
        observation = np.array([tl.compute_observation() for tl in self.tls])
        info = {}
        return observation, info

    def reset_truncated(self):
        self.truncated = False
        self.step_sample = 0

    def close(self):
        """
        Close the environment and stop the SUMO simulation.
        :return:
        """

        if self.sumo is None:
            return

        if not LIBSUMO:
            traci.switch(self.label)
        traci.close()


if __name__ == "__main__":
    env = SUMOEnv(yellow=np.array([3, 3, 3]),
                  num_stage=8,
                  num_agent=3,
                  use_gui=False,
                  net_file='corrdor.net.xml',
                  route_file='hangzhou.rou.xml',
                  addition_file='traffic_light.add.xml'
                  )
    env.reset()

    action = env.action_space.sample()
    for k, v in enumerate(action[0]):
        print(k, v)
    ts = TrafficSignal(env.tl_ids[0], 3, env.sumo)
    ts.get_subscription_result()
    # obs = ts.compute_observation()
    # rew = ts.compute_reward()

    obs, rew, ter, trun, info = env.step(action)
    print(obs, rew)
