import gymnasium as gym
import numpy as np
import traci
import sumolib

from gymnasium import spaces
from typing import Callable, Optional, Tuple, Union, List


class TrafficSignal:
    """
    This class represents a Traffic Signal controlling an intersection.
    It is responsible for retrieving information and changing the traffic phase using the Traci API.
    """
    def __init__(self,
                 tl_id,
                 sumo):

        self.id = tl_id
        self.sumo = sumo

        self.in_lanes = list(
            dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id))
        )
        self.out_lanes = None  # TODO

    # def build_phase:
    def subscribe(self):
        """
        Pre subscribe the information we interest, so as to accelerate the information retrieval.
        See https://sumo.dlr.de/docs/TraCI.html "Performance" for more detailed explanation.
        :return:
        """

        for lane_id in self.in_lanes:
            self.sumo.lane.subscribe(lane_id, [traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER, traci.constants.VAR_WAITING_TIME])


class SUMOEnv(gym.Env):
    CONNECTION_LABEL = 0  # For traci multi-client support

    def __init__(self,
                 yellow,
                 num_stage: int,
                 num_agent: int,
                 use_gui: bool,
                 net_file: str,
                 route_file: str,
                 min_green: int = 0,
                 max_green: int = 40,
                 sumo_seed: Union[str, int] = "random",
                 max_depart_delay: int = -1,
                 waiting_time_memory: int = 1000,
                 time_to_teleport: int = -1,
                 ):

        self.action_space = spaces.Dict({
            'stages': spaces.MultiDiscrete(np.array([num_stage - 1] * num_agent)),
            'duration': spaces.Box(low=np.array([min_green] * num_agent), high=np.array([max_green] * num_agent),
                                   dtype=np.float32)
        })

        self.use_gui = use_gui
        self.net = net_file
        self.route = route_file
        self.sumo_seed = sumo_seed
        if self.use_gui or self.render_mode is not None:
            self.sumo_binary = sumolib.checkBinary("sumo-gui")
        else:
            self.sumo_binary = sumolib.checkBinary("sumo")
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.waiting_time_memory = waiting_time_memory  # Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime)
        self.time_to_teleport = time_to_teleport
        self.label = str(SUMOEnv.CONNECTION_LABEL)
        SUMOEnv.CONNECTION_LABEL += 1  # Increments itself when an instance is initialized

        self.episode = 0
        self.sumo = None
        self.tl_ids = None
        self.observations = None
        self.rewards = None

    # def step(self,
    #          action: ActType
    # ):

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

        self.tl_ids = list(self.sumo.trafficlight_getIDList())
        self.observations = {tl: None for tl in self.tl_ids}
        self.rewards = {tl: None for tl in self.tl_ids}

    def reset(self, seed: Optional[int] = None, **kwargs):
        """

        :param seed:
        :param kwargs:
        :return:
        """
        super(SUMOEnv, self).reset(seed=seed, **kwargs)
        if self.episode != 0:
            self.close()
        self.episode += 1

        if seed is not None:
            self.sumo_seed = seed
        self.start_simulation()

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

