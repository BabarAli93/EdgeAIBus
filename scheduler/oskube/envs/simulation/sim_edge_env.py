import gymnasium as gym
import numpy as np
from scheduler.Scheduler import Scheduler
from gymnasium.spaces import (
    Box,
    MultiDiscrete,
    Discrete
)

from gymnasium.utils import seeding
from typing import (
    Dict,
    Any,
    Tuple
)

from datacenter.Datacenter import *
#from oskube.utils.annotations import override
from scheduler.oskube.utils.annotations import override

class SimEdgeEnv(gym.Env, Scheduler):
    def __init__(self, config: Dict[str, Any]):
        gym.Env.__init__(self)
        
        self.bitbrains_path = config['datasets']['bitbrains_path']
        schedule_config = dict({
            'prediction_length': config['prediction_length'],
            'bitbrains_path': config['datasets']['bitbrains_path'],
            'scheduler_path': config['scheduler_path']})
        Scheduler.__init__(self, schedule_config) 

        #self.bitbrains_path = config['datasets']['bitbrains_path']
        #self.yolo_path = config['datasets']['yolo_path']

        # Seed the environment
        self.seed(config.get('seed'))
        self.obs_elements = config['obs_elements']
        self.datacenter = config['datacenter']
        self.initial_placements = self.datacenter.containers_hosts
        self.prediction_length = config['prediction_length']
        self.observation_space, self.action_space = self._setup_space()
        print(self.observation_space)
        print(self.action_space)

        

        
    
    def seed(self, seed=None):
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        self._env_seed = seed
        self.base_env_seed = seed
        return [seed]
    
    @override(gym.Env)
    def step(self, action):
        print("Step function called")
        # # Implement the logic for taking a step in the environment
        # observation = self.observation_space.sample()
        # reward = 0.0  # Example reward
        # terminated = False  # Example done condition
        # truncated =  True
        # info = {}
        # return observation, reward, terminated, truncated, info
        return NotImplementedError
    
    @override(gym.Env)
    def reset(self, seed=None, options: dict[str, Any] = None):
        # # Handle seeding if provided
        # if seed is not None:
        #     self.seed(seed)

        # # Return the initial observation
        # observation = self.observation_space.sample()
        # info = {}  # Populate with any relevant info if necessary
        
        # # If using Gymnasium v0.26+, return (observation, info)
        # return observation, info

        return NotImplementedError
    

    @override(gym.Env)
    def reset(self) -> np.ndarray:
        """Resets the state of the environment and returns
        an initial observation.
        Returns:
            (object): the initial observation.
        Remarks:
            each time resets to a different initial state
        """
        if self.timestep_reset:
            self.global_timestep = 0
            self.timestep = 0
        if self.placement_reset:
            self.services_nodes = deepcopy(self.initial_placements)
        return self.observation

    
    def _setup_space(self):
        """
        "obs_elements": ["containers_hosts", "hosts_alloc", "hosts_requests", "hosts_usages"]
        sla_items = ['sla_violations']
        containers_requests = ['']

        ["", "", "", "", 
      "", ""]

        """
        #numuber of elements based on the obs in the observation space
        obs_size = 0
        predictions = ['cpu_predictions']  #                   ## Future CPU predictions of each node
        node_resource_size_items = [                           ## CPU and memory resources of each node
            "hosts_alloc",
            "hosts_requests",
            "hosts_resources_util"
        ]
        node_services_size_items = ['containers_requests']     ## CPU and mem request of each container
        containers_hosts = ['containers_hosts']               ## Current placement of containers on nodes. It should be one hot encoding
        sla_items = ['sla_violations']                        ## SLA violation rate of each container for slected core and model
        for elm in self.obs_elements:
            if elm in node_resource_size_items:
                obs_size += self.datacenter.num_hosts * self.datacenter.num_resources
            elif elm in node_services_size_items:
                obs_size += self.datacenter.num_containers * self.datacenter.num_resources
            elif elm in predictions:
                obs_size += self.datacenter.num_hosts * self.prediction_length
            elif elm in sla_items:
                obs_size += self.datacenter.num_containers
            elif elm in containers_hosts:
                obs_size += self.datacenter.num_hosts * self.datacenter.num_containers

        higher_bound = 1 
        # generate observation and action spaces
        observation_space = Box(
            low=0, high=higher_bound, shape=(obs_size, ),
            dtype=np.float64, seed=self._env_seed)

        # if self.discrete_actions:
        action_space = Discrete(2 * self.datacenter.num_hosts, seed=self._env_seed)

        return observation_space, action_space