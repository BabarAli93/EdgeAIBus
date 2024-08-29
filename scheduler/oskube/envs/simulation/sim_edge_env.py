import gymnasium as gym
import numpy as np
from scheduler.Scheduler import Scheduler
from scheduler.oskube.utils import (
    Preprocessor,
    override,
    Mapper
)

from gymnasium import spaces

from gymnasium.utils import seeding
from typing import (
    Dict,
    Any,
    List,
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
        # Seed the environment
        self.seed(config.get('seed'))
        self.datacenter = config['datacenter']
        self.yolodata = self.datacenter.yolo_reading(config['datasets']['yolo_path'])
        self.model_versions: int = 2             # YOLO Nano and Small only
        self.obs_elements: List[str] = config['obs_elements']
        
        self.timesteps:int = 0
        self.initial_placements = self.datacenter.containers_hosts
        self.prediction_length = config['prediction_length']
        self.action2host_mapper: Dict[int, Tuple] = {}
        self.observation_space, self.action_space = self._setup_space()
        print(self.observation_space)
        print(self.action_space)
        _, _ = self.reset()

        
    
    def seed(self, seed=None):
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        self._env_seed = seed
        self.base_env_seed = seed
        return [seed]
    
    @override(gym.Env)
    def step(self, action):
        print("Step function called")
        print(action)
        # # Implement the logic for taking a step in the environment
        # observation = self.observation_space.sample()
        # reward = 0.0  # Example reward
        # terminated = False  # Example done condition
        # truncated =  True
        # info = {}
        # return observation, reward, terminated, truncated, info

        # 1. To make sure action is not out of bounds
        assert self.action_space.contains(action)

        # 2. Mapper to convert action to tuples
        action_tuple = self.action2host_mapper[action]
        self.datacenter.containers_model = action_tuple[:,1]

        # TODO: timeste, done, 
        self.timesteps += 1

        # update the location of containers
        self.datacenter.containers_hosts = action_tuple[:, 0]

        # TODO: Make sure that obs are correct with updated hosts, containers and other stats
        obs = self.observation
        info = {}
        terminated = truncated = False
        reward = 0

        assert self.observation_space.contains(obs),\
                (f"observation:\n<{obs}>\n outside of "
                f"observation_space:\n <{self.observation_space}>")


        return obs, reward, terminated, truncated, info
    

    @override(gym.Env)
    def reset(self, seed=None, options=None) -> np.ndarray:
        """Resets the state of the environment and returns
        an initial observation.
        Returns:
            (object): the initial observation.
        Remarks:
            each time resets to a different initial state
        """
        # if self.timestep_reset:
        #     self.global_timestep = 0
        #     self.timestep = 0
        # if self.placement_reset:
        #     self.services_nodes = deepcopy(self.initial_placements)

        # TODO add the details into the info dict
        info = {}
        return self.observation, info

    def _setup_space(self):
        """
        "obs_elements": ["containers_hosts", "hosts_alloc", "hosts_requests", "hosts_usages"]
        sla_items = ['sla_violations']
        containers_requests = ['']

        ["", "", "", "", 
      "", ""]

        """
        ##########################   OBSERVATION     ################################
        #numuber of elements based on the obs in the observation space
        obs_size = 0
        predictions = ['cpu_predictions']  #                   ## Future CPU predictions of each node
        node_resource_size_items = [                           ## CPU and memory resources of each node
            "hosts_resources_alloc",
            "hosts_resources_req",
            "hosts_resources_usage"
        ]
        node_services_size_items = ['containers_request', "containers_usage"]     ## CPU and mem request of each container
        active_containers_accuracies = ['containers_accuracy']
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
            elif elm in active_containers_accuracies:
                obs_size += self.datacenter.num_containers
            elif elm in containers_hosts:
                obs_size += self.datacenter.num_hosts * self.datacenter.num_containers

        higher_bound = 1 
        # generate observation and action spaces
        observation_space = spaces.Box(
            low=0, high=higher_bound, shape=(obs_size, ),
            dtype=np.float64, seed=self._env_seed)
        
        #######################################       ACTION       #################################

        action_space_num = self.datacenter.num_hosts * 2 ## 2 for YOLO model versions

        action_space = spaces.MultiDiscrete(np.ones(self.datacenter.num_containers) * action_space_num,
                                            seed=self._env_seed)

        #######################################      Mappers        ###############################

        cpu_cores = sorted(set(self.datacenter.containers_request[:, 1]))
        host_mapper_dict = Mapper(hosts=self.datacenter.num_hosts, model_versions=self.model_versions, 
                                  unique_cores=cpu_cores)
        # First mapper generates Action int to Tuple(host_id, model_version)
        self.action2host_mapper =  host_mapper_dict._generate_mapping_dict()
        # this mapper maps the Tuple(core, model) to int. This int is the index of YOLO dataset in self.yolodata
        self.core_model_mapper = host_mapper_dict._get_core_model_mapper()

        return observation_space, action_space
    
    def preprocessor(self, obs):
        prep = Preprocessor(self.datacenter.hosts_resources_alloc[:,1:],
                            self.datacenter.containers_request[:,1:])   # TODO define the constructor arguments.
        # TODO: Update preprocessor code to normalize things
        obs = prep.transform(obs)

        #TODO: Make sure all the observation values are in the low and high bounds
        return obs
    
    @property
    def observation(self) -> np.ndarray:
        # 1. create raw observation
        # 2. call preprocessor to normalize the observations
        # 3. return final normalized observations

        # TODO: Incorporate the real usage of nodes coming from YOLO containers and the associated SLA violations
        
        # use it on the mapper
        

        yoloindices = self.yolo_active_indices(self.datacenter.containers_request[:, 1], 
                                               self.datacenter.containers_model, self.core_model_mapper)

        observation = {
            "cpu_predictions": np.tile(self.patch_np_preds[self.timesteps], self.datacenter.core_repeator),
            "hosts_resources_alloc": self.datacenter.hosts_resources_alloc[:,1:],
            "hosts_resources_req": self.datacenter.hosts_resources_req[:,1:],
            "hosts_resources_usage": self.datacenter.hosts_resources_usages(self.timesteps, yoloindices),
            "containers_request": self.datacenter.containers_request[:, 1:],
            "containers_usage": self.datacenter.containers_resources_usage(self.timesteps, yoloindices),#
            "containers_accuracy": self.datacenter.yolo_active_accuracies, 
            "containers_hosts": self.datacenter.containers_hosts
            # "sla_violations":
            # 
        }

        selected = dict(zip(self.obs_elements,
                            [observation[k] for k in self.obs_elements]))
        
        # TODO call the preprocessor with the 'selected'
        obs = self.preprocessor(selected)
        
        print('Working on Observations')
        return obs
        

