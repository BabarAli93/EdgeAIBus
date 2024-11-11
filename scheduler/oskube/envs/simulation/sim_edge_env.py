import gymnasium as gym
import numpy as np
from scheduler.Scheduler import Scheduler
from scheduler.oskube.utils import (
    Preprocessor,
    override,
    Mapper
)

import random
from gymnasium import spaces
from gymnasium.utils import seeding
from typing import (
    Dict,
    Any,
    List,
    Tuple
)
import types

from scheduler.oskube.envs.rewards import edgeaibus_reward
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
        self.no_action_on_overload = config['no_action_on_overloaded']
        self.workload = config['overload_threshold']
        self.consolidation_upper = config['consolidation_upper']
        self.consolidation_lower = config['consolidation_lower']
        self.accuracy_lower = config['accuracy_lower']
        self.accuracy_upper = config['accuracy_upper']
        self.usage_lower = config['usage_lower']
        self.usage_upper = config['usage_upper']
        self.sla_lower = config['sla_lower']
        self.sla_upper = config['sla_upper']
        self.episode_length = config['episode_length']

        self.penalty_consolidation = config['penalty_consolidation']
        self.penalty_accuracy = config['penalty_accuracy']
        self.penalty_sla = config['penalty_sla']
        self.penalty_illegal = config['penalty_illegal']

        # Seed the environment
        self.seed(config.get('seed'))
        self.datacenter = config['datacenter']
        self.yolodata = self.datacenter.yolo_reading(config['datasets']['yolo_path'])
        self.yolo_random_samples = random.sample(range(1000), self.datacenter.yolo_sample_count)                       # Will be sampling 50 values from Yolo randomly
        self.model_versions: int = 2             # YOLO Nano and Small only
        self.obs_elements: List[str] = config['obs_elements']
        self.edgeaibus_reward = types.MethodType(edgeaibus_reward, self)
        
        self.timesteps:int = 0
        self.global_timesteps:int = 0
        self.hard_reset = False
        self.initial_placements = deepcopy(self.datacenter.containers_hosts)
        self.initial_model = deepcopy(self.datacenter.containers_model)
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
        #print("Step function called")
        #print(action)
        # reward = 0.0  # Example reward
        # terminated = False  # Example done condition
        # truncated =  True
        # info = {}

        #print(self.global_timesteps)

        # Handeled the Overloading Scenario
        prev_placement = deepcopy(self.datacenter.containers_hosts)
        prev_models = deepcopy(self.datacenter.containers_model)

        # 1. To make sure action is not out of bounds
        assert self.action_space.contains(action)

        # 2. Mapper to convert action to tuples
        action_tuple = self.action2host_mapper[action]

        # Updating the model version and the container's locations
        self.datacenter.containers_model = action_tuple[:,1]
        # update the location of containers
        self.datacenter.containers_hosts = action_tuple[:, 0]

        # TODO: timeste, done, 
        self.timesteps += 1
        self.global_timesteps += 1

        # call the 'hosts_resources_requested_sim'. It gets the updated resource requested 
        self.datacenter.hosts_resources_req = self.datacenter.hosts_resources_requested_sim

        # TODO: use it correctly. pay attention to property functions
        self.yolo_random_samples = random.sample(range(1000), self.datacenter.yolo_sample_count)

        # Need updated actual usage to make sure it does not exceed the overload threshold of 80%
        yoloindices = self.yolo_active_indices(self.datacenter.containers_request[:, 1], 
                                               self.datacenter.containers_model, self.core_model_mapper)
                                               
        self.datacenter.mean_yolo_sample_usages(self.yolo_random_samples, yoloindices)

        num_overloaded = self.datacenter.num_overloaded

        if self.no_action_on_overload and num_overloaded > 0:
            print('Overload Caution! Reverting back...')
            self.datacenter.containers_hosts = prev_placement
            self.datacenter.containers_model = prev_models

        num_moves = len(np.where(self.datacenter.containers_hosts != prev_placement)[0])
        num_model_switchings = len(np.where(self.datacenter.containers_model != prev_models)[0])
        sla_violations = self.datacenter.yolo_sla(self.yolo_random_samples, yoloindices)
        mean_slavr = np.sum(sla_violations)/len(sla_violations)
        accuracy = self.datacenter.yolo_sample_accuracies
        mean_accuracy = np.sum(accuracy)/ len(accuracy)
        mean_cluster_util = self.datacenter.cluster_mean_usage_percentage
        # TODO: append the conserved cores
        conseve_reso = self.datacenter.conserved_resources

        print(accuracy)

        reward, rewards = self.edgeaibus_reward(
            num_moves = num_moves,
            num_overloaded = num_overloaded,
            num_model_switches = num_model_switchings,
            sla_violations = sla_violations,
            mean_accuracy = mean_accuracy
        )

        info = {'timestep': self.timesteps,
                'global_timestep': self.global_timesteps,
                'num_consolidated': self.datacenter.num_consolidated,
                'num_overloaded': num_overloaded,
                'num_moves': num_moves,
                "mean_accuracy": mean_accuracy,
                "num_model_switches": num_model_switchings,
                "num_slav": mean_slavr,                     
                'total_reward': reward,
                'rewards': rewards,
                'cpu_conserved_cost': conseve_reso[0],
                "mean_cluster_cpu_util": mean_cluster_util[0],
                "mean_cluster_mem_util": mean_cluster_util[1],
                "oversub_cores":self.datacenter.oversubscribed_cores,
                'seed': self.base_env_seed}

        # TODO: Make sure that obs are correct with updated hosts, containers and other stats
        obs = self.observation
        obs = obs.astype(np.float32)

        assert self.observation_space.contains(obs),\
                (f"observation:\n<{obs}>\n outside of "
                f"observation_space:\n <{self.observation_space}>, Container Placement is: {self.datacenter.containers_hosts}"
                f"Nodes Usgae: {self.datacenter.hosts_resources_usages}")

        return obs, reward, self.done, False, info
    

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
        if self.global_timesteps == 0:
            self.datacenter.containers_hosts = deepcopy(self.initial_placements)
            self.datacenter.containers_model = deepcopy(self.initial_model)

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

        higher_bound = 5
        # generate observation and action spaces
        observation_space = spaces.Box(
            low=0, high=higher_bound, shape=(obs_size, ),
            dtype=np.float32, seed=self._env_seed)
        
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

        yoloindices = self.yolo_active_indices(self.datacenter.containers_request[:, 1], 
                                               self.datacenter.containers_model, self.core_model_mapper)
        
        self.datacenter.mean_yolo_sample_usages(self.yolo_random_samples, yoloindices)

        # self.pred_repeat_handler

        observation = {
            "cpu_predictions": np.tile(self.patch_np_preds[self.global_timesteps], self.datacenter.core_repeator), 
            "hosts_resources_alloc": self.datacenter.hosts_resources_alloc[:,1:],
            "hosts_resources_req": self.datacenter.hosts_resources_requested_sim,
            "hosts_resources_usage": self.datacenter.hosts_resources_usages,    ## usage oriented, Remove the need of timestep
            "containers_request": self.datacenter.containers_request[:, 1:],
            "containers_usage": self.datacenter.containers_resources_usage,     ## usage oriented, Remove the need of timestep
            "containers_accuracy": self.datacenter.yolo_sample_accuracies,                                   ## usage oriented, Remove the need of timestep
            "containers_hosts": self.datacenter.containers_hosts,
            "sla_violations": self.datacenter.yolo_sla(self.yolo_random_samples, yoloindices)                          ## usage oriented, Remove the need of timestep
        }

        selected = dict(zip(self.obs_elements,
                            [observation[k] for k in self.obs_elements]))
        #print(f"Nodes Requested: \n {self.datacenter.hosts_resources_requested_sim}")
        # TODO call the preprocessor with the 'selected'
        obs = self.preprocessor(selected)
        if self.global_timesteps == 0:
            print(obs)
        print(self.datacenter.hosts_resources_requested_sim)
        print(self.datacenter.hosts_resources_usages)
        return obs
    
    @property
    def done(self) -> bool:
        episode = False
        
        if self.timesteps % self.episode_length == 0:
            episode = True
            self.timesteps = 0
        else:
            episode = False

        if self.global_timesteps >= 7800:
            episode = self.hard_reset = True
            self.timesteps = self.global_timesteps = 0
        
        return episode
        

