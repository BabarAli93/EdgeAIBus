import numpy as np
import warnings

from scheduler.edgeaibus.utils import (
    Mapper
)
from copy import deepcopy

class GKO():
    def __init__(self, datacenter: str):
        self.datacenter = datacenter
        cpu_cores = sorted(set(self.datacenter.containers_request[:, 1]))
        host_mapper_dict = Mapper(hosts=self.datacenter.num_hosts, model_versions=len(self.datacenter.yolo_accuracies), 
                                  unique_cores=cpu_cores)
        self.tuple2action_mapper = host_mapper_dict._generate_tuple_to_int_mapping()

    def compute_single_action(self, obs: np.ndarray):
        """
        1. Optimized GKE scheduler
        2. No action as the containers are already placed in binpacking mode
        """
        action = np.array(list(zip(self.datacenter.containers_hosts, self.datacenter.containers_model)), dtype=object)
        action = np.array([self.tuple2action_mapper[tuple(item)] for item in action], dtype=np.int32)
        return action
