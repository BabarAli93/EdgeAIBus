import numpy as np
from typing import Dict
#from edgeaibus.utils import rounding
from scheduler.edgeaibus.utils import rounding

# TODO complete based on the new needs

class Preprocessor():
    def __init__(self, cluster_hosts_allocatable: np.ndarray,
                 containers_resources_request: np.ndarray,):
        self.hosts_resources_alloc = cluster_hosts_allocatable
        self.containers_resources_request = containers_resources_request
        self.num_hosts = cluster_hosts_allocatable.shape[0]
        #self.max_services_nodes = max_services_nodes

    @rounding
    def transform(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        transform the input observation as the dictionary
        sends each key of the dictionary to the approperiate preprocessor
        and returns the concatenated and flattened numpy array
        """
        obs = np.array([])
        transformers = {
            "cpu_predictions": self._percentage_normalizer,
            "hosts_resources_alloc": self._hosts_normalizer,
            "hosts_resources_req": self._hosts_normalizer,
            "hosts_resources_usage": self._hosts_normalizer,
            "containers_request": self._services_request_normalizer,
            "containers_usage": self._services_request_normalizer,
            "containers_accuracy": self._percentage_normalizer,
            "containers_hosts": self._one_hot_containers_nodes,
            "sla_violations":self._none
        }
        for key, val in observation.items():
            obs = np.concatenate((obs, transformers.get(
                key, self._invalid_operation)(val).flatten()))
        return obs

    def _services_request_normalizer(self, obs: np.ndarray) -> np.ndarray:
        """
        divides the largest available of each resource by the
        capacity of the largest size of that resource in the cluster
        in any service
        e.g for ram:
            ram_usage_of_a_service / largest_ram_capacity_of_any_conrainer
        """
        lst = []
        for index in range(self.containers_resources_request.shape[1]):
            lst.append(max(self.containers_resources_request[:, index]))
        return obs/lst

    def _hosts_normalizer(self, obs: np.ndarray) -> np.ndarray:
        """
        divides the largest available of each resource by the
        capacity of the largest size of that resource in the cluster
        in any node
        e.g for ram:
            ram_usage_of_a_node / largest_ram_capacity_of_any_node
        """
        lst = []
        for index in range(self.hosts_resources_alloc.shape[1]):
            lst.append(max(self.hosts_resources_alloc[:, index]))
        return obs/lst
    
    def _percentage_normalizer(self, obs: np.ndarray) -> np.ndarray:
        """
            dividing CPU predicted usage by 100 to convert to [0,1]
            """
                
        return obs/100

    def _none(self, obs: np.ndarray) -> np.ndarray:
        return obs

    def _one_hot_containers_nodes(
        self, obs: np.ndarray) -> np.ndarray:
        """
        one hot encoding of the containers_hosts
        e.g in a cluster of 2 nodes and 4 services:
            [0, 1, 1, 0]
        results in:
            [0, 0, 0, 1, 0, 1, 0, 0]
        """

        obs_prep = np.array([])
        for host in obs:
            one_hot_encoded = np.zeros(self.num_hosts)
            one_hot_encoded[host] = 1
            obs_prep = np.concatenate((obs_prep, one_hot_encoded))
        return obs_prep

    def _invalid_operation(self, obs: np.ndarray) -> None:
        raise ValueError(f"invalid observation: <{obs}>")
