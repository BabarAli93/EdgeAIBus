import numpy as np
import warnings

from scheduler.edgeaibus.utils import (
    Mapper
)
from copy import deepcopy

class MSB():
    def __init__(self, datacenter: str):
        self.datacenter = datacenter
        cpu_cores = sorted(set(self.datacenter.containers_request[:, 1]))
        host_mapper_dict = Mapper(hosts=self.datacenter.num_hosts, model_versions=len(self.datacenter.yolo_accuracies), 
                                  unique_cores=cpu_cores)
        self.tuple2action_mapper = host_mapper_dict._generate_tuple_to_int_mapping()

    def compute_single_action(self, obs: np.ndarray):
        # last number of container size values are SLA violations of previous decsion
        # it will be a MS + balanced GKE scheduler
        """
        1. for indvidual container, 10% SLA step down model, 1% step up
        2. for each pod, filter the nodes using the requested resources, sort in the descending order of remaining CPU
        3. sorting in asecing order will increase utilization=> optimized
        4. after pod placement, reiterate for next container
        """
        # make local copies of requested, remaining, alloc and capacity

        hosts_resources_remain_local = deepcopy(self.datacenter.hosts_resources_remain)
        hosts_resources_req_local = deepcopy(self.datacenter.hosts_resources_req)

        prev_iter_slav = obs[-self.datacenter.num_containers:]
        placement = deepcopy(self.datacenter.containers_hosts)
        model = deepcopy(self.datacenter.containers_model)
        counter = 0
        for container in range(self.datacenter.num_containers):
            
            # TODO: DO THIS IF THERE IS SLA VIOLATION OTHERWISE NO ACTION
            model_changed = False
            if prev_iter_slav[container] >= 0.15 and model[container] != 0:                         # 10% SLA violation => Step down
                model[container] = 0
                model_changed = True
            elif prev_iter_slav[container] < 0.05 and model[container] != 1:                          # 1% SLA violation => Step Up
                model[container] = 1
                model_changed = True

            if model_changed:
                counter += 1
                # filering nodes to get the ones that can accommodate this container
                filtered_nodes = hosts_resources_remain_local[np.all(hosts_resources_remain_local[:, 1:] >= 
                                                                           self.datacenter.containers_request[container, 1:], axis=1)]
                if len(filtered_nodes) < 1:
                    warnings.warn("No node is available! Keeping same placemets and models!")
                    model[container] = self.datacenter.containers_model[container]
                    continue

                # Now scoring => sort in descending order for balanced utilization
                indices = np.argsort(-hosts_resources_remain_local[:,1])
                scored_nodes = hosts_resources_remain_local[indices]
                if np.any(scored_nodes[:, 1:] < 0):
                    raise ValueError("There is a negative value in the second column.")

                # pick the top node to place the container
            
                prev_node_id = self.datacenter.containers_hosts[container]
                new_node_id = indices[0]

                placement[container] = new_node_id
            # update resources for next pod
                hosts_resources_remain_local[prev_node_id, 1:] += self.datacenter.containers_request[container, 1:] 
                hosts_resources_req_local[prev_node_id,:] -= self.datacenter.containers_request[container, 1:] 

                hosts_resources_remain_local[new_node_id, 1:] -= self.datacenter.containers_request[container, 1:]
                hosts_resources_req_local[new_node_id, :] += self.datacenter.containers_request[container, 1:]

        print(f'counter: {counter}')
        action = np.array(list(zip(placement, model)), dtype=object)
        action = np.array([self.tuple2action_mapper[tuple(item)] for item in action], dtype=np.int32)
        return action
         



