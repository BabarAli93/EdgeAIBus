import numpy as np
import random
from datacenter import *

class Deployment():
    def __init__(self, num_hosts, num_containers, hosts_resources_reamin, hosts_resources_req, container_conf):

        self.num_hosts = num_hosts 
        self.num_containers = num_containers 
        self.hosts_resources_reamin = hosts_resources_reamin
        self.hosts_resources_req = hosts_resources_req
        self.container_conf = container_conf
        self.containers_hosts = self.init_deployment()

        return None
    
    def init_deployment(self):

        tries_limit = 100
        for try_id in range(tries_limit):
            nodes = list(np.arange(self.num_nodes))
            random.shuffle(nodes)
            popped_nodes = []
            node_id = nodes.pop()
            popped_nodes.append(node_id)
            for service_id in range(self.num_services):
                try:
                    # iterate through the currently popped nodes
                    # remmaining resources
                    # and check whether it is possible to fit the
                    # current service inside it
                    nodes_sorted = [node for _, node in
                                    sorted(zip(
                                        self.nodes_resources_available_frac_avg[popped_nodes],
                                        popped_nodes))]
                    for node in nodes_sorted:
                        if np.alltrue(self.services_resources_request[service_id] <
                                      self.nodes_resources_available[node]):
                            self.services_nodes[service_id] = node
                            break
                    else:  # no-break
                        node_id = nodes.pop()
                        popped_nodes.append(node_id)
                        self.services_nodes[service_id] = node_id
                except IndexError:
                    if try_id < tries_limit - 1:
                        break
                    else:
                        raise RuntimeError((f"tried <{tries_limit}> times but "
                                            "couldn't allocate services to"
                                            "node try eiher smaller range for"
                                            " services or larger range for"
                                            "nodes"))

        return None