# make use of SLA violation, active models accuray average and hosts_resources_usage, num_consolidated
import numpy as np
from typing import Dict, Tuple, Any

""" Need rewar based on consolidation, accuracy and SLA violations
    1. need a function to count the consolidations
    2. already have a function to get the accuracies of active model versions, use it
    3. 
    """

def edgeaibus_reward(
        self, *, 
        num_moves: int,
        num_overloaded: int,
        num_model_switches:int, 
        sla_violations:np.array,
        mean_accuracy:int) -> Tuple[
        float, Dict[str, Any]]:
    
    if num_overloaded > 0:

        reward_illegal = illegal_reward(self, num_overloaded)
        return reward_illegal, {
            "reward_illegal": reward_illegal,
            "reward_consolidation": 0,
            "reward_accuracy": 0,
            "reward_sla":0
            }
    
    energy_reward = reward_energy(self)
    accuracy_reward = reward_accuracy(self, mean_accuracy)
    sla_reward = reward_sla(self, sla_violations)
    total_reward = energy_reward + sla_reward  + accuracy_reward

    return total_reward, {
            "reward_illegal": 0,
            "reward_consolidation": energy_reward,
            "reward_accuracy": accuracy_reward,
            "reward_sla":sla_reward
            }


def rescale(values, old_min = 0, old_max = 1, new_min = 0, new_max = 100):
    output = []
    for v in values:
        new_v = (new_max - new_min) / (old_max - old_min) * (v - old_min) + new_min
        output.append(new_v)
    return np.array(output)


# def reward_energy(self):
#     consolidation_factor = self.datacenter.num_consolidated/ self.datacenter.num_hosts
#     reward_scaled = rescale(
#         values=[consolidation_factor],
#         old_min=self.consolidation_lower, 
#         old_max=self.consolidation_upper,
#         new_min=0, new_max=1)[0]
#     reward = self.penalty_consolidation * reward_scaled
#     return reward

# def reward_energy(self):
#     # Total resources requested and available
#     total_requested_cores = np.sum(self.datacenter.containers_request[:, 1])
#     total_available_cores = np.sum(self.datacenter.hosts_resources_alloc)

#     # Utilization factor based on container requests (for oversubscription)
#     utilization_factor = total_requested_cores / total_available_cores
    
#     # Rescale utilization to the range [0, 1.5] to reward oversubscription
#     utilization_scaled = rescale(
#         values=[utilization_factor],
#         old_min=0.7,  # Start encouraging oversubscription beyond 80% of available cores
#         old_max=1.2,  # Capping oversubscription at 120% based on requests
#         new_min=0, 
#         new_max=1)[0]

#     # Consolidation factor based on number of consolidated hosts
#     consolidation_factor = self.datacenter.num_consolidated / self.datacenter.num_hosts
#     consolidation_scaled = rescale(
#         values=[consolidation_factor],
#         old_min=self.consolidation_lower,
#         old_max=self.consolidation_upper,
#         new_min=0, 
#         new_max=1)[0]
    
#     # Weight both utilization (based on requests) and consolidation in the final reward
#     utilization_weight = 0.55  # Assign more weight to utilization
#     consolidation_weight = 0.45  # Assign weight to consolidation

#     # Final reward calculation with a deduction for actual usage penalties
#     reward = self.penalty_consolidation * (
#         utilization_weight * utilization_scaled + 
#         consolidation_weight * consolidation_scaled
#     )
    
#     return reward

def reward_energy(self):
    # Aggregate actual CPU usage across all hosts
    if self.datacenter.kube:
        total_actual_usage = np.sum(self.datacenter.hosts_resources_usage_gke[:,0])
    elif self.datacenter.simulation:
        total_actual_usage = np.sum(self.datacenter.hosts_resources_usages[:,0])
    total_available_cores = np.sum(self.datacenter.hosts_resources_alloc)

    # Average utilization of the entire datacenter
    avg_usage = total_actual_usage / total_available_cores

    # Scale the average usage to reward high utilization up to 80%
    usage_scaled = rescale(
        values=[avg_usage],
        old_min=self.usage_lower,  # Min usage is 0
        old_max=self.usage_upper,  # Max reward given at 80%
        new_min=0, 
        new_max=1)[0]

    # Consolidation factor based on the number of consolidated hosts
    consolidation_factor = self.datacenter.num_consolidated / self.datacenter.num_hosts
    consolidation_scaled = rescale(
        values=[consolidation_factor],
        old_min=self.consolidation_lower,
        old_max=self.consolidation_upper,
        new_min=0, 
        new_max=1)[0]

    # Weight both average actual usage and consolidation in the final energy reward
    usage_weight = 0.6 # Prioritize high actual usage
    consolidation_weight = 0.4  # Still reward consolidation

    reward = self.penalty_consolidation * (
        usage_weight * usage_scaled + 
        consolidation_weight * consolidation_scaled
    )
    
    return reward

def reward_accuracy(self, mean_accuracy):
    
    rescale_reward = rescale(values=[mean_accuracy],
                             old_min=self.accuracy_lower,
                             old_max=self.accuracy_upper,
                             new_min=0, new_max=1)[0]
    
    reward = self.penalty_accuracy * rescale_reward
    return reward

def reward_sla(self, sla_violations:np.array):

    mean_violations = np.sum(sla_violations)/len(sla_violations)
    rescale_reward = rescale(values=[mean_violations],
                                 old_min=self.sla_lower,
                                 old_max=self.sla_upper,
                                 new_min=0, new_max=1)[0]
    
    reward = self.penalty_sla * rescale_reward
    return reward


def illegal_reward(self, prev_num_overloaded: int):
    """reward for the number of illegal factors
    """
    nodes_overloaded_factor = prev_num_overloaded/self.datacenter.num_hosts
    reward_illegal = self.penalty_illegal * nodes_overloaded_factor
    return reward_illegal