from ray.rllib.algorithms.callbacks import DefaultCallbacks
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np

from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.annotations import (
    OldAPIStack,
    override,
    OverrideToImplementCustomLogic,
    PublicAPI,
)
from ray.rllib.utils.typing import AgentID, EnvType, EpisodeType, PolicyID

class CustomCallbacks(DefaultCallbacks):

    def on_episode_start(
            self, 
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: EpisodeV2,
            env_index: Optional[int]= None,
            **kwargs,
     ) -> None:
        # timestep 
        episode.user_data["timestep"] = []
        episode.hist_data["timestep"] = []

        # global timestep
        episode.user_data["global_timestep"] = []
        episode.hist_data["global_timestep"] = []

        # Consolidation counts
        episode.user_data["num_consolidated"] = []
        episode.hist_data["num_consolidated"] = []

        # Consolidation counts
        episode.user_data["num_overloaded"] = []
        episode.hist_data["num_overloaded"] = []

        episode.user_data["num_moves"] = []
        episode.hist_data["num_moves"] = []

        episode.user_data["mean_accuracy"] = []
        episode.hist_data["mean_accuracy"] = []

        episode.user_data["num_slav"] = []
        episode.hist_data["num_slav"] = []

        #
        episode.user_data["num_model_switches"] = []
        episode.hist_data["num_model_switches"] = []
        
        episode.user_data["rewards"] =[]
        episode.hist_data["rewards"] = []

        episode.user_data["cpu_conserved_cost"] =[]
        episode.hist_data["cpu_conserved_cost"] = []

        episode.user_data["oversub_cores"] = []
        episode.hist_data["oversub_cores"] = []

        episode.user_data["mean_cluster_cpu_util"] =[]
        episode.hist_data["mean_cluster_cpu_util"] = []

        episode.user_data["mean_cluster_mem_util"] =[]
        episode.hist_data["mean_cluster_mem_util"] = []


    def on_episode_step(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            episode: EpisodeV2,
            env_index: Optional[int] = None,
            **kwargs,
    ):
        
        if type(episode._last_infos) == dict:
            agentID = episode.get_agents()[0]

            timestep = episode._last_infos[agentID]['timestep']
            episode.user_data["timestep"].append(timestep)

            global_timestep = episode._last_infos[agentID]["global_timestep"]
            episode.user_data["global_timestep"].append(global_timestep)

            num_consolidated = episode._last_infos[agentID]["num_consolidated"]
            episode.user_data["num_consolidated"].append(num_consolidated)

            num_overloaded = episode._last_infos[agentID]["num_overloaded"]
            episode.user_data["num_overloaded"].append(num_overloaded)

            num_moves = episode._last_infos[agentID]["num_moves"]
            episode.user_data["num_moves"].append(num_moves)
            
            rewards = episode._last_infos[agentID]['rewards']
            episode.user_data["rewards"].append(rewards)

            mean_acc = episode._last_infos[agentID]["mean_accuracy"]
            episode.user_data["mean_accuracy"].append(mean_acc)
            
            num_switches = episode._last_infos[agentID]["num_model_switches"]
            episode.user_data["num_model_switches"].append(num_switches)
            
            num_violations = episode._last_infos[agentID]["num_slav"]
            episode.user_data["num_slav"].append(num_violations)

            cpu_conserved_cost = episode._last_infos[agentID]["cpu_conserved_cost"]
            episode.user_data["cpu_conserved_cost"].append(cpu_conserved_cost)

            oversub_cores = episode._last_infos[agentID]["oversub_cores"]
            episode.user_data["oversub_cores"].append(oversub_cores)
            
            mean_cluster_cpu_util = episode._last_infos[agentID]["mean_cluster_cpu_util"]
            episode.user_data["mean_cluster_cpu_util"].append(mean_cluster_cpu_util)

            mean_cluster_mem_util = episode._last_infos[agentID]["mean_cluster_mem_util"]
            episode.user_data["mean_cluster_mem_util"].append(mean_cluster_mem_util)



    def on_episode_end(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: EpisodeV2,
            env_index: Optional[int] = None,
            **kwargs,
    ):
        # extracting metrics at the end of episode
        num_consolidated_avg = np.mean(episode.user_data["num_consolidated"])
        num_overloaded_avg = np.mean(episode.user_data["num_overloaded"])
        num_moves_avg = np.mean(episode.user_data["num_moves"])
        mean_accuracy_avg = np.mean(episode.user_data["mean_accuracy"])
        num_model_switches_avg = np.mean(episode.user_data["num_model_switches"])
        num_slav_avg = np.mean(episode.user_data["num_slav"])
        cpu_cost_avg = np.mean(episode.user_data["cpu_conserved_cost"])
        oversub_cores_avg = np.mean(episode.user_data["oversub_cores"])
        mean_cluster_cpu_util_avg = np.mean(episode.user_data["mean_cluster_cpu_util"])
        mean_cluster_mem_util_avg = np.mean(episode.user_data["mean_cluster_mem_util"])
        
        timestep = np.max(episode.user_data["timestep"])
        global_timestep = np.max(episode.user_data["global_timestep"])

        # extract episodes rewards info
        episode_reward_consolidation = [a[
            'reward_consolidation']for a in episode.user_data[
                "rewards"]]
        episode_reward_illegal = [a[
            'reward_illegal']for a in episode.user_data[
                "rewards"]]
        episode_reward_accuracy = [a[
            'reward_accuracy']for a in episode.user_data[
                "rewards"]]
        episode_reward_sla = [a[
            'reward_sla']for a in episode.user_data[
                "rewards"]]
        
        reward_consolidation_mean = np.mean(episode_reward_consolidation)
        reward_illegal_mean = np.mean(episode_reward_illegal)
        reward_accuracy_mean = np.mean(episode_reward_accuracy)
        reward_sla_mean = np.mean(episode_reward_sla)

        reward_consolidation_sum = np.sum(episode_reward_consolidation)
        reward_illegal_sum = np.sum(episode_reward_illegal)
        reward_accuracy_sum = np.sum(episode_reward_accuracy)
        reward_sla_sum = np.sum(episode_reward_sla)

        # add custom metrics to tensorboard
        episode.custom_metrics['num_consolidated'] = num_consolidated_avg
        episode.custom_metrics['num_overloaded'] = num_overloaded_avg
        episode.custom_metrics['num_moves'] = num_moves_avg
        episode.custom_metrics['mean_accuracy'] = mean_accuracy_avg
        episode.custom_metrics['num_model_switches'] = num_model_switches_avg
        episode.custom_metrics['num_slav'] = num_slav_avg
        episode.custom_metrics["cpu_conserved_cost"] = cpu_cost_avg
        episode.custom_metrics['mean_cluster_cpu_util'] = mean_cluster_cpu_util_avg
        episode.custom_metrics['mean_cluster_mem_util'] = mean_cluster_mem_util_avg
        episode.custom_metrics["oversub_cores"] = oversub_cores_avg

        # Mean rewards
        episode.custom_metrics['reward_consolidation_mean'] = reward_consolidation_mean
        episode.custom_metrics['reward_illegal_mean'] = reward_illegal_mean
        episode.custom_metrics['reward_accuracy_mean'] = reward_accuracy_mean
        episode.custom_metrics['reward_sla_mean'] = reward_sla_mean

        # Sum Rewards
        episode.custom_metrics['reward_consolidation_sum'] = reward_consolidation_sum
        episode.custom_metrics['reward_illegal_sum'] = reward_illegal_sum
        episode.custom_metrics['reward_accuracy_sum'] = reward_accuracy_sum
        episode.custom_metrics['reward_sla_sum'] = reward_sla_sum
        episode.custom_metrics['timestep'] = timestep
        episode.custom_metrics['global_timestep'] = global_timestep
