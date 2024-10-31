"""
Testing phase of the experiments on the test data
based-on:
https://github.com/ray-project/ray/issues/9123
https://github.com/ray-project/ray/issues/7983
"""
import os
import sys
import pickle
import click
from typing import Dict, Any
import json
from copy import deepcopy
import ray
import pprint
import gymnasium as gym
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.algorithms.algorithm import Algorithm
import pandas as pd
from pprint import PrettyPrinter


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datacenter.Datacenter import *

pp = PrettyPrinter(indent=4)
# get an absolute path to the directory that contains parent files

from utils.constants import(
    TRAIN_RESULTS_PATH, 
    TESTS_RESULTS_PATH,
    DATASETS_PATH,
    SCHEDULER_PATH,
    ENVSMAP
)

from utils.class_builder import make_env_class

def run_experiments(
    *, config_file, type_env: str, local_mode: bool,
    episode_length, num_episodes: int, checkpoint_to_load: str, 
    num_containers:int):
    """
    """


    experiment_folder = os.path.join(TRAIN_RESULTS_PATH, 
                               "IMPALA_{}".format(num_containers))

    config_path = os.path.join(experiment_folder, 
                               f"{config_file}.json")

    with open(config_path) as cf:
        config_file = json.loads(cf.read())
        
    generator_config = deepcopy(config_file)
    del generator_config['notes']
    bitbrains_path = os.path.join(DATASETS_PATH, "bitbrains/rnd")
    yolo_path = os.path.join(DATASETS_PATH, "yolo")
    generator_config.update({'type_env': type_env})
    datacenter = DatacenterGeneration(generator_config)
    env_config = generator_config['env_config_base']
    env_config.update({
        "episode_length": episode_length,
        'datacenter': datacenter,
        'datasets': {
            'bitbrains_path': bitbrains_path,
            'yolo_path': yolo_path
        },
        'scheduler_path': SCHEDULER_PATH
    })

    learn_config = generator_config['learn_config']
    path_env = type_env if type_env != 'kube-edge' else 'sim-edge'
    ray.init(local_mode=local_mode)

    items = os.listdir(experiment_folder)
    folders = [item for item in items if os.path.isdir(os.path.join(experiment_folder, item))]
    experiment_str = folders[0]
    
    print('Environment Configurations:')
    pp.pprint(env_config)


    ## Hit it with different state, train vs test gym.make initiate the environment
    if type_env not in ['CartPole-v0', 'Pendulum-v0']:
        env = gym.make(ENVSMAP[type_env], config=env_config)
        # reset the env at the beginning of each episode
        ray_config = {"env": make_env_class('sim-edge'),
                    "env_config": env_config}
        ray_config.update(learn_config)
    else:
        ray_config = {"env": type_env}
        ray_config.update(learn_config)

    if checkpoint_to_load=='last':
        checkpoint_string = sorted([
            s for s in filter (
                lambda x: 'checkpoint' in x, os.listdir(
                    os.path.join(
                        experiment_folder, experiment_str)))])[-1]
        checkpoint = int(checkpoint_string.replace('checkpoint_',''))
        checkpoint_path = os.path.join(
            experiment_folder,
            experiment_str,
            checkpoint_string
        )
        checkpoint_to_load_info = checkpoint
    else:
        checkpoint_path = os.path.join(
            experiment_folder,
            experiment_str,
            # os.listdir(experiments_folder)[0],
            f"checkpoint_{checkpoint_to_load}",
            f"checkpoint-{int(checkpoint_to_load)}"
        )
        checkpoint_to_load_info = int(checkpoint_to_load)
        
    episodes = []
    agent = Algorithm.from_checkpoint(checkpoint_path)

    for i in range(0, num_episodes):
        print(f"---- \nepisode: {i} ----\n")
        episode_reward = 0
        done = truncate = False
        states = []
        obs, info = env.reset()
        # print(f"observation: {env.env.raw_observation}")
        # start = time.time()
        while not done:
            action = agent.compute_single_action(obs)
            obs, reward, done, truncate, info = env.step(action)
            state = flatten(env.observation, action, reward, info)
            states.append(state)
            episode_reward += reward
        states = pd.DataFrame(states)
        print(f"episode reward: {episode_reward}")
        episodes.append(states)

    info = {
        'type_env': type_env,
        'checkpoint': checkpoint_to_load_info,
        'experiment_str': experiment_str,
        'episode_length': episode_length,
        'num_episodes': num_episodes,
        'algorithm': generator_config['run_or_experiment'],
        'penalty_accuracy': env_config['penalty_accuracy'],
        'penalty_sla': env_config['penalty_sla'],
        'penalty_consolidation': env_config['penalty_consolidation'],
        'num_workers': learn_config['num_env_runners']
    }
    # make the new experiment folder
    test_series_path = os.path.join(
        TESTS_RESULTS_PATH, generator_config['run_or_experiment'],
        'containers', str(num_containers),
        'tests')
    if not os.path.isdir(test_series_path):
        os.makedirs(test_series_path)
    content = os.listdir(test_series_path)
    new_test = len(content)
    this_test_folder = os.path.join(test_series_path,
                                    str(new_test))
    os.makedirs(this_test_folder)

    states.to_csv(os.path.join(this_test_folder, 'states.csv'))
    # episodes = np.array(episodes)
    # flattened_episodes = episodes.reshape(-1, episodes.shape[-1])
    # episodes_df = pd.DataFrame(flattened_episodes)
    # episodes_df.to_csv(os.path.join(this_test_folder, 'episodes.csv'))

    # save the necesarry information
    with open(os.path.join(this_test_folder, 'info.json'), 'x') as out_file:
        json.dump(info, out_file, indent=4)
        json.dump(config_file, out_file, indent=4)
    with open(os.path.join(
        this_test_folder, 'episodes.pickle'), 'wb') as out_pickle:
        pickle.dump(episodes, out_pickle)

def flatten(raw_obs, action, reward, info):
    return {
        'action': action,
        #'services_nodes': raw_obs['services_nodes'],
        #'users_stations': raw_obs['users_stations'],
        'num_consolidated': info['num_consolidated'],
        'num_moves': info['num_moves'],
        'num_overloaded': info['num_overloaded'],
        'mean_accuracy': info['mean_accuracy'],
        "num_model_switches": info["num_model_switches"],
        "num_slav": info['num_slav'],
        "cpu_conserved_cost": info["cpu_conserved_cost"],
        "mean_cluster_cpu_util": info["mean_cluster_cpu_util"],
        "mean_cluster_mem_util": info["mean_cluster_mem_util"],
        "oversub_cores": info['oversub_cores'],
        'reward_sla': info['rewards']['reward_sla'],
        'reward_accuracy': info['rewards']['reward_accuracy'],
        'reward_illegal': info['rewards']['reward_illegal'],
        'reward_consolidation': info['rewards']['reward_consolidation'],
        'reward': reward
    }



@click.command()
@click.option('--config-file', type=str, default='datacenter_sim')
@click.option('--local-mode', type=bool, default=True)
@click.option('--num_containers', required= True, type=int, default=6)
@click.option('--type-env', required=True,
              type=click.Choice(['sim-edge', 'kube-edge']),
              default='sim-edge')
@click.option('--episode-length', required=False, type=int, default=1000)
@click.option('--num-episodes', required=False, type=int, default=1)
@click.option('--checkpoint-to-load', required=False, type=str, default='last')

def main(config_file: str, local_mode: bool, type_env: str,
         num_episodes: int, episode_length: int, num_containers:int,
         checkpoint_to_load: str):
    """[summary]
    Args:
        local_mode (bool): run in local machine
        type_env (str): the type of the used environment
        checkpoint (int): training checkpoint to load. There can be a total of 5 checkpoints in directory
        episode-length (int): number of steps in the test episode. Training has 100 steps length
        num_episodes (int): the number of episodes to run
    """
    run_experiments(config_file=config_file,type_env=type_env,
        num_episodes=num_episodes, episode_length=episode_length,
        local_mode=local_mode, num_containers = num_containers,
        checkpoint_to_load=checkpoint_to_load)
    

if __name__ == "__main__":
    main()