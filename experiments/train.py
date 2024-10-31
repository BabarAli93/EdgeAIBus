import os
import click
import json
import pprint
from copy import deepcopy
import sys
import ray
from ray import tune
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.air import CheckpointConfig, RunConfig
from utils import CustomCallbacks
#from scheduler.Scheduler import Scheduler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datacenter.Datacenter import * # TODO: update this list to import only reuqired functions

from utils.constants import (
    CONFIGS_PATH,
    SCHEDULER_PATH,
    TRAIN_RESULTS_PATH,
    DATASETS_PATH,
    RAY_LOGS_PATH
)
from utils.class_builder import make_env_class

# def impala_builder(env_class, env_config, use_callback:bool):
#     config = ImpalaConfig()

#     if use_callback:
#         config.callbacks(CustomCallbacks)
    
#     config = (
#         config
#         .environment(env=env_class, env_config=env_config)
#         .training(vtrace=True, vtrace_clip_rho_threshold=2, vtrace_clip_pg_rho_threshold=2,
#                   gamma=0.95,
#                   lr=0.0001,
#                   model={"fcnet_hiddens": [128, 128], "fcnet_activation": "linear", "vf_share_layers": "true"},
#                   train_batch_size=512,
#                   vf_loss_coeff=0.01,
#                   replay_proportion=0.3,
#                   replay_buffer_num_slots=300,
#                   train_batch_size_per_learner=256
#                   )
#         .debugging(log_level="INFO")
#         .resources(num_gpus=0)
#         .env_runners(num_env_runners=3)
#         #.rollouts(num_rollout_workers=2)
#     )
#     config.seed = 203
#     config.buffer_size = 2048

#     return config.to_dict()

def training(config_file, type_env, use_callback, checkpoint_freq):
    generator_config = deepcopy(config_file)
    del generator_config['notes']
    bitbrains_path = os.path.join(DATASETS_PATH, "bitbrains/rnd")
    yolo_path = os.path.join(DATASETS_PATH, "yolo")
    stop = config_file['stop']
    run_or_experiment = config_file['run_or_experiment']

    generator_config.update({'type_env': type_env})
    datacenter = DatacenterGeneration(generator_config)
    #scheduler = Scheduler() 

    env_config = generator_config['env_config_base']
    env_config.update({
        #"train": True,
        'datacenter': datacenter,
        'datasets': {
            'bitbrains_path': bitbrains_path,
            'yolo_path': yolo_path
        },
        'scheduler_path': SCHEDULER_PATH
    })

    # if type_env not in ['CartPole-v0', 'Pendulum-v0']:
    #     ray_config = {"env": make_env_class(type_env),
    #                 "env_config": env_config}
    # else:
    #     ray_config = {"env": type_env}

    # if run_or_experiment == 'IMPALA':
    #     learn_config = impala_builder(ray_config['env'], env_config, use_callback)
    
    learn_config = generator_config['learn_config']
    learn_config.update({
        "env": make_env_class(type_env),
        "env_config": env_config
    })
    if use_callback: 
        learn_config.update({
            "callbacks": CustomCallbacks

        })
    
    ray.init(local_mode=True)

    tuner = tune.Tuner(
        run_or_experiment,
        run_config=RunConfig(stop=stop,
                             storage_path=RAY_LOGS_PATH,
                             verbose=1,
                             checkpoint_config=CheckpointConfig(
                                 num_to_keep=5,
                                 checkpoint_frequency=100,
                                 checkpoint_at_end=True,
                             )
                             ),
        param_space=learn_config,
    ).fit()

    return None



@click.command()
@click.option('--config-file', type=str, default='datacenter_sim')
@click.option('--type-env', required=True,
             type=click.Choice(['sim-edge']),
             default='sim-edge')
@click.option('--use-callback', required=True, type=bool, default=True)
@click.option('--checkpoint-freq', required=False, type=int, default=100)

def main(config_file: str, type_env: str, use_callback: bool, checkpoint_freq: int):
    """[summary]

    Args:
        config_file (str): name of the config folder (only used in real mode)
        use_callback (bool): whether to use callbacks or storing and visualising
        checkpoint_freq (int): checkpoint the ml model at each (n-th) step
        type_env (str): the type of the used environment.
    """
    
    config_file_path = os.path.join(
        CONFIGS_PATH, f"{config_file}.json")
    with open(config_file_path) as cf:
        config = json.loads(cf.read())

    pp = pprint.PrettyPrinter(indent=4)
    print('start experiments with the following config:\n')
    pp.pprint(config)

    training(config, type_env, use_callback, checkpoint_freq)



if __name__ == "__main__":
    main()