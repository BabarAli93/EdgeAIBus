import os
import click
import json
import pprint
from copy import deepcopy
import sys
import ray
from ray import tune

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datacenter.Datacenter import * # TODO: update this list to import only reuqired functions

from utils.constants import (
    CONFIGS_PATH,
    TRAIN_RESULTS_PATH,
    DATASETS_PATH
)
from utils.class_builder import make_env_class

def training(config_file, type_env, use_callback, checkpoint_freq):
    generator_config = deepcopy(config_file)
    del generator_config['notes']
    bitbrains_path = os.path.join(DATASETS_PATH, "bitbrains/rnd")
    yolo_path = os.path.join(DATASETS_PATH, "yolo")

    generator_config.update({
        'type_env': type_env, 
        'use_callback': use_callback,
        'checkpoint_freq': checkpoint_freq,
        'datasets': {
            'bitbrains_path': bitbrains_path,
            'yolo_path': yolo_path
        }
    })

    datacenter = DatacenterGeneration(generator_config)

    env_config = generator_config['env_config_base']
    env_config.update({
        'datacenter': datacenter
    })

    if type_env not in ['CartPole-v0', 'Pendulum-v0']:
        ray_config = {"env": make_env_class(type_env),
                    "env_config": env_config}
    else:
        ray_config = {"env": type_env}

    learn_config = generator_config['learn_config']
    ray_config.update(learn_config)
    
    print(ray_config)
    
    ray.init(debug=True)
    _ = tune.run(config=ray_config,
                 run_or_experiment="IMPALA",
                 resume=False)

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