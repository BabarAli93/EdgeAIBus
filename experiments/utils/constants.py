import os
from scheduler.oskube.envs import SimEdgeEnv

# dfined by the user

import pathlib

p = pathlib.Path(__file__)

PROJECT_PATH = "/home/babarali/OSKube"
# DATA_PATH = "/data"

# generated baesd on the users' path
DATA_PATH = os.path.join(PROJECT_PATH, "data")
DATASETS_PATH = os.path.join(PROJECT_PATH, 'datasets')
SCHEDULER_PATH = os.path.join(PROJECT_PATH, 'scheduler')
TRAIN_RESULTS_PATH = os.path.join(DATA_PATH, "trainresults")
TESTS_RESULTS_PATH = os.path.join(DATA_PATH, "testresults")
RAY_LOGS_PATH = os.path.join(DATA_PATH, "ray_logs")

CONFIGS_PATH = os.path.join(DATA_PATH, "configs")
#BACKUP_PATH = os.path.join(DATA_PATH, "backup")
PLOTS_PATH = os.path.join(DATA_PATH, "plots")

def _create_dirs():
    """
    create directories if they don't exist
    """
    if not os.path.exists(DATA_PATH):
       os.makedirs(DATA_PATH)
    if not os.path.exists(DATASETS_PATH):
        os.makedirs(DATASETS_PATH)
    if not os.path.exists(TRAIN_RESULTS_PATH):
        os.makedirs(TRAIN_RESULTS_PATH)
    if not os.path.exists(CONFIGS_PATH):
        os.makedirs(CONFIGS_PATH)
    if not os.path.exists(TESTS_RESULTS_PATH):
        os.makedirs(TESTS_RESULTS_PATH)
    if not os.path.exists(PLOTS_PATH):
        os.makedirs(PLOTS_PATH)
    if not os.path.exists(RAY_LOGS_PATH):
        os.makedirs(RAY_LOGS_PATH)

_create_dirs()

ENVS = {
    'sim-edge': SimEdgeEnv,
    #'kube-scheduler': KubeSchedulerEnv,
}
#
ENVSMAP = {
    'sim-edge': 'SimEdgeEnv-v0',
#     'sim-binpacking': 'SimBinpackingEnv-v0',
#     'kube-scheduler': 'KubeSchedulerEnv-v0',
#     'kube-binpacking': 'KubeBinpackingEnv-v0',
#     'kube-greedy': 'KubeGreedyEnv-v0',
}
