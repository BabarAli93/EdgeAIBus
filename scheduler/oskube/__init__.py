from gymnasium.envs.registration import register

register(
    id='SimEdgeEnv-v0',
    entry_point='scheduler.oskube.envs:SimEdgeEnv',
    max_episode_steps=20
)
# register(
#     id='SimBinpackingEnv-v0',
#     entry_point='mobile_kube.envs:SimBinpackingEnv',
# )
# register(
#     id='SimGreedyEnv-v0',
#     entry_point='mobile_kube.envs:SimGreedyEnv',
# )
# register(
#     id='KubeEdgeEnv-v0',
#     entry_point='mobile_kube.envs:KubeEdgeEnv',
# )
# register(
#     id='KubeBinpackingEnv-v0',
#     entry_point='mobile_kube.envs:KubeBinpackingEnv',
# )
# register(
#     id='KubeGreedyEnv-v0',
#     entry_point='mobile_kube.envs:KubeGreedyEnv',
# )
