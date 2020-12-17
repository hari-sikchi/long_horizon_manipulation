from gym.envs.registration import register

register(
    id='PointNavEnv-v0',
    entry_point='point_nav.envs:PointNavEnv',
)

register(
    id='PointEnv-v0',
    entry_point='point_nav.envs:PointEnv',
)