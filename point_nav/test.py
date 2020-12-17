import gym
import point_nav
import numpy as np

env = gym.make("PointNavEnv-v0")
obs = env.reset()
obs_vec = []
goal = obs['desired_goal']
obs_vec.append(obs['observation'])

point_nav.envs.point_nav.plot_trajectory(np.array(obs_vec),goal,'Small')
