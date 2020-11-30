import gym
import numpy as np
from envs.maddux_env import MadduxEnv

def main():
	env = MadduxEnv(render=True)
	for ep in range(5):
		state = env.reset()
		done = False
		print("state:",state)
		step = 0
		while not done:
			action = env.action_space.sample()
			next_state, reward, done, info = env.step(action)
			print("ep: {} step: {} n_st: {} rew: {} done: {}".format(ep, step, next_state, reward, done))
			step += 1
			state = next_state
			env.render()

if __name__ == '__main__':
	main()