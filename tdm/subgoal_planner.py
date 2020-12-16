
import copy
import numpy as np
import math
import scipy.stats as stats
import torch
import time
import core as core
import gym
import tdm
import argparse
import gym
import maddux_gym
import sys
sys.path.append('maddux_gym/maddux_gym/')
from envs.maddux_env import MadduxEnv
device = torch.device("cpu")


class CEMoptimizer(object):
    def __init__(self, env, tdm, horizon=5, timesteps_per_horizon=10):

        ###########
        # params
        ###########
        self.horizon = horizon
        self.timesteps_per_horizon = timesteps_per_horizon
        self.N = 100
        self.env = env
        self.tdm = tdm
        #############
        # params for CEM controller
        #############
        self.max_iters = 50
        self.num_elites = 20
        self.action_dim = self.env.action_space.shape[0]
        self.obs_dim = self.env.observation_space.shape[0]
        self.sol_dim = self.env.observation_space.shape[0] * self.horizon
        self.ub = np.repeat(self.env.observation_space.high,self.horizon,axis=0)*0 + 1
        self.lb = np.repeat(self.env.observation_space.low,self.horizon,axis=0)*0 + -1
        self.epsilon = 0.001
        self.alpha = 0.1
        self.initial_var = 0.25
        self.mean = np.zeros((self.sol_dim,))



    def reset(self):
        self.mean = np.zeros((self.sol_dim,))

    def get_path(self, curr_state, goal_state):
        start = time.time()
        curr_state = np.array([curr_state.copy()] * self.N)
        goal_state = np.array([goal_state.copy()] * self.N)
        
        mean = self.mean
        var = np.tile(np.square(self.env.observation_space.high[0]-self.env.observation_space.low[0])/4, [self.sol_dim])
        t = 0
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale= np.ones_like(mean))

        # CEM
        while ((t < self.max_iters)):
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            subgoal_traj = (X.rvs(size=(self.N, self.sol_dim)) * np.sqrt(constrained_var) + mean).astype(np.float32)


            # clip subgoals between -1 and 1
            subgoal_traj = np.clip(subgoal_traj, -1, 1)
            # transform subgoals to true scale
            subgoal_traj_scale = (subgoal_traj+1)/2 * self.env.observation_space.high[0]
            # subgoal_traj_scale = subgoal_traj
            costs = np.zeros((self.N,))
            for h in range(self.horizon):
                
                # Get model reachability cost
                if(h==0):
                    curr_state_t = torch.FloatTensor(curr_state).to(device)
                else:
                    curr_state_t = torch.FloatTensor(subgoal_traj_scale[:,(h-1)*self.obs_dim: (h)*self.obs_dim]).to(device)
                if(h==self.horizon-1):
                    curr_subgoal_t = torch.FloatTensor(goal_state).to(device)
                else:
                    subgoal_h = subgoal_traj_scale[:,h*self.obs_dim: (h+1)*self.obs_dim]
                    curr_subgoal_t = torch.FloatTensor(subgoal_h).to(device)
                horizon_t = torch.FloatTensor(np.zeros((self.N))+self.timesteps_per_horizon).view(-1,1).to(device)
                curr_action_t = self.tdm.pi(torch.cat((curr_state_t,curr_subgoal_t, horizon_t),dim=1))[0].to(device)
                state_argument = torch.cat((curr_state_t,curr_subgoal_t,horizon_t),dim=1)
                if(h==self.horizon-1):
                    costs+=10*torch.abs(self.tdm.q1(state_argument,curr_action_t)-curr_subgoal_t).sum(1).detach().cpu().numpy().reshape(-1)
                else:
                    costs+= torch.abs(self.tdm.q1(state_argument,curr_action_t)-curr_subgoal_t).sum(1).detach().cpu().numpy().reshape(-1)


            indices = np.argsort(costs)
            elites = subgoal_traj[indices][:self.num_elites]
            mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)
            var =  (self.alpha) * var + (1 - self.alpha) * new_var
            # Diagonalize       
            t += 1
        self.mean = mean
        end = time.time()


        return (mean+1)/2 * self.env.observation_space.high[0]


if __name__=='__main__':

    # Add proper model path here
    tdm_model_path = "/Users/harshit/work/git/long_horizon_manipulation/data/tdm_models/experiment_nowrap_s0/pyt_save/model.pt"
    env = gym.make('Maddux-v0')
    obs_dim = env.observation_space.shape[0]

    # tdm = core.MLPtdmActorCritic(env.observation_space, env.action_space,special_policy='tdm')
    tdm = torch.load(tdm_model_path)

    # Planner params
    horizon = 5
    timesteps_per_horizon = 20

    cem_planner = CEMoptimizer(env, tdm, horizon=horizon, timesteps_per_horizon=timesteps_per_horizon)

    start = np.array([0,0,0,0,0])
    goal = np.array([1.6,1.6,1.6,1.6,1.6])

    path = cem_planner.get_path(start,goal)
    print("Planned subgoals along path, start: {}, goal: {}".format(start, goal))
    for i in range(horizon):
        print(path[i*obs_dim:(i+1)*obs_dim])
    print("---------------------------------------------")



