
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
def warp_dist(a, b):
    return np.linalg.norm(np.minimum(np.absolute(a-b), np.absolute(2*math.pi - np.maximum(a,b) + np.minimum(a,b))), ord=2)

def warp_dist_torch(a, b):
    return torch.min(torch.abs(a-b), torch.abs(2*math.pi - torch.max(a,b) + torch.min(a,b)))

class CEMoptimizer(object):
    def __init__(self, env, tdm, horizon=5, timesteps_per_horizon=10):

        ###########
        # params
        ###########
        self.horizon = horizon
        self.timesteps_per_horizon = timesteps_per_horizon
        self.N = 10000
        self.env = env
        self.tdm = tdm
        #############
        # params for CEM controller
        #############
        self.max_iters = 100
        self.num_elites = 200
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
        var = np.tile(np.square(self.env.observation_space.high[0]-self.env.observation_space.low[0])*4, [self.sol_dim])
        t = 0
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale= np.ones_like(mean))

        #import ipdb; ipdb.set_trace()

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
            #print(np.min(subgoal_traj_scale), np.mean(subgoal_traj_scale), np.max(subgoal_traj_scale))
            costs = np.zeros((self.N,))
            for h in range(self.horizon+1):
                
                # Get model reachability cost
                if(h==0):
                    curr_state_t = torch.FloatTensor(curr_state).to(device)
                else:
                    curr_state_t = torch.FloatTensor(subgoal_traj_scale[:,(h-1)*self.obs_dim: (h)*self.obs_dim]).to(device)
                if(h==self.horizon):
                    curr_subgoal_t = torch.FloatTensor(goal_state).to(device)
                else:
                    subgoal_h = subgoal_traj_scale[:,h*self.obs_dim: (h+1)*self.obs_dim]
                    curr_subgoal_t = torch.FloatTensor(subgoal_h).to(device)
                horizon_t = torch.FloatTensor(np.zeros((self.N))+self.timesteps_per_horizon).view(-1,1).to(device)
                curr_action_t = self.tdm.pi(torch.cat((curr_state_t,curr_subgoal_t, horizon_t),dim=1))[0].to(device)
                state_argument = torch.cat((curr_state_t,curr_subgoal_t,horizon_t),dim=1)
                # import ipdb;ipdb.set_trace()
                if(h==self.horizon):
                    #import ipdb; ipdb.set_trace()
                    costs += torch.abs(self.tdm.q1(state_argument,curr_action_t)+curr_state_t - curr_subgoal_t).sum(1).detach().cpu().numpy()
                    # costs+= warp_dist_torch(self.tdm.q1(state_argument,curr_action_t),curr_subgoal_t).sum(1).detach().cpu().numpy()
                    #costs += warp_dist_torch(curr_state_t, curr_subgoal_t).sum(1).detach().cpu().numpy()
                else:
                    costs += torch.abs(self.tdm.q1(state_argument,curr_action_t)+curr_state_t  - curr_subgoal_t).sum(1).detach().cpu().numpy()
                    # costs+= warp_dist_torch(self.tdm.q1(state_argument,curr_action_t),curr_subgoal_t).sum(1).detach().cpu().numpy()
                    #costs += warp_dist_torch(curr_state_t, curr_subgoal_t).sum(1).detach().cpu().numpy()
                
                # if h == 0:
                #     import ipdb; ipdb.set_trace();

            # if t == self.max_iters-1:
            #     import ipdb; ipdb.set_trace()
            indices = np.argsort(costs)
            print(costs[indices][:self.num_elites].mean())
            elites = subgoal_traj[indices][:self.num_elites]
            mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)
            var =  (self.alpha) * var + (1 - self.alpha) * new_var
            # print(new_var, var, "\n")
            # Diagonalize       
            t += 1
        self.mean = mean
        end = time.time()
        return (mean+1)/2 * self.env.observation_space.high[0]

# class CEMoptimizer(object):
#     def __init__(self, env, tdm, horizon=5, timesteps_per_horizon=10):

#         ###########
#         # params
#         ###########
#         self.horizon = horizon
#         self.timesteps_per_horizon = timesteps_per_horizon
#         self.N = 400
#         self.env = env
#         self.tdm = tdm
#         #############
#         # params for CEM controller
#         #############
#         self.max_iters = 200
#         self.num_elites = 20
#         self.action_dim = self.env.action_space.shape[0]
#         self.obs_dim = self.env.observation_space.shape[0]
#         self.sol_dim = self.env.observation_space.shape[0] * self.horizon
#         self.ub = np.repeat(self.env.observation_space.high,self.horizon,axis=0)*0 + 1
#         self.lb = np.repeat(self.env.observation_space.low,self.horizon,axis=0)*0 + -1
#         self.epsilon = 0.001
#         self.alpha = 0.1
#         self.initial_var = 0.25
#         self.mean = np.zeros((self.sol_dim,))



#     def reset(self):
#         self.mean = np.zeros((self.sol_dim,))

#     def get_path(self, curr_state, goal_state):
#         start = time.time()
#         curr_state = np.array([curr_state.copy()] * self.N)
#         goal_state = np.array([goal_state.copy()] * self.N)
        
#         mean = self.mean
        
#         var = np.tile(np.square(2)/48, [self.sol_dim])
#         t = 0
#         X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale= np.ones_like(mean))

#         # CEM
#         while ((t < self.max_iters)):
#             lb_dist, ub_dist = mean - self.lb, self.ub - mean
            
#             constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

#             subgoal_traj = (X.rvs(size=(self.N, self.sol_dim)) * np.sqrt(constrained_var) + mean).astype(np.float32)

            
#             # clip subgoals between -1 and 1
#             subgoal_traj = np.clip(subgoal_traj, -1, 1)
#             # transform subgoals to true scale
            
#             subgoal_traj_scale = (subgoal_traj+1)/2 * self.env.observation_space.high[0]
#             # subgoal_traj_scale = subgoal_traj
#             costs = np.zeros((self.N,))
#             for h in range(self.horizon):
                
#                 # Get model reachability cost
#                 if(h==0):
#                     curr_state_t = torch.FloatTensor(curr_state).to(device)
#                 else:
#                     curr_state_t = torch.FloatTensor(subgoal_traj_scale[:,(h-1)*self.obs_dim: (h)*self.obs_dim]).to(device)
#                 if(h==self.horizon-1):
#                     curr_subgoal_t = torch.FloatTensor(goal_state).to(device)
#                 else:
#                     # import ipdb;ipdb.set_trace()
#                     subgoal_h = subgoal_traj_scale[:,h*self.obs_dim: (h+1)*self.obs_dim]
#                     curr_subgoal_t = torch.FloatTensor(subgoal_h).to(device)
#                 horizon_t = torch.FloatTensor(np.zeros((self.N))+self.timesteps_per_horizon).view(-1,1).to(device)
#                 curr_action_t = self.tdm.pi(torch.cat((curr_state_t,curr_subgoal_t, horizon_t),dim=1))[0].to(device)
#                 state_argument = torch.cat((curr_state_t,curr_subgoal_t,horizon_t),dim=1)
#                 # import ipdb;ipdb.set_trace()
#                 if(h==self.horizon-1 or h==0):
#                     # if(h==self.horizon-1):
#                         # import ipdb;ipdb.set_trace()
#                     costs+=10*torch.abs(self.tdm.q1(state_argument,curr_action_t)+curr_state_t-curr_subgoal_t).sum(1).detach().cpu().numpy().reshape(-1)
#                 else:
#                     costs+= torch.abs(self.tdm.q1(state_argument,curr_action_t)+curr_state_t-curr_subgoal_t).sum(1).detach().cpu().numpy().reshape(-1)

#                 # print(costs)
#             indices = np.argsort(costs)
#             print(costs[indices][:self.num_elites].mean())
#             elites = subgoal_traj[indices][:self.num_elites]
#             mean = np.mean(elites, axis=0)
#             new_var = np.var(elites, axis=0)
#             var =  (self.alpha) * var + (1 - self.alpha) * new_var
#             # Diagonalize       
#             t += 1
#         self.mean = mean
#         end = time.time()


#         return (mean+1)/2 * self.env.observation_space.high[0]

def get_action(tdm, o, deterministic=False):
        return tdm.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic)

def get_q_value(tdm,o,a):
    return tdm.q1(torch.as_tensor(o, dtype=torch.float32), 
                    torch.as_tensor(a, dtype=torch.float32))

def test_tdm(tdm):
        goal_reaches = 0
        test_env = gym.make('Maddux-v0')
        for j in range(10):
            goal = test_env.sample_random_goal()    
            horizon = np.random.randint(1,30)
            q_values = []
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            start = o
            goal_reached = False
            for t in range(horizon,0,-1):   
                a = get_action(tdm, np.concatenate((o,goal,np.array([t]))),True)
                q_val = get_q_value(tdm, np.concatenate((o,goal,np.array([t]))).reshape(1,-1), a.reshape(1,-1)).detach().cpu().numpy()
                q_values.append(q_val)
                o, r, d, _ = test_env.step(a)
                ep_ret += r
                if((o-goal).sum()<(0.1*5)):
                    goal_reached=True
                
                if(t == 1):
                    #print(start, goal, o)
                    pass
            if(goal_reached):
                goal_reaches+=1

            q_errors = np.abs(np.abs(q_values-goal)-np.abs(o-goal))
            #import ipdb; ipdb.set_trace()
            #print(q_errors, goal_reaches)
            # q_errors = np.abs(np.array(q_values).squeeze()+np.abs(o-goal).reshape(1,-1))

if __name__=='__main__':

    # Add proper model path here
    tdm_model_path = "/Users/harshit/work/git/long_horizon_manipulation/data/dump1_s0/pyt_save/model.pt"
    # tdm_model_path = "/Users/harshit/work/git/long_horizon_manipulation/data/tdm_models/no_her_tejus_wrap_long_run_s0/pyt_save/model.pt"
    env = gym.make('Maddux-v0')
    obs_dim = env.observation_space.shape[0]

    tdm = torch.load(tdm_model_path)
    test_tdm(tdm)

    # Planner params
    horizon = 5
    timesteps_per_horizon = 4

    cem_planner = CEMoptimizer(env, tdm, horizon=horizon, timesteps_per_horizon=timesteps_per_horizon)

    start = np.array([0,0,0,0,0])
    goal = np.array([3.0,3.0,3.0,3.0,3.0])

    path = cem_planner.get_path(start,goal)
    print("Planned subgoals along path, start: {}, goal: {}".format(start, goal))
    for i in range(horizon):
        print(path[i*obs_dim:(i+1)*obs_dim])
    print("---------------------------------------------")



