import numpy as np

from math import pi
import gym
import math
import copy

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
np.random.seed(1)

REACHED = 0
ADVANCED = 1
TRAPPED = 2    

def warped_dist(q1, q2):
    return np.linalg.norm(q1-q2)
    # return np.linalg.norm(np.minimum(np.absolute(q1-q2), np.absolute(2*math.pi - np.maximum(q1,q2) + np.minimum(q1,q2))), ord=2)

class RRTPolicy:
    def __init__(self, env):
        self.env = env

    @staticmethod
    def nearest_neighbor(tree, q_rand):
        global dist
        dist = [warped_dist(q[0], q_rand) for q in tree]
        #dist = [np.linalg.norm(q[0]-q_rand) for q in tree]
        return dist.index(min(dist))
    
    def interpolate(self, q_near, q_rand):
        q_new = q_near + np.sign(q_rand-q_near) * np.minimum(np.abs(q_rand - q_near), 1.0)

        # close_direction = (np.absolute(q_rand-q_near) < np.absolute(2*math.pi - np.maximum(q_rand,q_near) + np.minimum(q_rand,q_near)))

        # q_new = q_near + close_direction * np.sign(q_rand-q_near) * np.minimum(np.abs(q_rand - q_near), 0.2) - (1-close_direction) * np.sign(q_rand-q_near) * np.minimum(np.abs(2*math.pi - np.maximum(q_rand, q_near) + np.minimum(q_rand, q_near)), 0.2)

        # q_new = q_new%(2*math.pi)
        if(self.env.env.is_blocked(q_new)):
            return q_near
        else:
            return q_new

    @staticmethod
    def equal(q1, q2):
        if np.linalg.norm(q1-q2) > 0.01:
            return False
        else:
            return True

    def extend(self, tree, q_rand):
        q_near_idx = self.nearest_neighbor(tree, q_rand)
        q_near = tree[q_near_idx][0]

        q_new = self.interpolate(q_near, q_rand)

        if self.equal(q_new, q_near):
            return q_new, TRAPPED 
        
        tree.append((q_new, q_near_idx))

        if self.equal(q_new, q_rand):
            return q_new, REACHED 
        else:
            return q_new, ADVANCED

    def connect(self, tree, q):
        while True:
            q_new, status = self.extend(tree, q)
            if status != ADVANCED:
                break 
        
        return status

    def RRTConnect(self, q_init, q_goal):
        nDoF = q_init.shape[0]
        tree_init = [(q_init, None)]
        tree_goal = [(q_goal, None)]

        #print(q_init)
        #print(q_goal)
        #print(np.linalg.norm(q_init-q_goal))

        K = 100000

        min_dist = 10000
        mid_point = None

        for k in range(K):
            if k%2 == 0:
                tree_a, tree_b = tree_init, tree_goal 
            else:
                tree_a, tree_b = tree_goal, tree_init 
            
            q_rand = np.zeros(2)
            # print(self.env.env.height_p,self.env.env.width_p)
            q_rand[0] = np.random.uniform(low=0,high=self.env.env.height_p)
            q_rand[1] = np.random.uniform(low=0,high=self.env.env.width_p)


            if k == 0:
                q_rand = q_goal
            
            q_new, status = self.extend(tree_a, q_rand)
            if status != TRAPPED:
                if self.connect(tree_b, q_new) == REACHED:
                    mid_point = q_new
                    print("Got plan", k)
                    break
            
            dist = warped_dist(q_new,tree_b[self.nearest_neighbor(tree_b, q_new)][0])
            if dist < min_dist:
                min_dist = dist 
                #print("min_dist", min_dist)
        else:
            # goal not found, return path to closest point to goal
            closest = tree_init[self.nearest_neighbor(tree_init, q_goal)]
            forward_plan = []

            current = closest
            while True:
                forward_plan.append(current[0])
                if current[1] == None:
                    break 

                current = tree_init[current[1]]
            
            forward_plan.reverse()
            return forward_plan
        
        forward_plan = []
        backward_plan = []
        for q,parent in tree_init:
            if self.equal(q, mid_point):
                current = (q,parent)
                break 
        
        while True:
            forward_plan.append(current[0])
            if current[1] == None:
                break 

            current = tree_init[current[1]]
        
        for q,parent in tree_goal:
            if self.equal(q, mid_point):
                current = (q,parent)
                break 
            
        while True:
            backward_plan.append(current[0])
            if current[1] == None:
                break 

            current = tree_goal[current[1]]
        
        forward_plan.reverse()
        return forward_plan + backward_plan
    

import sys
import maddux_gym
import point_nav
# sys.path.append('../maddux_gym/maddux_gym/')
# sys.path.append('../tdm/')

env = gym.make('PointNavEnv-v0')
# path = [[0.2,0.2],[0.5,0.5]]

# path = [[0.2,0.2],[0.48137558,0.2190204 ],[0.71618795 ,0.36848503],[0.79026234,0.5559957 ],[0.58309215,0.6270325 ],[0.5,0.5]]
# path = [[0.8,0.9],[0.2,0.2]]
path = [[0.8,0.9],[0.5996188,0.9001442],[0.32504478,0.85714436],[0.24592718,0.6704873 ],[0.23620796,0.46130514],[0.2,0.2]]
# [0.64862263 0.8578243 ]
# [0.6888677 0.7139572]
# [0.72529495 0.66701394]
# [0.684678   0.58784616]
# [0.55459696 0.8572341 ]
# [0.28557453 0.59361964]


# [0.5996188 0.9001442]
# [0.32504478 0.85714436]
# [0.24592718 0.6704873 ]
# [0.23620796 0.46130514]
for i in range(len(path)-1):
    # import ipdb;ipdb.set_trace()
    start = np.array(path[i])
    start = env.denormalize_obs(start)
    goal = env.denormalize_obs(np.array(path[i+1]))
    policy = RRTPolicy(env)
    plan = policy.RRTConnect(start,goal)
    plan = np.array(plan)
    # import ipdb;ipdb.set_trace()
    env.plot_trajectory(env.normalize_obs(plan),env.normalize_obs(goal.reshape(1,-1)).reshape(-1),'Spiral7x7') 


# [0.48137558 0.2190204 ]
# [0.71618795 0.36848503]
# [0.79026234 0.5559957 ]
# [0.58309215 0.6270325 ]


# policy = RRTPolicy(env)
# start = env.denormalize_obs(env.reset())
# goal = env.denormalize_obs(env.sample_random_goal())
# print(start,goal)
# start = env.denormalize_obs(np.array([0.9,0.9]))
# goal = env.denormalize_obs(np.array([0.5,0.5]))

# # plan = policy.RRTConnect(start,goal)

# plan = policy.RRTConnect(start,goal)
# # import ipdb;ipdb.set_trace()
# plan = np.array(plan)
# print(plan)
# # env.plot_trajectory(plan,goal,'Spiral7x7') 
# env.plot_trajectory(env.normalize_obs(plan),env.normalize_obs(goal.reshape(1,-1)).reshape(-1),'Spiral7x7') 

# print(policy.RRTConnect(start,goal))