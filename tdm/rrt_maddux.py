import numpy as np
from maddux.objects import Obstacle, Ball
from math import pi
from maddux_gym.maddux.robots.arm import Arm
import math
import copy

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
np.random.seed(1)

REACHED = 0
ADVANCED = 1
TRAPPED = 2    

def warped_dist(q1, q2):
    return np.linalg.norm(np.minimum(np.absolute(q1-q2), np.absolute(2*math.pi - np.maximum(q1,q2) + np.minimum(q1,q2))), ord=2)

class RRTPolicy:
    def __init__(self, env):
        self.obstacles = env.mad_env.static_objects
        self.links = env.links
        self.base_pos = env.base_pos 

    @staticmethod
    def nearest_neighbor(tree, q_rand):
        global dist
        dist = [warped_dist(q[0], q_rand) for q in tree]
        #dist = [np.linalg.norm(q[0]-q_rand) for q in tree]
        return dist.index(min(dist))
    
    def interpolate(self, q_near, q_rand):
        #q_new = q_near + np.sign(q_rand-q_near) * np.minimum(np.abs(q_rand - q_near), 0.2)
        close_direction = (np.absolute(q_rand-q_near) < np.absolute(2*math.pi - np.maximum(q_rand,q_near) + np.minimum(q_rand,q_near)))

        q_new = q_near + close_direction * np.sign(q_rand-q_near) * np.minimum(np.abs(q_rand - q_near), 0.2) - (1-close_direction) * np.sign(q_rand-q_near) * np.minimum(np.abs(2*math.pi - np.maximum(q_rand, q_near) + np.minimum(q_rand, q_near)), 0.2)

        q_new = q_new%(2*math.pi)

        r = Arm(copy.deepcopy(self.links), q_new, '1-link', base=copy.deepcopy(self.base_pos))
        if any([r.is_in_collision(obstacle) for obstacle in self.obstacles]):
            return q_near
        else:
            return q_new 
    
    @staticmethod
    def equal(q1, q2):
        if np.max(np.minimum(np.absolute(q1-q2), np.absolute(2*math.pi - np.maximum(q1,q2) + np.minimum(q1,q2)))) > 0.01:
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

        K = 10000

        min_dist = 10000
        mid_point = None

        for k in range(K):
            if k%2 == 0:
                tree_a, tree_b = tree_init, tree_goal 
            else:
                tree_a, tree_b = tree_goal, tree_init 
            
            q_rand = np.random.uniform(low=0, high=2*pi, size=(nDoF))
            q_rand = q_rand%(2*math.pi)

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
    
    def action(self, start, goal):
        plan = self.RRTConnect(start, goal)

        q_near = plan[0]
        q_rand = plan[1]

        close_direction = (np.absolute(q_rand-q_near) < np.absolute(2*math.pi - np.maximum(q_rand,q_near) + np.minimum(q_rand,q_near)))

        return close_direction * np.sign(q_rand-q_near) * np.minimum(np.abs(q_rand - q_near), 0.2) - (1-close_direction) * np.sign(q_rand-q_near) * np.minimum(np.abs(2*math.pi - np.maximum(q_rand, q_near) + np.minimum(q_rand, q_near)), 0.2)

import sys
import maddux_gym
sys.path.append('../maddux_gym/maddux_gym/')
sys.path.append('../tdm/')
from envs.maddux_env import MadduxEnv
env = MadduxEnv(render=False)

policy = RRTPolicy(env)
print(policy.RRTConnect(env.reset(), env.sample_random_goal()))