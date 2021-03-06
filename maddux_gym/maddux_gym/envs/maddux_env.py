import gym
from gym import spaces
from gym.envs.registration import register
import numpy as np
from maddux_gym.maddux.objects import Obstacle, Ball
from maddux_gym.maddux.environment import Environment
from maddux_gym.maddux.robots.link import Link
from maddux_gym.maddux.robots.arm import Arm
from maddux_gym.maddux.robots import noodle_arm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# register(
#     id='maddux-v0',
#     entry_point='envs:MadduxEnv',
# )

class MadduxEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False):
        super(MadduxEnv, self).__init__()

        # maddux env
        # obstacles = []
        obstacles = [Obstacle([1, 2, 1], [2, 2.5, 1.5]),
                     Obstacle([3, 2, 1], [4, 2.5, 1.5])]

        # obstacles = [Obstacle([1, 2, 1], [2, 2.5, 1.5]),
        #              Obstacle([3, 2, 1], [4, 2.5, 1.5]),
        #              Obstacle([0, 0, 1], [1,1.5, 2.5]), ]

        # obstacles = [Obstacle([1, 2, 1], [2, 2.5, 1.5]),
        #              Obstacle([3, 2, 1], [4, 2.5, 1.5]),
        #              Obstacle([0, 0, 1], [1,1.5, 2.5]),
        #              Obstacle([6, 0, 1], [7,1.5, 2.5]), ]


        # obstacles = [Obstacle([1, 2, 1], [2, 2.5, 1.5]),
        #              Obstacle([3, 2, 1], [4, 2.5, 1.5]),
        #              Obstacle([0, 0, 1], [1,1.5, 2.5]),
        #              Obstacle([6, 0, 1], [7,1.5, 2.5]),
        #              Obstacle([3, 0, 4], [4,1.5, 5.5]), ]

        # ball = Ball([2.5, 2.5, 2.0], 0.25)



        # obstacles = [Obstacle([1, 2, 1], [2, 2.5, 1.5]),
        #              Obstacle([3, 2, 1], [4, 2.5, 1.5])]
        # Create a series of links (each link has one joint)
        self.num_links = 5
        L1 = Link(0,0,0,1.571)
        L2 = Link(0,0,0,-1.571)
        L3 = Link(0,2,0,-1.571)
        L4 = Link(0,0,0,1.571)
        L5 = Link(0,2,0,1.571)
        self.links = np.array([L1, L2, L3, L4, L5])
        self.base_pos = np.array([2.0, 2.0, 0.0])

        # Initial arm angle
        q0 = np.array([0, 0, 0, np.pi/2, 0])

        # Create arm
        r = Arm(self.links, q0, '1-link', base=self.base_pos)

        self.mad_env = Environment(dimensions=[10.0, 10.0, 20.0],
                              dynamic_objects=[],
                              static_objects=obstacles,
                              robot=r)

        # actions space
        self.action_space = spaces.Box(low=-1.0,high=1.0,shape=(self.num_links,))
        self.action_scale = 0.2 # max delta theta

        # obs space
        self.observation_space = spaces.Box(low=0,high=2*np.pi,shape=(self.num_links,))

        # goal
        self.goal = None

        # conditions
        self.hit_obstacle = False
        self.steps = 0
        # self.max_steps = 10
        self._max_episode_length = 30
        self.reset_ang = q0

        # render stuff
        self.render_mode = False
        if render:
            self.render_mode = True
            self.fig = plt.figure(figsize=(12, 12))
            self.ax = Axes3D(self.fig)
            plt.ion()
            plt.show()


    def get_obs(self):
        return self.mad_env.robot.get_current_joint_config()


    def compute_reward(self, obs):
        # TODO: distance from target
        reward = 0
        if self.goal is not None:
            reward = -np.linalg.norm(np.minimum(np.absolute(self.goal - obs), np.absolute(2*math.pi - np.maximum(self.goal,obs) +np.minimum(self.goal,obs) )))
            #reward = -np.linalg.norm(self.goal-obs)
        return reward


    def check_done(self):
        if self.hit_obstacle or self.steps >= self._max_episode_length:
            return True
        return False


    def get_info(self):
        return {}


    def sample_random_goal(self):
        collision = True
        while collision:
            rand_goal = self.observation_space.sample()
            r = Arm(self.links, rand_goal, '1-link', base=self.base_pos)
            collision = False
            for obstacle in self.mad_env.static_objects:
                if r.is_in_collision(obstacle):
                    collision = True
                    break
            
        self.goal = rand_goal
        return rand_goal

    def check_collision(self):
        collision = False
        for obstacle in self.mad_env.static_objects:
            if self.mad_env.robot.is_in_collision(obstacle):
                collision = True
                break

        return collision


    def step(self, action):
        # apply new joint angles
        self.hit_obstacle = False

        q_olds = []
        self.steps += 1
        for i in range(self.num_links):
            q_olds.append(self.mad_env.robot.links[i].theta)
            q_new = (self.mad_env.robot.links[i].theta + (action[i] * self.action_scale))%(2*math.pi)
            #q_new = self.mad_env.robot.links[i].theta + (action[i] * self.action_scale)
            self.mad_env.robot.update_link_angle(i, q_new, True)

        self.hit_obstacle = self.check_collision()
        if self.hit_obstacle:
            for i in range(self.num_links):
                self.mad_env.robot.update_link_angle(i, q_olds[i], True)

        done = self.check_done()
        next_obs = self.get_obs()
        reward = self.compute_reward(next_obs)
        info = self.get_info()

        return next_obs, reward, done, info


    def reset(self):
        #self.reset_ang[:] = 1
        # reset joint angles

        collision = True
        while collision:
            self.reset_ang = self.observation_space.sample()
            for i in range(self.num_links):
                self.mad_env.robot.update_link_angle(i, self.reset_ang[i], True)
            collision = self.check_collision()

        self.steps = 0
        self.hit_obstacle = False
        return self.get_obs()
        # TODO: reset dynamic obstacles

        # TODO: select random goal?


    def render(self, mode='human'):
        if self.render_mode:
            self.ax.clear()
            self.mad_env.plot(ax=self.ax, show=False)
            plt.draw()
            plt.pause(0.001)
