import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import csv

def read__csv():
    dataset = list()
    with open('aa.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
        for col in range(len(dataset[0])):
            for row in dataset:
                row[col] = float(row[col].strip())
    return dataset

class loaddisEnv(gym.Env):
    """
        Description:
            When considering participation in a grid DSM program, the grid prefers a flat building power curve because it helps maintain a power balance between supply and demand and reduces the grid stress and equipment design sizes.
            the optimisation objective of load disaggregation for building air-conditioning systems when participating in grid DSM programs is to achieve maximum flatness of the main system load Q_m curve.


        Observation:
            12 hourly Q_m values provided by the main system and their negative variance are selected to be the state space, totalling 13 parameters.


        Actions:
            12 hourly Q_s during the daily cooling/heating hours are used as actions output by the agent, with each Q_s representing the hourly released energy as a proportion of the total load Q_t, discretized into ten values ranging from 0 to 100%.

            Note:    For the proposed EOLD framework, the disaggregated load for each appliance Q_i is formulated as Eq. (1).
                     Q_i=π_(i,j) C_i Q_tol        (π∈R^(N×M); i=1,…,N; j=1,…,M; C_i ∈[0,1])  (1)
                     where π_(i,j) represents a discrete action space matrix determining each appliance's proportion in total load disaggregation. The action dimension N can be flexibly configured according to the number of target appliances and time series, and the discrete value quantity M for individual action is adaptively determined based on individual appliance's load control features. The constraint C_i represents the ratio of a single appliance's rated power to the total rated load. Q_toi is the total load with time series measured by meters.
                     Notably, for real-world scenarios requiring simultaneous charge or discharge operations, 12 additional action space dimensions can be introduced and discretized into ten values ranging from -100% to 0%, which represent the hourly stored energy proportion. The PPO policy network can dynamically adapt to these constraints through gradient-based optimization of the reward function, ensuring feasible charge/discharge overlaps without predefined rules.


        Reward:
            The reward value function directly utilises -𝜎2 from the state, which represents the flatness of the load curve for the main system. With such a configuration, the agent will be trained to maximise the reward value. The larger the -𝜎2, the flatter the main system load curve.

            Note:For extended energy scenarios, the reward function for load disaggregation can be flexibly configured to incorporate practical optimization objectives, such as efficient, flexible, or economical single/multi-objective formulations.


        Episode Termination:
            max_episode_steps=10, or set up according to the actual scene.
        """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.s = read__csv()

        self.action_space = spaces.MultiDiscrete([11] * 12)
        self.observation_space = gym.spaces.Box(low=np.array([0] * 13),high=np.array([100000000] * 13))

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.s = read__csv()
        self.state = self.s
        done = False
        reward = -self.s[0][13-1]
        return np.array(self.s), reward, done, {}

    def reset(self):
        self.state = self.s
        self.steps_beyond_done = None
        return np.array(self.state).squeeze()

    def render(self, mode='human'):
        screen_width = None
        screen_height = None
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
