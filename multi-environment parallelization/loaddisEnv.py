import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import csv

def read__csvload():
    dataset = list()
    with open('load1-56,058.csv', 'r') as file:
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
        self.s = [0,0,0,0,0,0,0,0,0,0,0,0,0]
        read = read__csvload()
        self.Qb1 = read[0][1]
        self.Qb2 = read[1][1]
        self.Qb3 = read[2][1]
        self.Qb4 = read[3][1]
        self.Qb5 = read[4][1]
        self.Qb6 = read[5][1]
        self.Qb7 = read[6][1]
        self.Qb8 = read[7][1]
        self.Qb9 = read[8][1]
        self.Qb10 = read[9][1]
        self.Qb11 = read[10][1]
        self.Qb12 = read[11][1]

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
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        action=action* 10 * 0.01 * 0.16703
        Qb1 = self.Qb1
        Qb2 = self.Qb2
        Qb3 = self.Qb3
        Qb4 = self.Qb4
        Qb5 = self.Qb5
        Qb6 = self.Qb6
        Qb7 = self.Qb7
        Qb8 = self.Qb8
        Qb9 = self.Qb9
        Qb10 = self.Qb10
        Qb11 = self.Qb11
        Qb12 = self.Qb12
        state1_Qp01 = Qb1 * (1 - action[0])
        state2_Qp02 = Qb2 * (1 - action[1])
        state3_Qp03 = Qb3 * (1 - action[2])
        state4_Qp04 = Qb4 * (1 - action[3])
        state5_Qp05 = Qb5 * (1 - action[4])
        state6_Qp06 = Qb6 * (1 - action[5])
        state7_Qp07 = Qb7 * (1 - action[6])
        state8_Qp08 = Qb8 * (1 - action[7])
        state9_Qp09 = Qb9 * (1 - action[8])
        state10_Qp010 = Qb10 * (1 - action[9])
        state11_Qp011 = Qb11 * (1 - action[10])
        state12_Qp012 = Qb12 * (1 - action[11])
        Qs_sum = (Qb1 - state1_Qp01) + (Qb2 - state2_Qp02) + (Qb3 - state3_Qp03) + (Qb4 - state4_Qp04) + (Qb5 - state5_Qp05) + (Qb6 - state6_Qp06) + (
                    Qb7 - state7_Qp07) + (Qb8 - state8_Qp08) + (Qb9 - state9_Qp09) + (Qb10 - state10_Qp010) + (Qb11 - state11_Qp011) + (
                             Qb12 - state12_Qp012)
        Ssi = 25226.136 - Qs_sum
        Ssi12 = Ssi / 12
        state1_Qp1 = state1_Qp01 - Ssi12
        state2_Qp2 = state2_Qp02 - Ssi12
        state3_Qp3 = state3_Qp03 - Ssi12
        state4_Qp4 = state4_Qp04 - Ssi12
        state5_Qp5 = state5_Qp05 - Ssi12
        state6_Qp6 = state6_Qp06 - Ssi12
        state7_Qp7 = state7_Qp07 - Ssi12
        state8_Qp8 = state8_Qp08 - Ssi12
        state9_Qp9 = state9_Qp09 - Ssi12
        state10_Qp10 = state10_Qp010 - Ssi12
        state11_Qp11 = state11_Qp011 - Ssi12
        state12_Qp12 = state12_Qp012 - Ssi12
        list_Qp = [state1_Qp1, state2_Qp2, state3_Qp3, state4_Qp4, state5_Qp5, state6_Qp6, state7_Qp7, state8_Qp8, state9_Qp9, state10_Qp10,
                   state11_Qp11, state12_Qp12]
        state13_FC = np.var(list_Qp)
        done = False
        reward = -state13_FC
        state = [state1_Qp1, state2_Qp2, state3_Qp3, state4_Qp4, state5_Qp5, state6_Qp6, state7_Qp7, state8_Qp8, state9_Qp9, state10_Qp10, state11_Qp11, state12_Qp12, state13_FC]
        return np.array(state).squeeze(), reward, done, {}

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
