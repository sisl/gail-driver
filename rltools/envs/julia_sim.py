
import julia
import math
import numpy as np
import random
from gym.spaces import Box

import os

import time

#from path_to_Auto2D import LQG_path, auto1D_path, auto2D_path, pulltraces_path, passive_aggressive_path
from rllab.config_personal import auto2D_path

from drive import DriveEnv_1D

if os.environ.has_key('DISPLAY'):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    GX = True
else:
    GX = False

julia_env_dict = {}
julia_env_dict["Auto2D"] = auto2D_path


class JuliaEnv(object):
    def __init__(self,
                 env_name,  # name of the environment to load
                 batch_size,
                 # dictionary of parameters Dict{String,Any} passed to the env
                 # initialization
                 param_dict,
                 ):

        # Load in functions
        self.j = julia.Julia()
        self.j.eval("include(\"" + julia_env_dict[env_name] + "\")")
        self.j.using(env_name)

        self.simparams = self.j.gen_simparams(batch_size, param_dict)

        if GX:
            _, self.ax = plt.subplots(1, 1)

    def reset(self, render=False):

        if GX:
            self.ax.cla()

        self.j.reset(self.simparams)
        # [batch_size x n_features]
        observation = self.j.observe(self.simparams)

        return observation

    def render(self):
        return

    def save_gif(self, actions, filename):
        # TODO - have a way to record states over time to do rendering
        # actions is a matrix whose columns are the actions
        # self.j.reel_drive(filename+".gif", actions, self.simparams)
        return

    def step(self, actions):
        info = {}
        # features for next state
        # reward for (s,a,s')
        # done for whether in terminal state
        obs, reward, done = self.j.step(self.simparams, actions)
        return obs, reward, done, info

    @property
    def action_space(self):
        lo, hi = self.j.action_space_bounds(self.simparams)
        return Box(np.array(lo), np.array(hi))

    @property
    def observation_space(self):
        lo, hi = self.j.observation_space_bounds(self.simparams)
        return Box(np.array(lo), np.array(hi))

    @property
    def reward_mech(self):
        """
        If your step function returns multiple rewards for different agents
        """
        return 'global'


class JuliaLQGEnv():
    def __init__(self):

        # Load in functions
        self.j = julia.Julia()
        self.j.eval("include(\"" + LQG_path + "\")")
        self.j.using("juliaLQG")

    def reset(self, render=False):
        self.tstep = 0

        self.x = self.j.restart()
        return self.x

    def render(self, actions, filename):
        return

    def step(self, action):
        info = {}

        action = np.clip(action, self.action_space.low, self.action_space.high)
        reward = self.j.reward(self.x, action)
        self.x = self.j.tick(self.x, action)
        observation = self.j.observe(self.x)

        self.tstep += 1
        isdone = self.tstep >= 100

        return observation, reward, isdone, info

    @property
    def action_space(self):
        return Box(np.array([-5, -5]), np.array([5, 5]))

    @property
    def observation_space(self):
        return Box(np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))

    @property
    def reward_mech(self):
        """
        Should probably add more here ...
        """
        return 'global'


class JuliaDriveEnv():
    def __init__(self):
        self.dt = 0.1

        # Load in functions
        self.j = julia.Julia()
        self.j.eval("include(\"" + auto1D_path + "\")")
        self.j.using("Auto1D")

        self.roadway = self.j.gen_stadium_roadway(1)  # one-lane track

    def reset(self, render=False):
        self.tstep = 0

        # Initialize vehicles and scene, get state
        self.scene, self.model1, self.model2 = self.j.restart(
            self.roadway, render)
        self.d, self.r, self.s, _ = self.j.get_state(self.scene, self.roadway)
        if render:
            self.scene0 = self.j.deepcopy(self.scene)

        return np.array([self.d, self.r, self.s])

    def render(self, actions, filename):
        self.j.reel_drive_1d(filename + ".gif", actions,
                             self.scene0, self.model1, self.model2, self.roadway)
        return

    def step(self, action):
        info = {}
        done = False

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.d, self.r, self.s, done = self.j.step_forward(
            self.scene, self.roadway, self.model1, self.model2, action[0])
        self.tstep += 1

        if self.tstep == 1000:
            done = True

        reward = 0.0
        reward += (self.d >= 17 and self.d <= 23) * 1.0
        if self.s <= 5 or self.s >= 30:
            reward -= 1.
        # if self.d >= 40:
        #     reward -= 1.
        if self.d <= 1 or self.d >= 300:
            reward -= 1
        ob = np.array([self.d, self.r, self.s])
        return ob, reward, done, info

    @property
    def action_space(self):
        return Box(low=-5., high=3., shape=(1,))

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(3,))

    def _cost_d(self, d_des):
        return math.exp(-(self.d - d_des)**2 / 16.)

    @property
    def reward_mech(self):
        """
        Should probably add more here ...
        """
        return 'global'


class JuliaDriveEnv2D():
    def __init__(self, n_features, trajdata_indeces,
                 train_seg_index=0, frame_num=0, save_history=False):

        self.t = 0.1

        self.tstep = 0
        self.max_nsteps = 1000
        self.n_features = n_features
        self.train_seg_index = train_seg_index
        self.frame_num = frame_num

        # Load in functions
        self.j = julia.Julia()
        self.j.eval("include(\"" + auto2D_path + "\")")
        self.j.using("Auto2D")

        #self.j.set_TRAJDATAS(tj_ix= tj_ix)
        #self.simparams = self.j.gen_simparams([1])
        if trajdata_indeces == []:
            print "USING PASSIVE/AGGRESSIVE"

            def append_path(x): return pulltraces_path + x
            trajdatas = ["trajdata_passive_aggressive1.txt",
                         "trajdata_passive_aggressive2.txt"]
            roadways = ["roadway_passive_aggressive.txt",
                        "roadway_passive_aggressive.txt"]

            # self.simparams= self.j.gen_simparams_from_trajdatas(map(append_path,trajdatas),map(append_path,roadways),
            # weights[0], weights[1], weights[2], weights[3], weights[4],
            # weights[5])
            self.simparams = self.j.gen_simparams_from_trajdatas(
                map(append_path, trajdatas), map(append_path, roadways))

        else:
            #self.simparams = self.j.gen_simparams(trajdata_indeces, weights[0], weights[1], weights[2], weights[3], weights[4], weights[5])
            self.simparams = self.j.gen_simparams(trajdata_indeces)

        self.features = self.j.alloc_features()

        if GX:
            _, self.ax = plt.subplots(1, 1)

    def reset(self, render=False):
        self.tstep = 0

        if GX:
            self.ax.cla()

        # Initialize vehicles and scene, get state
        if self.train_seg_index != 0:
            self.j.restart_specific(
                self.simparams, self.train_seg_index, self.frame_num)
        else:
            self.j.restart(self.simparams)
        self.features = self.j.get_state(self.features, self.simparams)

        return self.features

    def render(self):
        plt.ion()
        plt.show()

        self.ax.cla()

        img = self.j.render(self.simparams, np.zeros(
            (500, 500)).astype('uint32'))
        #img=self.j.retrieve_frame_data(500, 500, self.simparams)
        self.ax.imshow(img, cmap=plt.get_cmap('bwr'))
        #self.ax.imshow(img, cmap=plt.get_cmap('seismic'))

        plt.draw()

        plt.pause(1e-6)

        return

    def save_gif(self, actions, filename):
        # TODO - have a way to record states over time to do rendering
        # actions is a matrix whose columns are the actions
        self.j.reel_drive(filename + ".gif", actions, self.simparams)
        return

    def step(self, action):

        action = np.squeeze(action)

        self.j.step_forward(self.simparams, action)
        self.features = self.j.get_state(self.features, self.simparams)
        reward, done = self.j.get_reward(self.simparams)

        self.tstep += 1

        if self.tstep + 1 == self.max_nsteps:
            done = True

        obs = self.features
        info = {}

        return obs, reward, done, info

    @property
    def action_space(self):
        # (accel [m/s2], turnrate [rad/s])
        return Box(np.array([-5.0, -1.0]), np.array([3.0, 1.0]))

    @property
    def observation_space(self):
        # 32 features
        return Box(low=-np.inf, high=np.inf, shape=(self.n_features,))

    @property
    def reward_mech(self):
        """
        Should probably add more here ...
        """
        return 'global'


class JuliaDriveEnv2DBatch():
    def __init__(self,
                 n_features,  # number of features
                 batch_size,  # number of simultaneous sims
                 weights,  # reward weights, as a vector
                 # NGSIM trajdata indeces, full set is [1,2,3,4,5,6]
                 trajdata_indeces,
                 max_nsteps=1000,
                 ):

        self.tstep = 0
        self.max_nsteps = max_nsteps
        self.n_features = n_features

        # Load in functions
        self.j = julia.Julia()
        self.j.eval("include(\"" + auto2D_path + "\")")
        self.j.using("Auto2D")

        self.simparams = self.j.gen_simparams(
            trajdata_indeces, weights[0], weights[1], weights[2], weights[3], weights[4], weights[5], weights[6], batch_size)
        self.features = self.j.alloc_features(batch_size)

        if GX:
            _, self.ax = plt.subplots(1, 1)

    def reset(self, render=False):
        self.tstep = 0

        if GX:
            self.ax.cla()

        # Initialize vehicles and scene, get state
        self.j.restart(self.simparams)
        self.features = self.j.observe(self.features, self.simparams)

        return self.features

    def render(self):
        # plt.ion()
        # plt.show()

        # self.ax.cla()

        # img = self.j.render(self.simparams, np.zeros((500,500)).astype('uint32'))
        # #img=self.j.retrieve_frame_data(500, 500, self.simparams)
        # self.ax.imshow(img, cmap=plt.get_cmap('bwr'))
        # #self.ax.imshow(img, cmap=plt.get_cmap('seismic'))

        # plt.draw()

        # plt.pause(1e-6)

        return

    def save_gif(self, actions, filename):
        # TODO - have a way to record states over time to do rendering
        # actions is a matrix whose columns are the actions
        # self.j.reel_drive(filename+".gif", actions, self.simparams)
        return

    def step(self, actions):
        info = {}

        self.j.step_forward(self.simparams, actions)
        self.features = self.j.get_state(self.features, self.simparams)
        reward, done = self.j.get_reward(self.simparams)

        self.tstep += 1

        if self.tstep + 1 == self.max_nsteps:
            done = True

        obs = self.features

        return obs, reward, done, info

    @property
    def action_space(self):
        # (accel [m/s2], turnrate [rad/s])
        return Box(np.array([-5.0, -1.0]), np.array([3.0, 1.0]))

    @property
    def observation_space(self):
        # 32 features
        return Box(low=-np.inf, high=np.inf, shape=(self.n_features,))

    @property
    def reward_mech(self):
        """
        Should probably add more here ...
        """
        return 'global'


class JuliaEnvWrapper(JuliaEnv):
    _env_name = None
    _batch_size = None
    _param_dict = None

    def __init__(self):
        super(JuliaEnvWrapper, self).__init__(JuliaEnvWrapper._env_name,
                                              JuliaEnvWrapper._batch_size,
                                              JuliaEnvWrapper._param_dict)

    @classmethod
    def set_initials(cls, env_name, batch_size, param_dict):
        cls._env_name = env_name
        cls._batch_size = batch_size
        cls._param_dict = param_dict
