import julia
import math
import numpy as np
import random
from gym.spaces import Box

import os
if os.environ.has_key('DISPLAY'):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    GRAPHICS = True
else:
    GRAPHICS = False


class DriveSpec():
    def __init__(self, reward_threshold, timestep_limit):
        self.reward_threshold = reward_threshold
        self.timestep_limit = timestep_limit


class DriveEnv(object):
    def __init__(self, reward_fn=None, mu=0., std=1.):
        self.dt = 0.1
        self.reward_fn = reward_fn
        self.mu = mu
        self.std = std

        # Imitation attributes.
        self.viewer = None
        # Might need this to store a reward threshold...
        self.spec = DriveSpec(100, 100)

    def update_obsnorm(self, mu, std):
        self.mu = mu
        self.std = std


class DriveEnv_1D(DriveEnv):
    def __init__(self, reward_fn=None, mu=0., std=1.0):
        super(DriveEnv_1D, self).__init__(reward_fn, mu, std)

        # Load in trajectory data from NGSIM
        self.j = j = julia.Julia()
        j.using("NGSIM")
        j.using("AutomotiveDrivingModels")
        j.add_module_functions("NGSIM")
        print 'Loading NGSIM data...'
        self.trajdata = j.eval("load_trajdata(1)")
        self.ids = self.j.get_ids(self.trajdata)
        print 'Done.'

        # Graphics
        if GRAPHICS:
            _, self.ax = plt.subplots(1, 1)
            drawParams = {}

            drawParams['xBottom'] = -50
            drawParams['yTop'] = 10.0
            drawParams['xTop'] = 10.0
            drawParams['carlength'] = carlength = 5
            drawParams['carheight'] = carheight = 2

            drawParams['txtDist'] = plt.text(
                0.5, 0.9, '', ha='center', va='center', transform=self.ax.transAxes)
            drawParams['txtS_ego'] = plt.text(
                0.3, 0.5, '', ha='center', va='center', transform=self.ax.transAxes)
            drawParams['txtS_lead'] = plt.text(
                0.7, 0.5, '', ha='center', va='center', transform=self.ax.transAxes)

            self.ax.set_xlim((drawParams['xBottom'], drawParams['xTop']))
            self.ax.set_ylim((0, drawParams['yTop']))
            self.ax.xaxis.set_major_locator(plt.NullLocator())
            self.ax.yaxis.set_major_locator(plt.NullLocator())
            self.ax.set_aspect(1)

            drawParams['ego'] = ego = mpl.patches.Rectangle(
                (0 - carlength, 0), carlength, carheight, color='b')
            drawParams['lead'] = lead = mpl.patches.Rectangle(
                (0, 0), carlength, carheight, color='r')

            self.drawParams = drawParams

            self.ax.add_patch(ego)
            self.ax.add_patch(lead)

    def reset(self):
        self.tstep = 0
        self.x_lead = 0
        self.x_ego = random.uniform(-35, -25)

        # Select random vehicle and store speed in vector
        # carid = random.randint(1, 1000)
        carid = np.random.choice(self.ids)
        self.s_lead = self._get_speeds(carid)
        self.s_ego = random.uniform(
            max(0, self.s_lead[0] - 5), self.s_lead[0] + 5)

        # Compute state and return
        self.d = self.x_lead - self.x_ego
        self.r = self.s_lead[self.tstep] - self.s_ego

        self.prev_x_ego = self.x_ego
        self.prev_s_ego = self.s_ego

        return np.array([self.d, self.r, self.s_ego])

    def render(self):
        if GRAPHICS:
            plt.ion()
            plt.show()

            xBottom = self.drawParams['xBottom']
            carlength = self.drawParams['carlength']
            ego = self.drawParams['ego']

            self.drawParams['txtDist'].set_text('distance: %f' % self.d)
            self.drawParams['txtS_ego'].set_text('ego speed: %f' % self.s_ego)
            self.drawParams['txtS_lead'].set_text(
                'lead speed: %f' % self.s_lead[self.tstep])

            self.ax.set_xlim(
                (min([-self.d - 2 * carlength, xBottom]), 2 * carlength))

            ego.set_xy((-self.d - carlength, 0))

            plt.draw()

            plt.pause(0.1)

    def step(self, action):
        info = {}
        done = False

        action = np.squeeze(
            np.clip(action, self.action_space.low, self.action_space.high)
        )

        self.x_lead += self.s_lead[self.tstep] * self.dt
        self.x_ego += self.s_ego * self.dt
        self.d = self.x_lead - self.x_ego
        self.s_ego += action * self.dt
        self.tstep += 1
        self.r = self.s_lead[self.tstep] - self.s_ego

        self.a_lead = self.s_lead[self.tstep] - self.s_lead[self.tstep - 1]

        if self.tstep + 1 == len(self.s_lead):
            done = True

        #reward = 0.0
        #reward += self._cost_d(30.)
        if self.s_ego <= 0:
            reward = self.reward_fn(150.)
            done = True
        if self.d <= 0:
            reward = self.reward_fn(150.)
            done = True

        # NOTE: Bogus reward signal to keep track of what GAIL is doing.
        if self.reward_fn:
            reward = self.reward_fn(self.d)
        else:
            reward = -self._cost_d(20.)

        ob = np.array([self.d, self.r, self.s_ego])
        ob = (ob - self.mu) / self.std

        assert ob.ndim == 1
        return ob, reward, done, info

    @property
    def action_space(self):
        # return Box(low=-5., high=5., shape=(1,))
        return Box(low=-12.0, high=12.0, shape=(1,))

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(3,))

    @property
    def reward_mech(self):
        return "global"

    def _get_speeds(self, carid):
        speeds = self.j.get_speeds(self.trajdata, carid)
        return speeds

    def _cost_d(self, d_des):
        return math.exp(-(self.d - d_des)**2 / 3.)

# def HandCraftedPolicy(observation, env, des_dist= 10.0):
    # Speed Matching
    #action = (env.s_lead[env.tstep + 10] - env.s_ego)
    # return action


def MaintainDistancePolicy(observation, env, d=10.0):

    #action = env.a_lead + 0.5*(env.d - des_dist) + 0.5*env.r
    action = (env.d - d) + env.r
    return action
