import gym
import argparse

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.box2d_env import Box2DEnv
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize

from rllab.envs.gym_env import GymEnv

from rltools.envs.julia_sim import JuliaEnvWrapper

from sandbox import RLLabRunner

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.envs.base import TfEnv

from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

from sandbox.rocky.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from sandbox.rocky.tf.policies.gaussian_gru_policy import GaussianGRUPolicy

from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp

import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--baseline_type',type=str,default='linear')
parser.add_argument('--exp_name',type=str,default='my_exp')
parser.add_argument('--tabular_log_file',type=str,default= 'tab.txt')
parser.add_argument('--text_log_file',type=str,default= 'tex.txt')
parser.add_argument('--params_log_file',type=str,default= 'args.txt')
parser.add_argument('--snapshot_mode',type=str,default='all')
parser.add_argument('--log_tabular_only',type=bool,default=False)
parser.add_argument('--log_dir',type=str)

parser.add_argument('--args_data',type=str)

args = parser.parse_args()

if True:
    env_name = "Auto1D"
    env_id = "Auto1D-v0"
else:
    env_name = "Auto2D"
    env_id = "Auto2D-v0"

env_dict = {'trajdata_indeces': [0]}
JuliaEnvWrapper.set_initials(env_name, 1, {})
gym.envs.register(
    id=env_id,
    entry_point='rltools.envs.julia_sim:JuliaEnvWrapper',
    timestep_limit=999,
    reward_threshold=195.0,
)

g_env = GymEnv(env_id) # this works
env = TfEnv(g_env) # this works

feat_mlp = MLP('encoder', 100, [100], tf.nn.tanh, tf.nn.tanh,
               input_shape= (np.prod(env.spec.observation_space.shape),))
policy = GaussianGRUPolicy(name= 'gru_policy', env_spec= env.spec,
                           hidden_dim=32,
                          feature_network=feat_mlp,
                          state_include_action=False)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=100,
    n_itr=10,
    discount=0.99,
    step_size=0.01,
    force_batch_sampler= True,
    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
)

runner = RLLabRunner(algo,args)
runner.train()

halt= True