import gym
import argparse
import calendar

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize

from rllab.envs.gym_env import GymEnv
from rllab.envs.tf_env import TfEnv

import rltools.util
from rltools.envs.julia_sim import JuliaEnvWrapper, JuliaEnv

from tf_rllab import RLLabRunner

from tf_rllab.algos.trpo import TRPO
from tf_rllab.algos.gail import GAIL

from tf_rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from tf_rllab.policies.gaussian_gru_policy import GaussianGRUPolicy

from tf_rllab.core.network import MLP, RewardMLP, WassersteinMLP, BaselineMLP
from tf_rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp

import tensorflow as tf
import numpy as np
import os

import os.path as osp
from rllab import config

parser = argparse.ArgumentParser()
# Logger Params
parser.add_argument('--exp_name', type=str, default='gail_exp')
parser.add_argument('--tabular_log_file', type=str, default='tab.txt')
parser.add_argument('--text_log_file', type=str, default='tex.txt')
parser.add_argument('--params_log_file', type=str, default='args.txt')
parser.add_argument('--snapshot_mode', type=str, default='all')
parser.add_argument('--log_tabular_only', type=bool, default=False)
parser.add_argument('--log_dir', type=str)
parser.add_argument('--args_data')

# Environment params
parser.add_argument('--trajdatas', type=int, nargs='+',
                    default=[1, 2, 3, 4, 5, 6])
parser.add_argument('--limit_trajs', type=int, default=12000)
# max length of a trajectory (ts)
parser.add_argument('--max_traj_len', type=int, default=100)
parser.add_argument('--env_name', type=str, default="Auto2D")
parser.add_argument('--normalize_obs', type=bool, default=True)
parser.add_argument('--normalize_act', type=bool, default=False)
parser.add_argument('--norm_tol', type=float, default=1e-1)

# Env dict
parser.add_argument('--use_playback_reactive', type=bool, default=False)

parser.add_argument('--extract_core', type=int, default=1)
parser.add_argument('--extract_well_behaved', type=int, default=1)
parser.add_argument('--extract_neighbor_features', type=int, default=0)
parser.add_argument('--extract_carlidar', type=int, default=1)
parser.add_argument('--extract_roadlidar', type=int, default=0)
parser.add_argument('--extract_carlidar_rangerate', type=int, default=1)

parser.add_argument('--carlidar_nbeams', type=int, default=20)
parser.add_argument('--roadlidar_nbeams', type=int, default=20)

parser.add_argument('--temporal_noise_thresh', type=int, default=100)

# Reward weights
parser.add_argument('--col_weight', type=float, default=0.0)
parser.add_argument('--off_weight', type=float, default=0.0)
parser.add_argument('--rev_weight', type=float, default=0.0)
parser.add_argument('--jrk_weight', type=float, default=0.0)
parser.add_argument('--acc_weight', type=float, default=0.0)
parser.add_argument('--cen_weight', type=float, default=0.0)
parser.add_argument('--ome_weight', type=float, default=0.0)

# Model Params
parser.add_argument('--policy_type',type=str,default='mlp')
parser.add_argument('--reward_type',type=str,default='mlp')
parser.add_argument('--policy_save_name',type=str,default='policy_gail')
parser.add_argument('--policy_ckpt_name',type=str,default=None)
parser.add_argument('--policy_ckpt_itr',type=int,default=1)
parser.add_argument('--baseline_type',type=str,default='linear')
parser.add_argument('--load_policy',type=bool,default=False)

parser.add_argument('--hspec',type=int,nargs='+',default=[256,128,64,64,32]) # specifies architecture of "feature" networks
parser.add_argument('--p_hspec',type=int,nargs='+',default=[]) # policy layers
parser.add_argument('--b_hspec',type=int,nargs='+',default=[]) # baseline layers
parser.add_argument('--r_hspec',type=int,nargs='+',default=[]) # reward layers

parser.add_argument('--gru_dim',type=int,default=32) # hidden dimension of gru

parser.add_argument('--nonlinearity',type=str,default='elu')
# TRPO Params
parser.add_argument('--trpo_batch_size', type=int, default=40 * 100)
parser.add_argument('--discount', type=float, default=0.95)
parser.add_argument('--gae_lambda', type=float, default=0.99)
parser.add_argument('--n_iter', type=int, default=500)  # trpo iterations
parser.add_argument('--max_kl', type=float, default=0.01)
parser.add_argument('--vf_max_kl', type=float, default=0.01)
parser.add_argument('--vf_cg_damping', type=float, default=0.01)
parser.add_argument('--trpo_step_size', type=float, default=0.01)

# GAILS Params
parser.add_argument('--gail_batch_size', type=int, default=1024)
parser.add_argument('--adam_steps', type=int, default=1)
parser.add_argument('--adam_lr', type=float, default=0.00005)
parser.add_argument('--adam_beta1', type=float, default=0.9)
parser.add_argument('--adam_beta2', type=float, default=0.99)
parser.add_argument('--adam_epsilon', type=float, default=1e-8)
parser.add_argument('--decay_steps', type=int, default=1)
parser.add_argument('--decay_rate', type=float, default=1.0)
# freeze learning rate when discriminator reaches threshold
parser.add_argument('--hard_freeze', type=bool, default=False)
parser.add_argument('--freeze_upper', type=float, default=1.0)
parser.add_argument('--freeze_lower', type=float, default=0.5)
parser.add_argument('--policy_ent_reg', type=float, default=0.0)
parser.add_argument('--env_r_weight', type=float, default=0.0)

# Load ckpt?
parser.add_argument('--ckpt_name', type=str, default='')
parser.add_argument('--ckpt_itr', type=int, default=-1)

args = parser.parse_args()

from rllab.config_personal import expert_trajs_path, model_path

if args.nonlinearity == 'tanh':
    nonlinearity = tf.nn.tanh
elif args.nonlinearity == 'relu':
    nonlinearity = tf.nn.relu
elif args.nonlinearity == 'elu':
    nonlinearity = tf.nn.elu
elif args.nonlinearity == "sigmoid":
    nonlinearity = tf.nn.sigmoid
else:
    raise NotImplementedError

if args.hspec is None:
    p_hspec = args.p_hspec
    b_hspec = args.b_hspec
    r_hspec = args.r_hspec
else:
    p_hspec = args.hspec
    b_hspec = args.hspec
    r_hspec = args.hspec

env_id = "Auto2D-v0"
expert_data_path = expert_trajs_path + \
    '/core{}_temp{}_well{}_neig{}_carl{}_roal{}_clrr{}_mtl{}_clb{}_rlb{}_rll{}_clmr{}_rlmr{}_seed{}.h5'.format(
        int(args.extract_core), 0, int(args.extract_well_behaved),
        int(args.extract_neighbor_features), int(
            args.extract_carlidar), int(args.extract_roadlidar),
        int(args.extract_carlidar_rangerate), 100, args.carlidar_nbeams, args.roadlidar_nbeams, 2, 100, 50, 456)

env_dict = {'trajdata_indeces': args.trajdatas,
            'col_weight': args.col_weight,
            'off_weight': args.off_weight,
            'rev_weight': args.rev_weight,
            'jrk_weight': args.jrk_weight,
            'acc_weight': args.acc_weight,
            'cen_weight': args.cen_weight,
            'ome_weight': args.ome_weight,
            'nsteps': args.max_traj_len,
            'use_playback_reactive': args.use_playback_reactive,
            'extract_core': bool(args.extract_core),
            'extract_temporal': False,
            'extract_well_behaved': bool(args.extract_well_behaved),
            'extract_neighbor_features': bool(args.extract_neighbor_features),
            'extract_carlidar_rangerate': bool(args.extract_carlidar_rangerate),
            'carlidar_nbeams': args.carlidar_nbeams,
            'roadlidar_nbeams': args.roadlidar_nbeams,
            'roadlidar_nlanes': 2,
            'carlidar_max_range': 100.,
            'roadlidar_max_range': 100.}

if not args.extract_carlidar:
    env_dict['carlidar_nbeams'] = 0
if not args.extract_roadlidar:
    env_dict['roadlidar_nbeams'] = 0

JuliaEnvWrapper.set_initials(args.env_name, 1, env_dict)
gym.envs.register(
    id=env_id,
    entry_point='rltools.envs.julia_sim:JuliaEnvWrapper',
    timestep_limit=999,
    reward_threshold=195.0,
)

expert_data, _ = rltools.util.load_trajs(
    expert_data_path, args.limit_trajs, swap=False)
expert_data_stacked = rltools.util.prepare_trajs(expert_data['exobs_B_T_Do'], expert_data['exa_B_T_Da'], expert_data['exlen_B'],
                                                 labeller=None)

# normalization statistics extracted from expert dataset
initial_obs_mean = expert_data_stacked['exobs_Bstacked_Do'].mean(axis=0)
initial_obs_std = expert_data_stacked['exobs_Bstacked_Do'].std(axis=0)
initial_obs_std[initial_obs_std < args.norm_tol] = 1.0
initial_obs_var = np.square(initial_obs_std)

# create normalize environments
g_env = normalize(GymEnv(env_id),
                  initial_obs_mean=initial_obs_mean,
                  initial_obs_var=initial_obs_var,
                  normalize_obs=True,
                  running_obs=False)
env = TfEnv(g_env)

# normalize observations
if args.normalize_obs:
    expert_data = {'obs': (
        expert_data_stacked['exobs_Bstacked_Do'] - initial_obs_mean) / initial_obs_std}
else:
    expert_data = {'obs': expert_data_stacked['exobs_Bstacked_Do']}

# normalize actions
if args.normalize_act:
    initial_act_mean = expert_data_stacked['exa_Bstacked_Da'].mean(axis=0)
    initial_act_std = expert_data_stacked['exa_Bstacked_Da'].std(axis=0)

    expert_data.update({'act': (
        expert_data_stacked['exa_Bstacked_Da'] - initial_act_mean) / initial_act_std})
else:
    initial_act_mean = 0.0
    initial_act_std = 1.0

    expert_data.update({'act': expert_data_stacked['exa_Bstacked_Da']})

# compute dimensions of convolution-component and no-conv features.
dense_input_shape = (env_dict["extract_core"] * 8 +
                     env_dict["extract_well_behaved"] * 3 + env_dict["extract_neighbor_features"] * 28,)

roadlidar_input_shape = (1, env_dict["roadlidar_nbeams"], 1)

if env_dict["extract_temporal"]:
    temporal_indices = range(0, 6)
    if env_dict["extract_core"]:
        temporal_indices = [t_ix + 8 for t_ix in temporal_indices]

else:
    temporal_indices = None

# create policy
if args.policy_type == 'mlp':
    policy = GaussianMLPPolicy('mlp_policy', env.spec, hidden_sizes=p_hspec,
                               std_hidden_nonlinearity=nonlinearity, hidden_nonlinearity=nonlinearity
                               )
    if args.policy_ckpt_name is not None:
        with tf.Session() as sess:
            policy.load_params(args.policy_ckpt_name, args.policy_ckpt_itr)

elif args.policy_type == 'gru':
    if p_hspec == []:
        feat_mlp = None
    else:
        feat_mlp = MLP('mlp_policy', p_hspec[-1], p_hspec[:-1], nonlinearity, nonlinearity,
                       input_shape=(np.prod(env.spec.observation_space.shape),))

    policy = GaussianGRUPolicy(name='gru_policy', env_spec=env.spec,
                               hidden_dim=args.gru_dim,
                               feature_network=feat_mlp,
                               state_include_action=False)

else:
    raise NotImplementedError

# TODO: Add naming convention
policy.save_name = args.policy_save_name

# create baseline
if args.baseline_type == 'linear':
    baseline = LinearFeatureBaseline(env_spec=env.spec)

elif args.baseline_type == 'mlp':
    baseline = BaselineMLP(name='mlp_baseline',
                           output_dim=1,
                           hidden_sizes=b_hspec,
                           hidden_nonlinearity=nonlinearity,
                           output_nonlinearity=None,
                           input_shape=(np.prod(env.spec.observation_space.shape),))
    baseline.initialize_optimizer()
else:
    raise NotImplementedError

# create adversary
if args.reward_type == 'wgan':
	assert args.adam_steps > 1
	reward = WassersteinMLP('mlp_reward', 1, r_hspec, nonlinearity, None,
			           input_shape= (np.prod(env.spec.observation_space.shape) + env.action_dim,)
			           )
	fo_optimizer_cls = tf.train.RMSPropOptimizer
	fo_optimizer_args = dict(learning_rate = args.adam_lr)
elif args.reward_type == 'mlp':
	reward = RewardMLP('mlp_reward', 1, r_hspec, nonlinearity, tf.nn.sigmoid,
			           input_shape= (np.prod(env.spec.observation_space.shape) + env.action_dim,)
			           )	
	fo_optimizer_cls = tf.train.AdamOptimizer
	fo_optimizer_args = dict(learning_rate = args.adam_lr,
							beta1 = args.adam_beta1,
							beta2 = args.adam_beta2,
							epsilon= args.adam_epsilon)
else:
	raise NotImplementedError

algo = GAIL(
	env=env,
	policy=policy,
	baseline=baseline,
	reward=reward,
	expert_data=expert_data,
	batch_size= args.trpo_batch_size,
	gail_batch_size=args.gail_batch_size,
	max_path_length=args.max_traj_len,
	n_itr=args.n_iter,
	discount=args.discount,
	step_size=args.trpo_step_size,
	force_batch_sampler= True,
	whole_paths= True,
	adam_steps= args.adam_steps,
	decay_rate= args.decay_rate,
	decay_steps= args.decay_steps,
	act_mean= initial_act_mean,
	act_std= initial_act_std,
	freeze_upper = args.freeze_upper,
	freeze_lower = args.freeze_lower,
	fo_optimizer_cls=fo_optimizer_cls,
	load_params_args = None,
	temporal_indices = temporal_indices,
	temporal_noise_thresh = args.temporal_noise_thresh,
	fo_optimizer_args= fo_optimizer_args,
    wgan = args.reward_type == 'wgan',
	optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
)

# Load checkpoint if desired
if len(args.ckpt_name) > 0:
    policy.load_params(args.ckpt_name, args.ckpt_itr, [])
    baseline.load_params(args.ckpt_name, args.ckpt_itr, [])
    reward.load_params(args.ckpt_name, args.ckpt_itr, [])

# use date and time to create new logging directory for each run
date = calendar.datetime.date.today().strftime('%y-%m-%d')
if date not in os.listdir(model_path):
    os.mkdir(model_path + '/' + date)

c = 0
exp_name = args.exp_name + '-' + str(c)

while exp_name in os.listdir(model_path + '/' + date + '/'):
    c += 1
    exp_name = args.exp_name + '-' + str(c)

exp_dir = date + '/' + exp_name
log_dir = osp.join(config.LOG_DIR, exp_dir)

policy.set_log_dir(log_dir)

# run experiment
runner = RLLabRunner(algo, args, exp_dir)
policy.save_extra_data(["initial_obs_mean", "initial_obs_std"], [
                       initial_obs_mean, initial_obs_std])
runner.train()
