import gym
import rltools.util
import joblib
from rltools.envs.julia_sim import JuliaEnvWrapper
import tensorflow as tf

import h5py

from rllab.sampler.utils import rollout

env_dict = {'trajdata_indeces': [3],
        'use_playback_reactive': True,
        'extract_core':True,
        'extract_temporal':False,
        'extract_well_behaved':True,
        'extract_neighbor_features':False,
        'extract_carlidar_rangerate':True,
        'carlidar_nbeams':20,
        'roadlidar_nbeams':0,
        'roadlidar_nlanes':2,
        'carlidar_max_range':100.0,
        'roadlidar_max_range':100.}

JuliaEnvWrapper.set_initials("Auto2D",1,env_dict)
gym.envs.register(
        id="Auto2D-v0",
        entry_point='rltools.envs.julia_sim:JuliaEnvWrapper',
        timestep_limit=999,
        reward_threshold=195.0,)


with tf.Session() as sess:
	x = joblib.load('16-11-04/radar_gru-0/itr_450.pkl')
	pi = x['policy']
	halt= True
	key = 'iter{:05}/'.format(0)
	with h5py.File('pleasework.h5', 'a') as hf:
		if key in hf:
			dset = hf[key]
		else:
			dset = hf.create_group(key)
		vs = pi.get_params()
		vals = sess.run(vs)

		for v, val in zip(vs, vals):
			dset[v.name] = val

	#while True:
		#rollout(x['env'],x['policy'],max_path_length=100,animated=True)

#policy = x['policy']
#policy.set_

#import pdb
#pdb.set_trace()
