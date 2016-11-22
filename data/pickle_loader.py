import gym
import rltools.util
import joblib
from rltools.envs.julia_sim import JuliaEnvWrapper
import tensorflow as tf
import numpy as np

import h5py

import sandbox.rocky.tf.core.layers as L

from rllab.sampler.utils import rollout

env_dict = {'trajdata_indeces': [1],
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
#	x = joblib.load('16-11-04/radar_gru-3/itr_417.pkl') # 417 looked good
	with h5py.File("16-11-04/radar_gru-3/policy_gail.h5","r") as hf:
		import pdb; pdb.set_trace()
		halt= True

#	x = joblib.load('16-11-04/radar_gru-2/itr_417.pkl')
	#pi = x['policy']
	#halt= True
	key = 'iter{:05}/'.format(0)
	with h5py.File('goodpolicy.h5', 'a') as hf:
		if key in hf:
			dset = hf[key]
		else:
			dset = hf.create_group(key)
		vs = pi.get_params()
		vals = sess.run(vs)

		#for v, val in zip(vs, vals):
			#dset[v.name] = val
	#halt= True
	#activations = []
	#z = np.zeros((1,51))
	#with h5py.File('activations.h5', 'a') as hf:
	#	for layer in pi.feature_network.layers[1:]:
	#		a = sess.run(L.get_output(layer),
	#		             {pi.feature_network.input_var: z})
	#		hf.create_dataset(layer.name, data=a)


	while True:
		rollout(x['env'],x['policy'],max_path_length=100,animated=True)

#policy = x['policy']
#policy.set_

#import pdb
#pdb.set_trace()
