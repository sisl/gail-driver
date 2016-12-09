from rllab.misc import ext
from rllab.misc.overrides import overrides

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.optimizers.first_order_optimizer import Solver, SimpleSolver

from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.misc import tensor_utils

import tensorflow as tf
import numpy as np

class GAIL(TRPO):
    """
    Generative adversarial imitation learning.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            fo_optimizer_cls=None,
            fo_optimizer_args=None,
            reward=None,
            expert_data=None,
            gail_batch_size=100,
            adam_steps=1,
            decay_steps=1,
            decay_rate=1.0,
            act_mean = 0.0,
            act_std = 1.0,
            hard_freeze = True,
            freeze_upper = 1.0,
            freeze_lower = 0.5,
            include_safety= False,
            temporal_indices=None,
	    temporal_noise_thresh=100,
            **kwargs):

        super(GAIL, self).__init__(optimizer=optimizer, optimizer_args=optimizer_args, **kwargs)
        self.reward_model = reward
        self.expert_data = expert_data
        self.gail_batch_size = gail_batch_size

        self.act_mean = act_mean
        self.act_std = act_std

        self.include_safety = include_safety
        self.temporal_indices = temporal_indices
	self.temporal_noise_thresh = temporal_noise_thresh

        #self.global_step = tf.Variable(0, trainable=False, name='global_step')
        #decayed_learning_rate = tf.train.exponential_decay(fo_optimizer_args['learning_rate'], self.global_step.initialized_value(),
                                                           #decay_steps, decay_rate, staircase=True, name= 'dlr')

        #variable_learning_rate = tf.Variable(5e-4, name= 'vlr')

        #self.assign_lr_to_vlr = variable_learning_rate.assign(decayed_learning_rate)
        #self.assign_zero_to_vlr = variable_learning_rate.assign(0.0)
        #self.assign_cur = self.assign_lr_to_vlr

        #fo_optimizer_args['learning_rate']= variable_learning_rate

        self.background_lr = fo_optimizer_args['learning_rate']
        self.working_lr = fo_optimizer_args['learning_rate']
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

        self.hard_freeze = hard_freeze
        self.freeze_upper = freeze_upper
        self.freeze_lower = freeze_lower

        #self.solver = Solver(self.reward_model, 0.0, 0.0, adam_steps, self.gail_batch_size, None,
                             #tf_optimizer_cls= fo_optimizer_cls, tf_optimizer_args= fo_optimizer_args)

        self.solver = SimpleSolver(self.reward_model, adam_steps, self.gail_batch_size)

        """
        obs_ex= self.expert_data['obs']
        act_ex= self.expert_data['act']

        p1 = np.random.choice(obs_ex.shape[0], size=(self.batch_size,), replace=False)
        p2 = np.random.choice(trajbatch.obs.stacked.shape[0], size=(self.batch_size,), replace=False)

        ex_obs_batch = obs_ex[p1]
        ex_act_batch = act_ex[p1]

        pi_obs_batch = trajbatch.obs.stacked[p2]
        pi_act_batch = trajbatch.a.stacked[p2]

        reward_print_fields= self.reward.fit(sess, pi_obs_batch, pi_act_batch,
                                             ex_obs_batch, ex_act_batch)

        self.reward.fit(sess, pi_obs_batch, pi_act_batch, ex_obs_batch, ex_act_batch)
        """

    @overrides
    def optimize_policy(self, itr, samples_data):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)

        # update policy
        super(GAIL, self).optimize_policy_from_inputs(all_input_values)

        # update discriminator
        obs_pi = all_input_values[0].reshape((-1, np.prod(self.env.observation_space.shape)))
        act_pi = all_input_values[1].reshape((-1, np.prod(self.env.action_space.shape)))

	# filter zero rows
	nonzero_ix = ~np.all(obs_pi == 0, axis=1)
	obs_pi = obs_pi[nonzero_ix]
	act_pi = act_pi[nonzero_ix]

        # normalize actions . observations are normalized by environment
        act_pi -= self.act_mean
        act_pi /= self.act_std

        obs_ex= self.expert_data['obs']
        act_ex= self.expert_data['act']
	
        if itr < self.temporal_noise_thresh and self.temporal_indices is not None:
            obs_ex[:,self.temporal_indices] = np.random.normal(0,1,obs_ex[:,self.temporal_indices].shape)
            #obs_ex[:,self.temporal_indices] = np.zeros_like(obs_ex[:,self.temporal_indices])

        p_ex = np.random.choice(obs_ex.shape[0], size=(self.gail_batch_size,), replace=False)
        p_pi = np.random.choice(obs_pi.shape[0], size=(self.gail_batch_size,), replace=False)

        obs_pi_batch = obs_pi[p_pi]
        act_pi_batch = act_pi[p_pi]

        obs_ex_batch = obs_ex[p_ex]
        act_ex_batch = act_ex[p_ex]
        trans_Bpi_Do = np.column_stack((obs_pi_batch,act_pi_batch))
        trans_Bex_Do = np.column_stack((obs_ex_batch,act_ex_batch))

        if self.include_safety:
            trans_B_Do = np.row_stack((trans_Bpi_Do[:,:-1],trans_Bex_Do))
        else:
            trans_B_Do = np.row_stack((trans_Bpi_Do,trans_Bex_Do))

        Bpi = trans_Bpi_Do.shape[0] # policy batch size.
        Bex = trans_Bex_Do.shape[0] # expert batch size.

        assert Bpi == Bex

        labels= np.concatenate((np.zeros(Bpi), np.ones(Bex)))[...,None] # 0s for policy, 1s for expert
        # weights=np.concatenate((np.ones(Bpi)/Bpi, np.ones(Bex)/Bex))

        #self.solver.train(trans_B_Do, labels, assign_vlr= self.assign_cur)

        #lr = self.background_lr * decay_rate ** (global_step / decay_steps)
        self.solver.train(trans_B_Do, labels, self.working_lr)
        scores = self.reward_model.compute_score(trans_B_Do)

        #self.global_step += 1

        #accuracy = .5 * ((scores < 0) == (labels == 0)).sum()
        accuracy = ((scores < 0) == (labels == 0)).mean()
        accuracy_for_currpolicy = (scores[:Bpi] <= 0).mean()
        accuracy_for_expert = (scores[Bpi:] > 0).mean()
        assert np.allclose(accuracy, .5*(accuracy_for_currpolicy + accuracy_for_expert))

        if self.hard_freeze:
            if accuracy >= self.freeze_upper:
                self.working_lr = 0.0
            if accuracy <= self.freeze_lower:
                self.working_lr = self.background_lr #* np.power(self.decay_rate, itr / self.decay_steps)
        else:
            self.working_lr *= np.maximum(1, np.minimum(0,
                                                        (accuracy-self.freeze_upper)/
                                                        (self.freeze_lower-self.freeze_upper)))

        self.working_lr = self.working_lr * np.power(self.decay_rate, itr/self.decay_steps)

        # assign a new mean activation on expert data
        #mu_ex = tf.reduce_sum(self.matching_layer * tf.expand_dims(self.targets_B,-1),
                              #reduction_indices= 0) / tf.reduce_sum(self.targets_B)
        #sess.run(self.mu_ex.assign(mu_ex), feed)

        logger.record_tabular('working_lr', self.working_lr)
        logger.record_tabular('background_lr', self.background_lr)
        logger.record_tabular('racc', accuracy)
        logger.record_tabular('raccpi', accuracy_for_currpolicy)
        logger.record_tabular('raccex', accuracy_for_expert)

        return dict()

    @overrides
    def process_samples(self, itr, paths):
        path_lengths = []
        if self.include_safety:
            fraction_safe = []

        for path in paths:
            if itr < self.temporal_noise_thresh and self.temporal_indices is not None:
                path['observations'][:,self.temporal_indices] = \
			np.random.normal(0,1,path['observations'][:,self.temporal_indices].shape)
			#inp.zeros_like(path['observations'][:,self.temporal_indices])

            X = np.column_stack((path['observations'],path['actions']))
            if self.include_safety:
                X = X[:,:-1]

            path['env_rewards'] = path['rewards']
            rewards = np.squeeze( self.reward_model.compute_reward(X) )

            if rewards.ndim == 0:
                rewards = rewards[np.newaxis]
            path['rewards'] = rewards
            if self.include_safety:
                assert path['actions'].shape[-1] == 3
                acts = path['actions'][:,-1]
                fraction_safe.append(
                    (acts > 0).sum() / float(len(acts))
                    )

            path_lengths.append(X.shape[0])

        assert all([path['rewards'].ndim == 1 for path in paths])
        if self.include_safety:
            logger.record_tabular('aveFactionSafe', np.mean(fraction_safe))
        logger.record_tabular('pathLengths',np.mean(path_lengths))
        return self.sampler.process_samples(itr, paths)

