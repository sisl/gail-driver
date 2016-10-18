from rllab.misc import ext
from rllab.misc.overrides import overrides

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.optimizers.first_order_optimizer import Solver

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
            **kwargs):

        super(GAIL, self).__init__(optimizer=optimizer, optimizer_args=optimizer_args, **kwargs)
        self.reward_model = reward
        self.expert_data = expert_data
        self.gail_batch_size = gail_batch_size
        self.solver = Solver(self.reward_model, 0.0, 0.0, adam_steps, self.gail_batch_size, None,
                             tf_optimizer_cls= fo_optimizer_cls, tf_optimizer_args= fo_optimizer_args)

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

        obs_ex= self.expert_data['obs']
        act_ex= self.expert_data['act']

        p_ex = np.random.choice(obs_ex.shape[0], size=(self.gail_batch_size,), replace=False)
        p_pi = np.random.choice(obs_pi.shape[0], size=(self.gail_batch_size,), replace=False)

        obs_pi_batch = obs_pi[p_pi]
        act_pi_batch = act_pi[p_pi]

        obs_ex_batch = obs_ex[p_ex]
        act_ex_batch = act_ex[p_ex]

        trans_Bpi_Do = np.column_stack((obs_pi_batch,act_pi_batch))
        trans_Bex_Do = np.column_stack((obs_ex_batch,act_ex_batch))
        trans_B_Do = np.row_stack((trans_Bpi_Do,trans_Bex_Do))

        Bpi = trans_Bpi_Do.shape[0] # policy batch size.
        Bex = trans_Bex_Do.shape[0] # expert batch size.

        assert Bpi == Bex

        labels= np.concatenate((np.zeros(Bpi), np.ones(Bex)))[...,None] # 0s for policy, 1s for expert
        # weights=np.concatenate((np.ones(Bpi)/Bpi, np.ones(Bex)/Bex))

        self.solver.train(trans_B_Do, labels)
        scores = self.reward_model.compute_score(trans_B_Do)

        #accuracy = .5 * ((scores < 0) == (labels == 0)).sum()
        accuracy = ((scores < 0) == (labels == 0)).mean()
        accuracy_for_currpolicy = (scores[:Bpi] <= 0).mean()
        accuracy_for_expert = (scores[Bpi:] > 0).mean()
        assert np.allclose(accuracy, .5*(accuracy_for_currpolicy + accuracy_for_expert))

        # assign a new mean activation on expert data
        #mu_ex = tf.reduce_sum(self.matching_layer * tf.expand_dims(self.targets_B,-1),
                              #reduction_indices= 0) / tf.reduce_sum(self.targets_B)
        #sess.run(self.mu_ex.assign(mu_ex), feed)

        logger.record_tabular('racc', accuracy)
        logger.record_tabular('raccpi', accuracy_for_currpolicy)
        logger.record_tabular('raccex', accuracy_for_expert)

        return dict()

#    def fit(self, sess, obs_pi, act_pi, obs_ex, act_ex):
        #trans_Bpi_Do = np.column_stack((obs_pi,act_pi))
        #trans_Bex_Do = np.column_stack((obs_ex,act_ex))
        #trans_B_Do = np.row_stack((trans_Bpi_Do,trans_Bex_Do))

        ##Normalize obs and acts together.
        #self.transnorm.update(trans_B_Do, sess= sess)
        #strans_B_Do = self.transnorm.standardize(trans_B_Do, sess= sess)

        #Bpi = trans_Bpi_Do.shape[0] # policy batch size.
        #Bex = trans_Bex_Do.shape[0] # expert batch size.

        #assert Bpi == Bex

        #labels= np.concatenate((np.zeros(Bpi), np.ones(Bex))) # 0s for policy, 1s for expert
        #weights=np.concatenate((np.ones(Bpi)/Bpi, np.ones(Bex)/Bex))

        ##What is Df versus Do?
        #feed = {self.transfeat_B_Df: strans_B_Do,
                #self.targets_B: labels,
                #self.weights_B: weights}
        ##if z_pi is not None and z_ex is not None:
            ##zfeat_B_z = np.row_stack((z_pi, z_ex))
            ##feed[self._zfeat_B_z]= zfeat_B_z

        #for _ in range(self._adam_steps):
            #loss, optout, scores, reward = sess.run([self.obj, self._adam_opt, self.scores_B, self.rew_B], feed)

        ##Update decay variables
        #self.global_step += 1

        #accuracy = .5 * (weights * ((scores < 0) == (labels == 0))).sum()
        #accuracy_for_currpolicy = (scores[:Bpi] <= 0).mean()
        #accuracy_for_expert = (scores[Bpi:] > 0).mean()
        #assert np.allclose(accuracy, .5*(accuracy_for_currpolicy + accuracy_for_expert))

        ## assign a new mean activation on expert data
        #mu_ex = tf.reduce_sum(self.matching_layer * tf.expand_dims(self.targets_B,-1),
                              #reduction_indices= 0) / tf.reduce_sum(self.targets_B)
        #sess.run(self.mu_ex.assign(mu_ex), feed)

        #return [
            #('rloss', loss, float), # reward function fitting loss
            #('racc', accuracy, float), # reward function accuracy
            #('raccpi', accuracy_for_currpolicy, float), # reward function accuracy
            #('raccex', accuracy_for_expert, float), # reward function accuracy
        #]

    @overrides
    def process_samples(self, itr, paths):
        for path in paths:
            X = np.column_stack((path['observations'],path['actions']))
            path['env_rewards'] = path['rewards']
            rewards = np.squeeze( self.reward_model.compute_reward(X) )
            if rewards.ndim == 0:
                rewards = rewards[np.newaxis]
            path['rewards'] = rewards

        return self.sampler.process_samples(itr, paths)

