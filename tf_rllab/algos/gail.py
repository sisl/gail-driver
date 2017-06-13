from rllab.misc import ext
from rllab.misc.overrides import overrides

from tf_rllab.algos.trpo import TRPO
from tf_rllab.optimizers.first_order_optimizer import Solver, SimpleSolver

from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from tf_rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from tf_rllab.algos.batch_polopt import BatchPolopt
from tf_rllab.misc import tensor_utils

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
            act_mean=0.0,
            act_std=1.0,
            hard_freeze=True,
            freeze_upper=1.0,
            freeze_lower=0.5,
            temporal_indices=None,
        temporal_noise_thresh=100,
        wgan=False,
            **kwargs):
        kwargs['temporal_noise_thresh'] = temporal_noise_thresh
        super(GAIL, self).__init__(optimizer=optimizer,
                                   optimizer_args=optimizer_args, **kwargs)
        self.reward_model = reward
        self.expert_data = expert_data
        self.gail_batch_size = gail_batch_size

        self.act_mean = act_mean
        self.act_std = act_std

        self.wgan = wgan

        self.temporal_indices = temporal_indices
        self.temporal_noise_thresh = temporal_noise_thresh

        self.background_lr = fo_optimizer_args['learning_rate']
        self.working_lr = fo_optimizer_args['learning_rate']
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

        self.hard_freeze = hard_freeze
        self.freeze_upper = freeze_upper
        self.freeze_lower = freeze_lower

        self.solver = SimpleSolver(
            self.reward_model, adam_steps, self.gail_batch_size)

    @overrides
    def optimize_policy(self, itr, samples_data):
        """
        Perform policy optimization with TRPO, then draw samples from paths
        to fit discriminator/surrogate reward network.
        """

        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k]
                          for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)

        # update policy
        super(GAIL, self).optimize_policy_from_inputs(all_input_values)

        # update discriminator
        obs_pi = all_input_values[0].reshape(
            (-1, np.prod(self.env.observation_space.shape)))
        act_pi = all_input_values[1].reshape(
            (-1, np.prod(self.env.action_space.shape)))

        # filter zero rows
        nonzero_ix = ~np.all(obs_pi == 0, axis=1)
        obs_pi = obs_pi[nonzero_ix]
        act_pi = act_pi[nonzero_ix]

        # normalize actions . observations are normalized by environment
        act_pi -= self.act_mean
        act_pi /= self.act_std

        obs_ex = self.expert_data['obs']
        act_ex = self.expert_data['act']

        # replace temporal features with noise up until later epoch
        if itr < self.temporal_noise_thresh and self.temporal_indices is not None:
            obs_ex[:, self.temporal_indices] = np.random.normal(
                0, 1, obs_ex[:, self.temporal_indices].shape)

        p_ex = np.random.choice(
            obs_ex.shape[0], size=(
                self.gail_batch_size,), replace=False)
        p_pi = np.random.choice(
            obs_pi.shape[0], size=(
                self.gail_batch_size,), replace=False)

        obs_pi_batch = obs_pi[p_pi]
        act_pi_batch = act_pi[p_pi]

        obs_ex_batch = obs_ex[p_ex]
        act_ex_batch = act_ex[p_ex]

        trans_Bpi_Do = np.column_stack((obs_pi_batch, act_pi_batch))
        trans_Bex_Do = np.column_stack((obs_ex_batch, act_ex_batch))

        trans_B_Do = np.row_stack((trans_Bpi_Do, trans_Bex_Do))

        Bpi = trans_Bpi_Do.shape[0]  # policy batch size.
        Bex = trans_Bex_Do.shape[0]  # expert batch size.

        assert Bpi == Bex

        if self.wgan:
            # 1s for policy, (-1)s for expert
            labels = np.concatenate((np.ones(Bpi), -np.ones(Bex)))[..., None]
        else:
            # 0s for policy, 1s for expert
            labels = np.concatenate((np.zeros(Bpi), np.ones(Bex)))[..., None]

        self.solver.train(trans_B_Do, labels, self.working_lr)
        scores = self.reward_model.compute_score(trans_B_Do)

        accuracy_for_currpolicy = (scores[:Bpi] <= 0).mean()
        accuracy_for_expert = (scores[Bpi:] > 0).mean()
        accuracy = .5 * (accuracy_for_currpolicy + accuracy_for_expert)

        if self.hard_freeze:
            if accuracy >= self.freeze_upper:
                self.working_lr = 0.0
            if accuracy <= self.freeze_lower:
                # * np.power(self.decay_rate, itr / self.decay_steps)
                self.working_lr = self.background_lr
        else:
            self.working_lr *= np.maximum(1, np.minimum(0,
                                                        (accuracy - self.freeze_upper) /
                                                        (self.freeze_lower - self.freeze_upper)))

        self.working_lr = self.working_lr * \
            np.power(self.decay_rate, itr / self.decay_steps)

        logger.record_tabular('working_lr', self.working_lr)
        logger.record_tabular('background_lr', self.background_lr)
        logger.record_tabular('racc', accuracy)
        logger.record_tabular('raccpi', accuracy_for_currpolicy)
        logger.record_tabular('raccex', accuracy_for_expert)

        return dict()

    @overrides
    def process_samples(self, itr, paths):
        path_lengths = []

        for path in paths:
            X = np.column_stack((path['observations'], path['actions']))

            path['env_rewards'] = path['rewards']
            rewards = np.squeeze(self.reward_model.compute_reward(X))

            if rewards.ndim == 0:
                rewards = rewards[np.newaxis]
            path['rewards'] += rewards

            path_lengths.append(X.shape[0])

        assert all([path['rewards'].ndim == 1 for path in paths])
        logger.record_tabular('pathLengths', np.mean(path_lengths))
        return self.sampler.process_samples(itr, paths)
