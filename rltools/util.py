from __future__ import print_function

import errno
import os
import timeit

import h5py
import numpy as np
from colorama import Fore, Style

from gym.spaces import Discrete, Box


class Timer(object):

    def __enter__(self):
        self.t_start = timeit.default_timer()
        return self

    def __exit__(self, _1, _2, _3):
        self.t_end = timeit.default_timer()
        self.dt = self.t_end - self.t_start


def load_trajs(filename, limit_trajs, swap=True):
    # Load expert data
    with h5py.File(filename, 'r') as f:
        # Read data as written by scripts/format_data.py (openai format)
        if swap:
            obs = np.array(f['obs_B_T_Do']).T
            act = np.array(f['a_B_T_Da']).T
            #rew= np.array(f['r_B_T']).T
            lng = np.array(f['len_B']).T
        else:
            obs = np.array(f['obs_B_T_Do'])
            act = np.array(f['a_B_T_Da'])
            #rew= np.array(f['r_B_T'])
            lng = np.array(f['len_B'])

        full_dset_size = obs.shape[0]
        dset_size = min(
            full_dset_size, limit_trajs) if limit_trajs is not None else full_dset_size

        exobs_B_T_Do = obs[:dset_size, ...][...]
        exa_B_T_Da = act[:dset_size, ...][...]
        #exr_B_T = rew[:dset_size,...][...]
        exlen_B = lng[:dset_size, ...][...]

    # compute trajectory intervals from lengths.
    interval = np.ones(full_dset_size,).astype(int)
    for i, l in enumerate(exlen_B):
        if i == 0:
            continue
        interval[i] = interval[i - 1] + l

    stats = {'N': dset_size}
    # record trajectory statistics
    stats['obs_min'] = np.nanmin(exobs_B_T_Do, axis=(0, 1))
    stats['obs_max'] = np.nanmax(exobs_B_T_Do, axis=(0, 1))

    stats['obs_minsq'] = np.nanmin(exobs_B_T_Do ** 2., axis=(0, 1))
    stats['obs_maxsq'] = np.nanmax(exobs_B_T_Do ** 2., axis=(0, 1))

    stats['obs_mean'] = np.nanmean(exobs_B_T_Do, axis=(0, 1))
    stats['obs_meansq'] = np.nanmean(np.square(exobs_B_T_Do), axis=(0, 1))
    stats['obs_std'] = np.nanstd(exobs_B_T_Do, axis=(0, 1))

    stats['act_mean'] = np.nanmean(exa_B_T_Da, axis=(0, 1))
    stats['act_meansq'] = np.nanmean(np.square(exa_B_T_Da), axis=(0, 1))
    stats['act_std'] = np.nanstd(exa_B_T_Da, axis=(0, 1))

    data = {'exobs_B_T_Do': exobs_B_T_Do,
            'exa_B_T_Da': exa_B_T_Da,
            #'exr_B_T' : exr_B_T,
            'exlen_B': exlen_B,
            'interval': interval
            }
    return data, stats


def prepare_trajs(exobs_B_T_Do, exa_B_T_Da, exlen_B, data_subsamp_freq=1, labeller=None):
    print('exlen_B inside: %i' % exlen_B.shape[0])

    start_times_B = np.random.RandomState(0).randint(
        0, data_subsamp_freq, size=exlen_B.shape[0])
    exobs_Bstacked_Do = np.concatenate(
        [exobs_B_T_Do[i, start_times_B[i]:l:data_subsamp_freq, :]
            for i, l in enumerate(exlen_B)],
        axis=0)
    exa_Bstacked_Da = np.concatenate(
        [exa_B_T_Da[i, start_times_B[i]:l:data_subsamp_freq, :]
            for i, l in enumerate(exlen_B)],
        axis=0)

    assert exobs_Bstacked_Do.shape[0] == exa_Bstacked_Da.shape[0]

    data = {'exobs_Bstacked_Do': exobs_Bstacked_Do,
            'exa_Bstacked_Da': exa_Bstacked_Da}

    return data
