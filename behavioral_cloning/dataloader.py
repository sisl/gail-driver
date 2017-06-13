import collections
import cPickle
import h5py
import math
import numpy as np
import os
import random

# Class to load and preprocess data


class DataLoader():
    def __init__(self, batch_size, val_frac, seq_length, extract_temporal):
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.seq_length = seq_length

        print 'validation fraction: ', self.val_frac

        print "loading data..."
        self._load_data(extract_temporal)

        print 'creating splits...'
        self._create_split()

        print 'shifting/scaling data...'
        self._shift_scale(extract_temporal)

    def _trim_data(self, full_s, full_a, intervals):
        # Python indexing; find bounds on data given seq_length
        intervals -= 1
        lengths = np.floor(np.diff(np.append(intervals, len(
            full_s) - 1)) / self.seq_length) * self.seq_length
        intervals = np.vstack((intervals, intervals + lengths)).T.astype(int)
        ret_bounds = np.insert(np.cumsum(lengths), 0, 0.).astype(int)

        # Remove states that don't fit due to value of seq_length
        s = np.zeros((int(sum(lengths)), full_s.shape[1]))
        for i in xrange(len(ret_bounds) - 1):
            s[ret_bounds[i]:ret_bounds[i + 1]
              ] = full_s[intervals[i, 0]:intervals[i, 1]]
        s = np.reshape(s, (-1, self.seq_length, full_s.shape[1]))

        # Remove actions that don't fit due to value of seq_length
        a = np.zeros((int(sum(lengths)), full_a.shape[1]))
        for i in xrange(len(ret_bounds) - 1):
            a[ret_bounds[i]:ret_bounds[i + 1]
              ] = full_a[intervals[i, 0]:intervals[i, 1]]
        a = np.reshape(a, (-1, self.seq_length, full_a.shape[1]))

        return s, a

    def _load_data(self, extract_temporal):
        data_dir = '../julia/2d_drive_data/'
        filenames = ['data_trajdata_i101_trajectories-0750am-0805am', 'data_trajdata_i101_trajectories-0805am-0820am',
                     'data_trajdata_i101_trajectories-0820am-0835am', 'data_trajdata_i80_trajectories-0400-0415',
                     'data_trajdata_i80_trajectories-0500-0515', 'data_trajdata_i80_trajectories-0515-0530']
        data_suff = '_clb20_rlb20_rll2_clmr100_rlmr50.jld'

        filename = data_dir + filenames[0] + data_suff
        data = h5py.File(filename, 'r')
        s = data['features'][:]
        if extract_temporal:
            s = s[:, range(17) + range(45, 85)]
        else:
            s = s[:, range(8) + range(14, 17) + range(45, 85)]
        a = data['targets'][:]
        intervals = data['intervals'][:]
        data.close()

        cat_s, cat_a = self._trim_data(s, a, intervals)

        for i in xrange(1, len(filenames)):
            filename = data_dir + filenames[i] + data_suff
            data = h5py.File(filename, 'r')
            s = data['features'][:]
            if extract_temporal:
                s = s[:, range(17) + range(45, 85)]
            else:
                s = s[:, range(8) + range(14, 17) + range(45, 85)]
            a = data['targets'][:]
            intervals = data['intervals'][:]
            data.close()

            ret_s, ret_a = self._trim_data(s, a, intervals)

            cat_s = np.concatenate((ret_s, cat_s), axis=0)
            cat_a = np.concatenate((ret_a, cat_a), axis=0)

        # Make sure batch_size divides into num of examples
        self.s = cat_s[:int(
            np.floor(len(cat_s) / self.batch_size) * self.batch_size)]
        self.s = np.reshape(self.s, (-1, self.batch_size,
                                     self.seq_length, cat_s.shape[2]))
        self.a = cat_a[:int(
            np.floor(len(cat_a) / self.batch_size) * self.batch_size)]
        self.a = np.reshape(self.a, (-1, self.batch_size,
                                     self.seq_length, cat_a.shape[2]))

        # Print tensor shapes
        print 'states: ', self.s.shape
        print 'actions: ', self.a.shape

        # Create batch_dict
        self.batch_dict = {}
        self.batch_dict["states"] = np.zeros(
            (self.batch_size, self.seq_length, cat_s.shape[2]))
        self.batch_dict["actions"] = np.zeros(
            (self.batch_size, self.seq_length, cat_a.shape[2]))

        # Shuffle data
        print 'shuffling...'
        p = np.random.permutation(len(self.s))
        self.s = self.s[p]
        self.a = self.a[p]

    # Separate data into train/validation sets
    def _create_split(self):

        # compute number of batches
        self.n_batches = len(self.s)
        self.n_batches_val = int(math.floor(self.val_frac * self.n_batches))
        self.n_batches_train = self.n_batches - self.n_batches_val

        print 'num training batches: ', self.n_batches_train
        print 'num validation batches: ', self.n_batches_val

        self.reset_batchptr_train()
        self.reset_batchptr_val()

    # Shift and scale data to be zero-mean, unit variance
    def _shift_scale(self, extract_temporal):
        data = h5py.File("../trpo_vehicle/validation/models/gail_gru.h5", "r")
        self.shift = data["initial_obs_mean"][:]
        self.scale = data["initial_obs_std"][:]

        # Transform data
        self.s = (self.s - self.shift) / self.scale

    # Sample a new batch of data
    def next_batch_train(self):
        # Extract next batch
        batch_index = self.batch_permuation_train[self.batchptr_train]
        self.batch_dict["states"] = self.s[batch_index]
        self.batch_dict["actions"] = self.a[batch_index]

        # Update pointer
        self.batchptr_train += 1
        return self.batch_dict

    # Return to first batch in train set
    def reset_batchptr_train(self):
        self.batch_permuation_train = np.random.permutation(
            self.n_batches_train)
        self.batchptr_train = 0

    # Return next batch of data in validation set
    def next_batch_val(self):
        # Extract next validation batch
        batch_index = self.batchptr_val + self.n_batches_train - 1
        self.batch_dict["states"] = self.s[batch_index]
        self.batch_dict["actions"] = self.a[batch_index]

        # Update pointer
        self.batchptr_val += 1
        return self.batch_dict

    # Return to first batch in validation set
    def reset_batchptr_val(self):
        self.batchptr_val = 0
