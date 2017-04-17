import argparse
import cPickle
import h5py
import json
import numpy as np
import os
from policy_network import PolicyNetwork
from dataloader import DataLoader
import tensorflow as tf
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',           type=str,
                        default='./models', help='directory to store checkpointed models')
    parser.add_argument('--val_frac',           type=float, default=0.1,
                        help='fraction of data to be witheld in validation set')
    parser.add_argument('--ckpt_name',          type=str,  default='',
                        help='name of checkpoint file to load (blank means none)')
    parser.add_argument('--data_file',          type=str,
                        default='0750am-0805am', help='time slot for data file')
    parser.add_argument('--data_dir',           type=str,
                        default='./2d_drive_data/', help='directory containing data')

    parser.add_argument('--batch_size',         type=int,
                        default=64,        help='minibatch size')
    parser.add_argument('--seq_length',         type=int,
                        default=100,       help='training sequence length')
    parser.add_argument('--state_dim',          type=int,
                        default=51,       help='number of state variables')
    parser.add_argument('--extract_temporal',   type=bool,
                        default=False,    help='Whether to extract temporal features')
    parser.add_argument('--action_dim',         type=int,
                        default=2,        help='number of action variables')

    parser.add_argument('--num_epochs',         type=int,
                        default=50,        help='number of epochs')
    parser.add_argument('--learning_rate',      type=float,
                        default=0.004,     help='learning rate')
    parser.add_argument('--decay_rate',         type=float,
                        default=0.5,       help='decay rate for learning rate')
    parser.add_argument('--grad_clip',          type=float,
                        default=5.0,       help='clip gradients at this value')
    parser.add_argument('--save_h5',            type=bool,  default=False,
                        help='Whether to save network params to h5 file')
    parser.add_argument('--h5_name',            type=str,
                        default='',         help='Name for saved h5 file')

    ############################
    #       Policy Network     #
    ############################
    parser.add_argument('--mlp_activation',  default=tf.nn.elu,
                        help='activation function in policy network')
    parser.add_argument('--mlp_size',           type=int, nargs='+',
                        default=[],        help='number of neurons in each feedforward layer')
    parser.add_argument('--gru_input_dim',      type=int,
                        default=64,        help='size of input to gru layer')
    parser.add_argument('--gru_size',           type=int,
                        default=64,        help='size of gru layer')
    parser.add_argument('--policy_varscope',    type=str,
                        default='policy',   help='variable scope for policy network')

    args = parser.parse_args()

    # Construct model
    net = PolicyNetwork(args)

    # store config to disk
    with open(os.path.join(args.save_dir, 'config.pkl'), 'w') as f:
        cPickle.dump(args, f)

    # Export model parameters or perform training
    if args.save_h5:
        save_h5(args, net)
    else:
        train(args, net)


def safezip(*ls):
    assert all(len(l) == len(ls[0]) for l in ls)
    return zip(*ls)


def save_h5(args, net):
    # Begin tf session
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())

        # load from previous save
        if len(args.ckpt_name) > 0:
            saver.restore(sess, os.path.join(args.save_dir, args.ckpt_name))
        else:
            print 'checkpoint name not specified... exiting.'
            return

        vs = tf.get_collection(tf.GraphKeys.VARIABLES)
        vals = sess.run(vs)
        exclude = ['learning_rate', 'beta', 'Adam']

        with h5py.File(args.h5_name, 'a') as f:
            dset = f.create_group('iter00001')
            for v, val in safezip(vs, vals):
                if all([e not in v.name for e in exclude]):
                    dset[v.name] = val

# Train network


def train(args, net):
    data_loader = DataLoader(
        args.batch_size, args.val_frac, args.seq_length, args.extract_temporal)

    # Begin tf session
    with tf.Session() as sess:
        # Function to evaluate loss on validation set
        def val_loss():
            data_loader.reset_batchptr_val()
            loss = 0.
            for b in xrange(data_loader.n_batches_val):

                # Get batch of inputs/targets
                batch_dict = data_loader.next_batch_val()
                s = batch_dict["states"]
                a = batch_dict["actions"]
                if args.gru_input_dim > 0:
                    hprev = net.hprev.eval()

                # Now loop over all timesteps, finding loss
                for t in xrange(args.seq_length):
                    s_t, a_t = s[:, t], a[:, t]

                    # Construct inputs to network
                    feed_in = {}
                    feed_in[net.inputs] = s_t
                    feed_in[net.targets] = a_t
                    if args.gru_input_dim > 0:
                        feed_in[net.hprev] = hprev
                        feed_out = [net.cost, net.h]
                        cost, hprev = sess.run(feed_out, feed_in)
                    else:
                        feed_out = net.cost
                        cost = sess.run(feed_out, feed_in)

                    loss += cost
            return loss / data_loader.n_batches_val / args.seq_length

        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=15)

        # load from previous save
        if len(args.ckpt_name) > 0:
            saver.restore(sess, os.path.join(args.save_dir, args.ckpt_name))

        # Initialize variable to track validation score over time
        old_score = 1e6
        count_decay = 0
        decay_epochs = []

        # Initialize loss
        loss = 0.0

        # Set initial learning rate
        print 'setting learning rate to ', args.learning_rate
        sess.run(tf.assign(net.learning_rate, args.learning_rate))

        # Set up tensorboard summary
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter('summaries/')

        # Loop over epochs
        for e in xrange(args.num_epochs):

            # Evaluate loss on validation set
            score = val_loss()
            print('Validation Loss: {0:f}'.format(score))

            # Set learning rate
            if (old_score - score) < 0.01:
                count_decay += 1
                decay_epochs.append(e)
                if len(decay_epochs) >= 3 and np.sum(np.diff(decay_epochs)[-2:]) == 2:
                    break
                print 'setting learning rate to ', args.learning_rate * (args.decay_rate ** count_decay)
                sess.run(tf.assign(net.learning_rate,
                                   args.learning_rate * (args.decay_rate ** count_decay)))
            old_score = score

            data_loader.reset_batchptr_train()

            # Loop over batches
            for b in xrange(data_loader.n_batches_train):
                start = time.time()

                # Get batch of inputs/targets
                batch_dict = data_loader.next_batch_train()
                s = batch_dict["states"]
                a = batch_dict["actions"]
                if args.gru_input_dim > 0:
                    hprev = net.hprev.eval()

                # Now loop over all timesteps, finding loss
                for t in xrange(args.seq_length):
                    s_t, a_t = s[:, t], a[:, t]

                    # Construct inputs to network
                    feed_in = {}
                    feed_in[net.inputs] = s_t
                    feed_in[net.targets] = a_t
                    if args.gru_input_dim > 0:
                        feed_in[net.hprev] = hprev
                        feed_out = [net.cost, net.h,
                                    net.summary_policy, net.train]
                        train_loss, hprev, summary_policy, _ = sess.run(
                            feed_out, feed_in)
                    else:
                        feed_out = [net.cost, net.summary_policy, net.train]
                        train_loss, summary_policy, _ = sess.run(
                            feed_out, feed_in)

                    writer.add_summary(summary_policy, e *
                                       data_loader.n_batches_train + b)
                    loss += train_loss

                    end = time.time()

                # Print loss
                if (e * data_loader.n_batches_train + b) % 10 == 0 and b > 0:
                    print "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                        .format(e * data_loader.n_batches_train + b,
                                args.num_epochs * data_loader.n_batches_train,
                                e, loss / 10. / args.seq_length, end - start)
                    loss = 0.0

            # Save model every epoch
            checkpoint_path = os.path.join(args.save_dir, 'bc_policy.ckpt')
            saver.save(sess, checkpoint_path, global_step=e)
            print "model saved to {}".format(checkpoint_path)


if __name__ == '__main__':
    main()
