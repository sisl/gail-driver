import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers


def py_ortho_init(scale):
    def _init(shape):
        u, s, v = np.linalg.svd(np.random.uniform(size=shape))
        return np.cast['float32'](u * scale)

    return _init


class OrthogonalInitializer(object):
    def __init__(self, scale=1.1):
        self.scale = scale

    def __call__(self, shape, dtype=tf.float32, *args, **kwargs):
        result, = tf.py_func(py_ortho_init(self.scale), [shape], [tf.float32])
        result.set_shape(shape)
        return result


class PolicyNetwork():
    def __init__(self, args):
        # Placeholder for data
        self.inputs = tf.placeholder(
            tf.float32, [args.batch_size, args.state_dim], name="inputs")
        self.targets = tf.placeholder(
            tf.float32, [args.batch_size, args.action_dim], name="targets")
        self.learning_rate = tf.Variable(
            0.0, trainable=False, name="learning_rate")

        # Create the computational graph
        self._create_mlp(args)
        if args.gru_input_dim > 0:
            self._create_gru(args)
        self._create_optimizer(args)

    def _create_mlp(self, args):

        # Create fully connected network of desired size
        W = tf.get_variable("mlp_policy/hidden_0/W",
                            [args.state_dim, args.mlp_size[0]], initializer=initializers.xavier_initializer())
        b = tf.get_variable("mlp_policy/hidden_0/b",
                            [args.mlp_size[0]])
        output = args.mlp_activation(tf.nn.xw_plus_b(self.inputs, W, b))

        # Hidden layers
        for i in xrange(1, len(args.mlp_size)):
            W = tf.get_variable("mlp_policy/hidden_" + str(i) + "/W",
                                [args.mlp_size[i - 1], args.mlp_size[i]], initializer=initializers.xavier_initializer())
            b = tf.get_variable("mlp_policy/hidden_" + str(i) + "/b",
                                [args.mlp_size[i]])
            output = args.mlp_activation(tf.nn.xw_plus_b(output, W, b))

        if args.gru_input_dim > 0:
            # Output/input to GRU
            W = tf.get_variable("mlp_policy/output/W",
                                [args.mlp_size[-1], args.gru_input_dim], initializer=initializers.xavier_initializer())
            b = tf.get_variable("mlp_policy/output/b",
                                [args.gru_input_dim])

            self.gru_input = args.mlp_activation(tf.nn.xw_plus_b(output, W, b))
        else:
            # Map to actions
            W = tf.get_variable("mlp_policy/mean_network/output_flat/W",
                                [args.mlp_size[-1], args.action_dim], initializer=initializers.xavier_initializer())
            b = tf.get_variable("mlp_policy/mean_network/output_flat/b",
                                [args.action_dim])
            self.a_mean = tf.nn.xw_plus_b(output, W, b)

            # Initialize logstd
            self.a_logstd = tf.Variable(np.zeros(
                args.action_dim), name="mlp_policy/output_log_std/param", dtype=tf.float32)

    def _create_gru(self, args):

        # Weights for the initial hidden state
        self.hprev = tf.get_variable("gru_policy/mean_network/gru/h0",
                                     [args.batch_size, args.gru_size], initializer=tf.zeros_initializer, trainable=False)
        # Weights for the reset gate
        W_xr = tf.get_variable("gru_policy/mean_network/gru/W_xr",
                               [args.gru_input_dim, args.gru_size], initializer=initializers.xavier_initializer())
        W_hr = tf.get_variable("gru_policy/mean_network/gru/W_hr",
                               [args.gru_size, args.gru_size], initializer=OrthogonalInitializer())
        b_r = tf.get_variable("gru_policy/mean_network/gru/b_r",
                              [args.gru_size], initializer=tf.zeros_initializer)
        # Weights for the update gate
        W_xu = tf.get_variable("gru_policy/mean_network/gru/W_xu",
                               [args.gru_input_dim, args.gru_size], initializer=initializers.xavier_initializer())
        W_hu = tf.get_variable("gru_policy/mean_network/gru/W_hu",
                               [args.gru_size, args.gru_size], initializer=OrthogonalInitializer())
        b_u = tf.get_variable("gru_policy/mean_network/gru/b_u",
                              [args.gru_size], initializer=tf.zeros_initializer)
        # Weights for the cell gate
        W_xc = tf.get_variable("gru_policy/mean_network/gru/W_xc",
                               [args.gru_input_dim, args.gru_size], initializer=initializers.xavier_initializer())
        W_hc = tf.get_variable("gru_policy/mean_network/gru/W_hc",
                               [args.gru_size, args.gru_size], initializer=OrthogonalInitializer())
        b_c = tf.get_variable("gru_policy/mean_network/gru/b_c",
                              [args.gru_size], initializer=tf.zeros_initializer)

        # Concatenate matrices
        self.W_x_ruc = tf.concat(1, [W_xr, W_xu, W_xc])
        self.W_h_ruc = tf.concat(1, [W_hr, W_hu, W_hc])
        self.b_ruc = tf.concat(0, [b_r, b_u, b_c])

        # Compute output from GRU layer
        xb_ruc = tf.matmul(self.gru_input, self.W_x_ruc) + \
            tf.reshape(self.b_ruc, (1, -1))
        h_ruc = tf.matmul(self.hprev, self.W_h_ruc)
        self.xb_r, self.xb_u, self.xb_c = tf.split(
            split_dim=1, num_split=3, value=xb_ruc)
        self.h_r, self.h_u, self.h_c = tf.split(
            split_dim=1, num_split=3, value=h_ruc)
        self.r = tf.nn.sigmoid(self.xb_r + self.h_r)
        self.u = tf.nn.sigmoid(self.xb_u + self.h_u)
        self.c = tf.nn.tanh(self.xb_c + self.r * self.h_c)
        self.h = (1 - self.u) * self.hprev + self.u * self.c

        # Map to actions
        W = tf.get_variable("gru_policy/mean_network/output_flat/W",
                            [args.gru_size, args.action_dim], initializer=initializers.xavier_initializer())
        b = tf.get_variable("gru_policy/mean_network/output_flat/b",
                            [args.action_dim])
        self.a_mean = tf.nn.xw_plus_b(self.h, W, b)

        # Initialize logstd
        self.a_logstd = tf.Variable(np.zeros(
            args.action_dim), name="gru_policy/output_log_std/param", dtype=tf.float32)

    def _create_optimizer(self, args):
        # Find negagtive log-likelihood of true actions
        std_a = tf.exp(self.a_logstd)
        pl_1 = 0.5 * tf.to_float(args.action_dim) * np.log(2. * np.pi)
        pl_2 = tf.to_float(args.action_dim) * tf.reduce_sum(tf.log(std_a))
        pl_3 = 0.5 * \
            tf.reduce_mean(tf.reduce_sum(
                tf.square((self.targets - self.a_mean) / std_a), 1))
        policy_loss = pl_1 + pl_2 + pl_3

        # Find overall loss
        self.cost = policy_loss
        self.summary_policy = tf.scalar_summary(
            "Policy loss", tf.reduce_mean(policy_loss))

        # Perform parameter update
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, tvars), args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train = optimizer.apply_gradients(zip(grads, tvars))
