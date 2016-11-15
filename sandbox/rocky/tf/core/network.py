import sandbox.rocky.tf.core.layers as L
import tensorflow as tf
import numpy as np
import itertools
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core.parameterized import Parameterized, Model
from sandbox.rocky.tf.core.layers_powered import LayersPowered

from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides

from sandbox.rocky.tf.optimizers.first_order_optimizer import Solver
from sandbox.rocky.tf.optimizers.lbfgs_optimizer import LbfgsOptimizer

class NeuralNetwork(Model):

    def _predict(self, t, X):
        sess = tf.get_default_session()

        N, _ = X.shape
        B = self.input_var.get_shape()[0].value

        if B is None or B == N:
            pred = sess.run(t, {self.input_var: X})
        else:
            pred = [sess.run(t, {self.input_var: X[i:i+B]}) for i in range(0,N,B)]
            pred = np.row_stack(pred)

        return pred

    def likelihood_loss(self):
        if self.output_layer.nonlinearity == tf.nn.softmax:
            logits = self.output_layer.get_logits_for(L.get_output(self.layers[-2]))
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits, tf.squeeze(self.target_var))
                )

        elif self.output_layer.nonlinearity == tf.identity:
            outputs = self.output_layer.get_output_for(L.get_output(self._layers[-2]))
            loss = tf.reduce_mean(
                0.5 * tf.square(outputs - self.target_var), name='like_loss'
            )

        elif self.output_layer.nonlinearity == tf.nn.sigmoid:

            logits = self.output_layer.get_logits_for(L.get_output(self.layers[-2]))
            sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, tf.squeeze(self.target_var))

            if sigmoid_loss.get_shape().ndims == 2:
                loss = tf.reduce_mean(
                    tf.reduce_sum(sigmoid_loss, reduction_indices= 1)
                )
            else:
                loss = tf.reduce_mean(sigmoid_loss)

        return loss

    def complexity_loss(self, reg, cmx):
        """
        Compute penalties for model complexity (e.g., l2 regularization, or kl penalties for vae and bnn).
        """
        # loss coming from weight regularization
        loss = reg * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # loss coming from data-dependent regularization
        for layer in self.layers:
            if layer.penalize_complexity:
                z_mu, z_sig = layer.get_dparams_for(L.get_output(layer.input_layer))
                d_loss = layer.bayesreg.activ_kl(z_mu,z_sig)

                loss += cmx * d_loss

        return reg * loss

    def loss(self, reg= 0.0, cmx= 1.0):
        return tf.add(self.likelihood_loss(), self.complexity_loss(reg, cmx),name= 'loss')

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def target_var(self):
        return self._l_tar.input_var

    @property
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output

    @property
    def n_params(self):
        return sum([np.prod(param.get_shape()).value for param in self.get_params()])


class DeterministicNetwork(NeuralNetwork):

    def predict(self, X):

        if self.output_layer.nonlinearity == tf.nn.softmax:
            y_p = tf.argmax(self._output,1)
        else:
            y_p = self._output

        Y_p = self._predict(y_p, X)
        return Y_p

class StochasticNetwork(NeuralNetwork):

    def predict(self, X, k= 1):
        sess = tf.get_default_session()

        o_p= []
        for _ in range(k):

            o_p.append(self._predict(self._output, X))
            o_p = np.concatenate([o[None,...] for o in o_p], axis= 0)
            mu_p = np.mean(o_p,axis= 0)
            std_p = np.std(o_p,axis= 0)

        if self.output_layer.nonlinearity == tf.nn.softmax:
            Y_p = np.argmax(mu_p,1)
        elif self.output_layer.nonlinearity == tf.identity:
            Y_p = mu_p
        elif self.output_layer.nonlinearity == tf.nn.sigmoid:
            Y_p = mu_p

        return Y_p


class MLP(LayersPowered, Serializable, DeterministicNetwork):
    def __init__(self, name, output_dim, hidden_sizes, hidden_nonlinearity,
                 output_nonlinearity, hidden_W_init=L.XavierUniformInitializer(), hidden_b_init=tf.zeros_initializer,
                 output_W_init=L.XavierUniformInitializer(), output_b_init=tf.zeros_initializer, batch_size=None,
                 input_var=None, input_layer=None, input_shape=None, batch_normalization=False, weight_normalization=False,
                 ):

        Serializable.quick_init(self, locals())
        self.name= name

        with tf.variable_scope(name):
            if input_layer is None:
                l_in = L.InputLayer(shape=(batch_size,) + input_shape, input_var=input_var, name="input")
            else:
                l_in = input_layer
            self._layers = [l_in]
            l_hid = l_in
            if batch_normalization:
                ls = L.batch_norm(l_hid)
                l_hid = ls[-1]
                self._layers += ls
            for idx, hidden_size in enumerate(hidden_sizes):
                l_hid = L.DenseLayer(
                    l_hid,
                    num_units=hidden_size,
                    nonlinearity=hidden_nonlinearity,
                    name="hidden_%d" % idx,
                    W=hidden_W_init,
                    b=hidden_b_init,
                    weight_normalization=weight_normalization
                )
                if batch_normalization:
                    ls = L.batch_norm(l_hid)
                    l_hid = ls[-1]
                    self._layers += ls
                self._layers.append(l_hid)
            l_out = L.DenseLayer(
                l_hid,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output",
                W=output_W_init,
                b=output_b_init,
                weight_normalization=weight_normalization
            )
            if batch_normalization:
                ls = L.batch_norm(l_out)
                l_out = ls[-1]
                self._layers += ls
            self._layers.append(l_out)
            self._l_in = l_in
            self._l_out = l_out
            self._l_tar = L.InputLayer(shape=(batch_size,) + (output_dim,), input_var=input_var, name="target")

            # self._input_var = l_in.input_var
            self._output = L.get_output(l_out)

            LayersPowered.__init__(self, l_out)


class BayesMLP(LayersPowered, Serializable, StochasticNetwork):
    def __init__(self, name, output_dim, hidden_sizes, hidden_nonlinearity,
                 output_nonlinearity, hidden_W_init=L.XavierUniformInitializer(), hidden_b_init=tf.zeros_initializer,
                 output_W_init=L.XavierUniformInitializer(), output_b_init=tf.zeros_initializer, batch_size=None,
                 input_var=None, input_layer=None, input_shape=None, batch_normalization=False, weight_normalization=False,
                 reg_params= {},
                 ):

        Serializable.quick_init(self, locals())
        self.name= name

        with tf.variable_scope(name):
            if input_layer is None:
                l_in = L.InputLayer(shape=(batch_size,) + input_shape, input_var=input_var, name="input")

            else:
                l_in = input_layer
            self._layers = [l_in]
            l_hid = l_in
            if batch_normalization:
                l_hid = L.batch_norm(l_hid)
            for idx, hidden_size in enumerate(hidden_sizes):
                l_hid = L.BayesLayer(
                    l_hid,
                    num_units=hidden_size,
                    nonlinearity=hidden_nonlinearity,
                    name="hidden_%d" % idx,
                    W=hidden_W_init,
                    b=hidden_b_init,
                    weight_normalization=weight_normalization,
                    reg_params= reg_params
                )
                if batch_normalization:
                    l_hid = L.batch_norm(l_hid)
                self._layers.append(l_hid)
            l_out = L.BayesLayer(
                l_hid,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output",
                W=output_W_init,
                b=output_b_init,
                weight_normalization=weight_normalization,
                reg_params= reg_params
            )
            if batch_normalization:
                l_out = L.batch_norm(l_out)
            self._layers.append(l_out)
            self._l_in = l_in
            self._l_out = l_out
            self._l_tar = L.InputLayer(shape=(batch_size,) + (output_dim,), input_var=input_var, name="target")

            # self._input_var = l_in.input_var
            self._output = L.get_output(l_out)

            LayersPowered.__init__(self, l_out)


class LatentMLP(LayersPowered, Serializable, StochasticNetwork):
    def __init__(self, name, output_dim, hidden_spec, hidden_nonlinearity,
                 output_nonlinearity, hidden_W_init=L.XavierUniformInitializer(), hidden_b_init=tf.zeros_initializer,
                 output_W_init=L.XavierUniformInitializer(), output_b_init=tf.zeros_initializer, batch_size=None,
                 input_var=None, input_layer=None, input_shape=None, batch_normalization=False, weight_normalization=False,
                 reg_params= {},
                 ):

        Serializable.quick_init(self, locals())
        self.name= name

        with tf.variable_scope(name):
            if input_layer is None:
                l_in = L.InputLayer(shape=(batch_size,) + input_shape, input_var=input_var, name="input")
            else:
                l_in = input_layer
            self._layers = [l_in]
            l_hid = l_in
            if batch_normalization:
                l_hid = L.batch_norm(l_hid)
            for idx, (hidden_type, hidden_size) in enumerate(hidden_spec):
                if hidden_type == 'dense':
                    l_hid = L.DenseLayer(
                        l_hid,
                        num_units=hidden_size,
                        nonlinearity=hidden_nonlinearity,
                        name="hidden_%d" % idx,
                        W=hidden_W_init,
                        b=hidden_b_init,
                        weight_normalization=weight_normalization,
                        reg_params= reg_params
                    )

                elif hidden_type == 'latent':
                    l_hid = L.LatentLayer(
                        l_hid,
                        num_units=hidden_size,
                        name="hidden_%d" % idx,
                        W=hidden_W_init,
                        b=hidden_b_init,
                        weight_normalization=weight_normalization,
                        reg_params= reg_params
                    )

                else:
                    raise NotImplementedError

                if batch_normalization:
                    l_hid = L.batch_norm(l_hid)
                self._layers.append(l_hid)
            l_out = L.DenseLayer(
                l_hid,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output",
                W=output_W_init,
                b=output_b_init,
                weight_normalization=weight_normalization
            )
            if batch_normalization:
                l_out = L.batch_norm(l_out)
            self._layers.append(l_out)
            self._l_in = l_in
            self._l_out = l_out
            self._l_tar = L.InputLayer(shape=(batch_size,) + (output_dim,), input_var=input_var, name="target")

            # self._input_var = l_in.input_var
            self._output = L.get_output(l_out)

            LayersPowered.__init__(self, l_out)


    def generate(self, Z= None, prior_ix= -1):
        # retrieve all layers receiving inputs from a LatentLayer
        encoders = [layer for layer in self.layers if type(layer) is L.LatentLayer]
        sampling_layer = encoders[prior_ix]
        if Z is None:
            Z = tf.random_normal(sampling_layer.output_shape)
        else:
            Z = tf.constant(Z, dtype=tf.float32)

        sess = tf.get_default_session()
        Y = sess.run(L.get_output(self.output_layer, inputs= {sampling_layer : Z}))

        return Y

class RewardMLP(MLP):
    """
    overrides MLP with methods / properties used in generative adversarial learning.
    """

    def compute_reward(self, X):
        predits = -tf.log(1.0 - self.output)
        #predits = -tf.log(1.0 - tf.sigmoid(self.output))
        Y_p = self._predict(predits, X)
        return Y_p

    def compute_score(self, X):
        """
        predict logits ...
        """
        logits = self.output_layer.get_logits_for(L.get_output(self.layers[-2]))
        #logits = self.output
        Y_p = self._predict(logits, X)
        return Y_p

    def likelihood_loss(self):
        logits = self.output_layer.get_logits_for(L.get_output(self.layers[-2]))
        #logits = L.get_output(self.layers[-1])
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, self.target_var)
        #ent_B = tfutil.logit_bernoulli_entropy(logits)
        #self.obj = tf.reduce_sum(loss_B - self.ent_reg_weight * ent_B)
        return tf.reduce_sum(loss)

    def complexity_loss(self, reg, cmx):
        return tf.constant(0.0)

    def loss(self, reg= 0.0, cmx= 0.0):
        #logits = self.output_layer.get_logits_for(L.get_output(self.layers[-2]))
        #loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, self.target_var)
        #ent_B = tfutil.logit_bernoulli_entropy(logits)
        #self.obj = tf.reduce_sum(loss_B - self.ent_reg_weight * ent_B)
        #return tf.reduce_sum(loss)
        loss = self.likelihood_loss()
        return loss


class BaselineMLP(MLP, Baseline):
    def initialize_optimizer(self):
        self._optimizer = LbfgsOptimizer('optim')

        optimizer_args = dict(
            loss=self.loss(),
            target=self,
            inputs = [self.input_var, self.target_var],
            network_outputs=[self.output]
        )

        self._optimizer.update_opt(**optimizer_args)

    @overrides
    def predict(self, path):
        # X = np.column_stack((path['observations'], path['actions']))
        X = path['observations']
        return super(BaselineMLP, self).predict(X)

    @overrides
    def fit(self, paths):
        observations = np.concatenate([p["observations"] for p in paths])
        returns = np.concatenate([p["returns"] for p in paths])
        #self._regressor.fit(observations, returns.reshape((-1, 1)))
        self._optimizer.optimize([observations, returns[...,None]])


class ConvNetwork(LayersPowered, Serializable):
    def __init__(self, name, input_shape, output_dim,
                 conv_filters, conv_filter_sizes, conv_strides, conv_pads,
                 hidden_sizes, hidden_nonlinearity, output_nonlinearity,
                 hidden_W_init=L.XavierUniformInitializer(), hidden_b_init=tf.zeros_initializer,
                 output_W_init=L.XavierUniformInitializer(), output_b_init=tf.zeros_initializer,
                 input_var=None, input_layer=None, batch_normalization=False, weight_normalization=False):
        Serializable.quick_init(self, locals())
        """
        A network composed of several convolution layers followed by some fc layers.
        input_shape: (width,height,channel)
            HOWEVER, network inputs are assumed flattened. This network will first unflatten the inputs and then apply the standard convolutions and so on.
        conv_filters: a list of numbers of convolution kernel
        conv_filter_sizes: a list of sizes (int) of the convolution kernels
        conv_strides: a list of strides (int) of the conv kernels
        conv_pads: a list of pad formats (either 'SAME' or 'VALID')
        hidden_nonlinearity: a nonlinearity from tf.nn, shared by all conv and fc layers
        hidden_sizes: a list of numbers of hidden units for all fc layers
        """
        with tf.variable_scope(name):
            if input_layer is not None:
                l_in = input_layer
                l_hid = l_in
            elif len(input_shape) == 3:
                l_in = L.InputLayer(shape=(None, np.prod(input_shape)), input_var=input_var, name="input")
                l_hid = L.reshape(l_in, ([0],) + input_shape, name="reshape_input")
            elif len(input_shape) == 2:
                l_in = L.InputLayer(shape=(None, np.prod(input_shape)), input_var=input_var, name="input")
                input_shape = (1,) + input_shape
                l_hid = L.reshape(l_in, ([0],) + input_shape, name="reshape_input")
            else:
                l_in = L.InputLayer(shape=(None,) + input_shape, input_var=input_var, name="input")
                l_hid = l_in

            if batch_normalization:
                l_hid = L.batch_norm(l_hid)
            for idx, conv_filter, filter_size, stride, pad in zip(
                    range(len(conv_filters)),
                    conv_filters,
                    conv_filter_sizes,
                    conv_strides,
                    conv_pads,
            ):
                l_hid = L.Conv2DLayer(
                    l_hid,
                    num_filters=conv_filter,
                    filter_size=filter_size,
                    stride=(stride, stride),
                    pad=pad,
                    nonlinearity=hidden_nonlinearity,
                    name="conv_hidden_%d" % idx,
                    weight_normalization=weight_normalization,
                )
                if batch_normalization:
                    l_hid = L.batch_norm(l_hid)

            if output_nonlinearity == L.spatial_expected_softmax:
                assert len(hidden_sizes) == 0
                assert output_dim == conv_filters[-1] * 2
                l_hid.nonlinearity = tf.identity
                l_out = L.SpatialExpectedSoftmaxLayer(l_hid)
            else:
                l_hid = L.flatten(l_hid, name="conv_flatten")
                for idx, hidden_size in enumerate(hidden_sizes):
                    l_hid = L.DenseLayer(
                        l_hid,
                        num_units=hidden_size,
                        nonlinearity=hidden_nonlinearity,
                        name="hidden_%d" % idx,
                        W=hidden_W_init,
                        b=hidden_b_init,
                        weight_normalization=weight_normalization,
                    )
                    if batch_normalization:
                        l_hid = L.batch_norm(l_hid)
                l_out = L.DenseLayer(
                    l_hid,
                    num_units=output_dim,
                    nonlinearity=output_nonlinearity,
                    name="output",
                    W=output_W_init,
                    b=output_b_init,
                    weight_normalization=weight_normalization,
                )
                if batch_normalization:
                    l_out = L.batch_norm(l_out)
            self._l_in = l_in
            self._l_out = l_out
            # self._input_var = l_in.input_var

        LayersPowered.__init__(self, l_out)

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._l_in.input_var


class GRUNetwork(object):
    def __init__(self, name, input_shape, output_dim, hidden_dim, hidden_nonlinearity=tf.nn.relu,
                 gru_layer_cls=L.GRULayer,
                 output_nonlinearity=None, input_var=None, input_layer=None, layer_args=None):
        with tf.variable_scope(name):
            if input_layer is None:
                l_in = L.InputLayer(shape=(None, None) + input_shape, input_var=input_var, name="input")
            else:
                l_in = input_layer
            l_step_input = L.InputLayer(shape=(None,) + input_shape, name="step_input")
            l_step_prev_state = L.InputLayer(shape=(None, hidden_dim), name="step_prev_state")
            if layer_args is None:
                layer_args = dict()
            l_gru = gru_layer_cls(l_in, num_units=hidden_dim, hidden_nonlinearity=hidden_nonlinearity,
                                  hidden_init_trainable=False, name="gru", **layer_args)
            l_gru_flat = L.ReshapeLayer(
                l_gru, shape=(-1, hidden_dim),
                name="gru_flat"
            )
            l_output_flat = L.DenseLayer(
                l_gru_flat,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output_flat"
            )
            l_output = L.OpLayer(
                l_output_flat,
                op=lambda flat_output, l_input:
                tf.reshape(flat_output, tf.pack((tf.shape(l_input)[0], tf.shape(l_input)[1], -1))),
                shape_op=lambda flat_output_shape, l_input_shape:
                (l_input_shape[0], l_input_shape[1], flat_output_shape[-1]),
                extras=[l_in],
                name="output"
            )
            l_step_state = l_gru.get_step_layer(l_step_input, l_step_prev_state, name="step_state")
            l_step_hidden = l_step_state
            l_step_output = L.DenseLayer(
                l_step_hidden,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                W=l_output_flat.W,
                b=l_output_flat.b,
                name="step_output"
            )

            self._l_in = l_in
            self._hid_init_param = l_gru.h0
            self._l_gru = l_gru
            self._l_out = l_output
            self._l_step_input = l_step_input
            self._l_step_prev_state = l_step_prev_state
            self._l_step_hidden = l_step_hidden
            self._l_step_state = l_step_state
            self._l_step_output = l_step_output
            self._hidden_dim = hidden_dim

    @property
    def state_dim(self):
        return self._hidden_dim

    @property
    def hidden_dim(self):
        return self._hidden_dim

    @property
    def input_layer(self):
        return self._l_in

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def output_layer(self):
        return self._l_out

    @property
    def recurrent_layer(self):
        return self._l_gru

    @property
    def step_input_layer(self):
        return self._l_step_input

    @property
    def step_prev_state_layer(self):
        return self._l_step_prev_state

    @property
    def step_hidden_layer(self):
        return self._l_step_hidden

    @property
    def step_state_layer(self):
        return self._l_step_state

    @property
    def step_output_layer(self):
        return self._l_step_output

    @property
    def hid_init_param(self):
        return self._hid_init_param

    @property
    def state_init_param(self):
        return self._hid_init_param


class LSTMNetwork(object):
    def __init__(self, name, input_shape, output_dim, hidden_dim, hidden_nonlinearity=tf.nn.relu,
                 lstm_layer_cls=L.LSTMLayer,
                 output_nonlinearity=None, input_var=None, input_layer=None, forget_bias=1.0, use_peepholes=False,
                 layer_args=None):
        with tf.variable_scope(name):
            if input_layer is None:
                l_in = L.InputLayer(shape=(None, None) + input_shape, input_var=input_var, name="input")
            else:
                l_in = input_layer
            l_step_input = L.InputLayer(shape=(None,) + input_shape, name="step_input")
            # contains previous hidden and cell state
            l_step_prev_state = L.InputLayer(shape=(None, hidden_dim * 2), name="step_prev_state")
            if layer_args is None:
                layer_args = dict()
            l_lstm = lstm_layer_cls(l_in, num_units=hidden_dim, hidden_nonlinearity=hidden_nonlinearity,
                                    hidden_init_trainable=False, name="lstm", forget_bias=forget_bias,
                                    cell_init_trainable=False, use_peepholes=use_peepholes, **layer_args)
            l_lstm_flat = L.ReshapeLayer(
                l_lstm, shape=(-1, hidden_dim),
                name="lstm_flat"
            )
            l_output_flat = L.DenseLayer(
                l_lstm_flat,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output_flat"
            )
            l_output = L.OpLayer(
                l_output_flat,
                op=lambda flat_output, l_input:
                tf.reshape(flat_output, tf.pack((tf.shape(l_input)[0], tf.shape(l_input)[1], -1))),
                shape_op=lambda flat_output_shape, l_input_shape:
                (l_input_shape[0], l_input_shape[1], flat_output_shape[-1]),
                extras=[l_in],
                name="output"
            )
            l_step_state = l_lstm.get_step_layer(l_step_input, l_step_prev_state, name="step_state")
            l_step_hidden = L.SliceLayer(l_step_state, indices=slice(hidden_dim), name="step_hidden")
            l_step_cell = L.SliceLayer(l_step_state, indices=slice(hidden_dim, None), name="step_cell")
            l_step_output = L.DenseLayer(
                l_step_hidden,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                W=l_output_flat.W,
                b=l_output_flat.b,
                name="step_output"
            )

            self._l_in = l_in
            self._hid_init_param = l_lstm.h0
            self._cell_init_param = l_lstm.c0
            self._l_lstm = l_lstm
            self._l_out = l_output
            self._l_step_input = l_step_input
            self._l_step_prev_state = l_step_prev_state
            self._l_step_hidden = l_step_hidden
            self._l_step_cell = l_step_cell
            self._l_step_state = l_step_state
            self._l_step_output = l_step_output
            self._hidden_dim = hidden_dim

    @property
    def state_dim(self):
        return self._hidden_dim * 2

    @property
    def input_layer(self):
        return self._l_in

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def output_layer(self):
        return self._l_out

    @property
    def recurrent_layer(self):
        return self._l_lstm

    @property
    def step_input_layer(self):
        return self._l_step_input

    @property
    def step_prev_state_layer(self):
        return self._l_step_prev_state

    @property
    def step_hidden_layer(self):
        return self._l_step_hidden

    @property
    def step_state_layer(self):
        return self._l_step_state

    @property
    def step_cell_layer(self):
        return self._l_step_cell

    @property
    def step_output_layer(self):
        return self._l_step_output

    @property
    def hid_init_param(self):
        return self._hid_init_param

    @property
    def cell_init_param(self):
        return self._cell_init_param

    @property
    def state_init_param(self):
        return tf.concat(0, [self._hid_init_param, self._cell_init_param])


class ConvMergeNetwork(LayersPowered, Serializable):
    """
    This network allows the input to consist of a convolution-friendly component, plus a non-convolution-friendly
    component. These two components will be concatenated in the fully connected layers. There can also be a list of
    optional layers for the non-convolution-friendly component alone.


    The input to the network should be a matrix where each row is a single input entry, with both the aforementioned
    components flattened out and then concatenated together
    """

    def __init__(self, name, input_shape, extra_input_shape, output_dim, hidden_sizes,
                 conv_filters, conv_filter_sizes, conv_strides, conv_pads,
                 extra_hidden_sizes=None,
                 hidden_W_init=L.XavierUniformInitializer(), hidden_b_init=tf.zeros_initializer,
                 output_W_init=L.XavierUniformInitializer(), output_b_init=tf.zeros_initializer,
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=None, n=2,
                 input_var=None, input_layer=None):
        Serializable.quick_init(self, locals())

        if extra_hidden_sizes is None:
            extra_hidden_sizes = []

        with tf.variable_scope(name):

            input_flat_dim = np.prod(input_shape)
            extra_input_flat_dim = np.prod(extra_input_shape)
            total_input_flat_dim = input_flat_dim + extra_input_flat_dim

            if input_layer is None:
                l_in = L.InputLayer(shape=(None, total_input_flat_dim), input_var=input_var, name="input")
            else:
                l_in = input_layer

            l_conv_in = L.reshape(
                L.SliceLayer(
                    l_in,
                    indices=slice(input_flat_dim),
                    name="conv_slice"
                ),
                ([0],) + input_shape,
                name="conv_reshaped"
            )
            l_extra_in = L.reshape(
                L.SliceLayer(
                    l_in,
                    indices=slice(input_flat_dim, None),
                    name="extra_slice"
                ),
                ([0],) + extra_input_shape,
                name="extra_reshaped"
            )

            l_conv_hid = l_conv_in
            for idx, conv_filter, filter_size, stride, pad in zip(
                    range(len(conv_filters)),
                    conv_filters,
                    conv_filter_sizes,
                    conv_strides,
                    conv_pads,
            ):
                l_conv_hid = L.Conv2DLayer(
                    l_conv_hid,
                    num_filters=conv_filter,
                    filter_size=filter_size,
                    stride=(stride, stride),
                    pad=pad, n=n,
                    nonlinearity=hidden_nonlinearity,
                    name="conv_hidden_%d" % idx,
                )

            l_extra_hid = l_extra_in
            for idx, hidden_size in enumerate(extra_hidden_sizes):
                l_extra_hid = L.DenseLayer(
                    l_extra_hid,
                    num_units=hidden_size,
                    nonlinearity=hidden_nonlinearity,
                    name="extra_hidden_%d" % idx,
                    W=hidden_W_init,
                    b=hidden_b_init,
                )

            l_joint_hid = L.concat(
                [L.flatten(l_conv_hid, name="conv_hidden_flat"), l_extra_hid],
                name="joint_hidden"
            )

            for idx, hidden_size in enumerate(hidden_sizes):
                l_joint_hid = L.DenseLayer(
                    l_joint_hid,
                    num_units=hidden_size,
                    nonlinearity=hidden_nonlinearity,
                    name="joint_hidden_%d" % idx,
                    W=hidden_W_init,
                    b=hidden_b_init,
                )
            l_out = L.DenseLayer(
                l_joint_hid,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output",
                W=output_W_init,
                b=output_b_init,
            )
            self._l_in = l_in
            self._l_out = l_out

            LayersPowered.__init__(self, [l_out], input_layers=[l_in])

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._l_in.input_var


class RewardCMN(ConvMergeNetwork):
    """
    overrides ConvMergeNetwork with methods / properties used in generative adversarial learning.
    """

    def compute_reward(self, X):
        predits = -tf.log(1.0 - self.output)
        #predits = -tf.log(1.0 - tf.sigmoid(self.output))
        Y_p = self._predict(predits, X)
        return Y_p

    def compute_score(self, X):
        """
        predict logits ...
        """
        logits = self.output_layer.get_logits_for(L.get_output(self.layers[-2]))
        #logits = self.output
        Y_p = self._predict(logits, X)
        return Y_p

    def likelihood_loss(self):
        logits = self.output_layer.get_logits_for(L.get_output(self.layers[-2]))
        #logits = L.get_output(self.layers[-1])
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, self.target_var)
        #ent_B = tfutil.logit_bernoulli_entropy(logits)
        #self.obj = tf.reduce_sum(loss_B - self.ent_reg_weight * ent_B)
        return tf.reduce_sum(loss)

    def complexity_loss(self, reg, cmx):
        return tf.constant(0.0)

    def loss(self, reg= 0.0, cmx= 0.0):
        #logits = self.output_layer.get_logits_for(L.get_output(self.layers[-2]))
        #loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, self.target_var)
        #ent_B = tfutil.logit_bernoulli_entropy(logits)
        #self.obj = tf.reduce_sum(loss_B - self.ent_reg_weight * ent_B)
        #return tf.reduce_sum(loss)
        loss = self.likelihood_loss()
        return loss

