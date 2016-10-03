


from rllab.misc import ext
from rllab.misc import logger
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.misc import tensor_utils
# from rllab.algo.first_order_method import parse_update_method
from rllab.optimizers.minibatch_dataset import BatchDataset
from collections import OrderedDict
import tensorflow as tf
import time
from functools import partial
import pyprind

import numpy as np


class FirstOrderOptimizer(Serializable):
    """
    Performs (stochastic) gradient descent, possibly using fancier methods like adam etc.
    """

    def __init__(
            self,
            tf_optimizer_cls=None,
            tf_optimizer_args=None,
            # learning_rate=1e-3,
            max_epochs=1000,
            tolerance=1e-6,
            batch_size=32,
            callback=None,
            verbose=False,
            **kwargs):
        """

        :param max_epochs:
        :param tolerance:
        :param update_method:
        :param batch_size: None or an integer. If None the whole dataset will be used.
        :param callback:
        :param kwargs:
        :return:
        """
        Serializable.quick_init(self, locals())
        self._opt_fun = None
        self._target = None
        self._callback = callback
        if tf_optimizer_cls is None:
            tf_optimizer_cls = tf.train.AdamOptimizer
        if tf_optimizer_args is None:
            tf_optimizer_args = dict(learning_rate=1e-3)
        self._tf_optimizer = tf_optimizer_cls(**tf_optimizer_args)
        self._max_epochs = max_epochs
        self._tolerance = tolerance
        self._batch_size = batch_size
        self._verbose = verbose
        self._input_vars = None
        self._train_op = None

    def update_opt(self, loss, target, inputs, extra_inputs=None, **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs
        :return: No return value.
        """

        self._target = target

        self._train_op = self._tf_optimizer.minimize(loss, var_list=target.get_params(trainable=True))

        # define operations for updating prior.
        update_mus = [(l.bayesreg.hyperparams['empirical'], l.bayesreg.w_mu.assign(l.W_mu)) for l in target.layers if hasattr(l, 'bayesreg')]
        update_rhos = [(l.bayesreg.hyperparams['empirical'], l.bayesreg.w_sig.assign(tf.log(1.0 + tf.exp(l.W_rho))))
                       for l in target.layers if hasattr(l, 'bayesreg')]
        self._update_priors_ops= update_mus + update_rhos

        # updates = OrderedDict([(k, v.astype(k.dtype)) for k, v in updates.iteritems()])

        if extra_inputs is None:
            extra_inputs = list()
        self._input_vars = inputs + extra_inputs
        self._opt_fun = ext.lazydict(
            f_loss=lambda: tensor_utils.compile_function(inputs + extra_inputs, loss),
        )

        if kwargs.has_key('like_loss'):
            l_loss=lambda: tensor_utils.compile_function(inputs + extra_inputs, kwargs['like_loss'])
            self._opt_fun.set('l_loss', l_loss)
            
        if kwargs.has_key('cmpx_loss'):
            c_loss=lambda: tensor_utils.compile_function(inputs + extra_inputs, kwargs['cmpx_loss'])
            self._opt_fun.set('c_loss', c_loss)

    def loss(self, inputs, extra_inputs=None):
        raise NotImplementedError # Not sure what this is for yet ...
        #if extra_inputs is None:
            #extra_inputs = tuple()
        #return self._opt_fun["f_loss"](*(tuple(inputs) + extra_inputs))

    def optimize(self, inputs, extra_inputs=None, callback=None,
                 val_inputs= None, val_extra_inputs= None):

        if len(inputs) == 0:
            # Assumes that we should always sample mini-batches
            raise NotImplementedError
        
        assert len(inputs) == 1

        dataset_size, _ = inputs[0].shape
        f_loss = self._opt_fun["f_loss"]
        
        # Plot individual costs from complexity / likelihood terms.
        try:
            use_c_loss= True
            c_loss = self._opt_fun["c_loss"]
        except KeyError:
            use_c_loss= False
        try:
            use_l_loss= True
            l_loss = self._opt_fun["l_loss"]
        except KeyError:
            use_l_loss= False        

        if extra_inputs is None:
            extra_inputs = tuple()

        #last_loss = f_loss(*(tuple(inputs) + extra_inputs))

        start_time = time.time()

        train_dataset = BatchDataset(inputs, self._batch_size, extra_inputs=extra_inputs)
        if val_inputs is not None:
            val_dataset = BatchDataset(val_inputs, self._batch_size, extra_inputs= val_extra_inputs)

        sess = tf.get_default_session()

        for epoch in range(self._max_epochs):
            if self._verbose:
                logger.log("Epoch %d" % (epoch))
                progbar = pyprind.ProgBar(len(inputs[0]))

            train_losses= []
            train_c_losses, train_l_losses = [], []
            for t, batch in enumerate(train_dataset.iterate(update=True)): # batch is a (matrix X, matrix Y) tuple
                sess.run(self._train_op, dict(list(zip(self._input_vars, batch))))
                train_losses.append(f_loss(*batch))
                train_c_losses.append(c_loss(*batch))
                train_l_losses.append(l_loss(*batch))
                
                if self._verbose:
                    progbar.update(len(batch[0]))
                    
            train_loss = np.mean(train_losses)
            train_c_loss = np.mean(train_c_losses)
            train_l_loss = np.mean(train_l_losses)
            
            val_losses= []
            if val_inputs is not None:
                for t, batch in enumerate(val_dataset.iterate(update= True)):
                    val_losses.append(f_loss(*batch))
                    
                val_loss = np.mean(val_losses)
                                
            for interval, op in self._update_priors_ops:
                if interval != 0 and epoch % interval == 0:
                    sess.run(op)
            
            if self._verbose:
                if progbar.active:
                    progbar.stop()           

            if self._verbose:
                logger.log("Epoch: %d | Loss: %f" % (epoch, train_loss))
            if self._callback or callback:
                elapsed = time.time() - start_time
                callback_args = dict(
                    loss=train_loss,
                    params=self._target.get_param_values(trainable=True) if self._target else None,
                    itr=epoch,
                    elapsed=elapsed,
                )
                if use_c_loss:
                    callback_args['c_loss'] = train_c_loss
                if use_l_loss:
                    callback_args['l_loss'] = train_l_loss                    
                    
                if val_loss is not None:
                    callback_args.update({'val_loss': val_loss})
                if self._callback:
                    self._callback(callback_args)
                if callback:
                    callback(**callback_args)

            #if abs(last_loss - train_loss) < self._tolerance:
                #break
            #last_loss = train_loss
