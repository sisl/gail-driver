import tensorflow as tf
import numpy as np


def flatten(t):
    flat_dim = np.prod(t.get_shape().as_list())
    return tf.reshape(t, (-1, flat_dim))


class BayesRegularizer(object):
    def __init__(self, fan_in, fan_out,
                 hyperparams=dict(approx=True,
                                  muldiag=True,
                                  samples=1,
                                  empirical=0)
                 ):
        self.hyperparams = hyperparams
        self.fan_in = fan_in
        self.fan_out = fan_out

        self._mu = tf.Variable(np.zeros((fan_in, fan_out)),
                               trainable=False, name='prior_mu', dtype=tf.float32)
        self._sig = tf.Variable(
            np.ones((fan_in, fan_out)), trainable=False, name='prior_sig', dtype=tf.float32)

    def weight_kl(self, t, name=None):  # smaller seems better ...
        """
        Computes the kl divergence between distribution defined by tensor t
        and standard normal guassian.

        t is split in half. Top rows interpreted as mean, bottom rows as stdev.
        """
        # unpack posterior params
        mu, rho = tf.split(1, 2, t)
        sig = tf.log(1.0 + tf.exp(rho))

        # flatten mu and stdev.
        mu, sig = flatten(mu), flatten(sig)
        mu_prior, sig_prior = flatten(self.w_mu), flatten(self.w_sig)

        kl_d = self._compute_kl_d(mu, sig, mu_prior, sig_prior)

        return tf.reduce_sum(kl_d)

    def activ_kl(self, mu, sig):
        """
        Given batch of mu and sigma vectors, compute kl divergence from standard gaussian.
        """
        mu_prior = tf.constant(np.zeros((self.fan_out,)), dtype=tf.float32)
        sig_prior = tf.constant(np.ones((self.fan_out,)), dtype=tf.float32)

        kl_d = self._compute_kl_d(mu, sig, mu_prior, sig_prior)

        return tf.reduce_sum(kl_d)

    def _compute_kl_d(self, mu, sig, mu_prior, sig_prior):
        """
        helper function for computing kl divergence.
        """
        # unpack hyperparams
        k = self.hyperparams['samples']
        muldiag = self.hyperparams['muldiag']
        approx = self.hyperparams['approx']

        if muldiag:
            posterior = tf.contrib.distributions.MultivariateNormalDiag(
                mu, sig, name="posterior")
            prior = tf.contrib.distributions.MultivariateNormalDiag(
                mu_prior, sig_prior, name="prior")

            if approx:
                kl_d = posterior.log_pdf(posterior.sample(
                    k)) - prior.log_pdf(posterior.sample(k))
            else:
                kl_d = tf.contrib.distributions.kl(posterior, prior)
        else:
            posterior = tf.contrib.distributions.Normal(
                mu, sig, name="posterior")
            prior = tf.contrib.distributions.Normal(
                mu_prior, sig_prior, name="prior")

            if approx:
                kl_d = tf.reduce_mean(posterior.log_pdf(posterior.sample(k)), [1, 2])\
                    - tf.reduce_mean(prior.log_pdf(posterior.sample(k)), [1, 2])
            else:
                kl_d = tf.contrib.distributions.kl(posterior, prior)

        return kl_d

    @property
    def w_mu(self):
        return self._mu

    @property
    def w_sig(self):
        return self._sig

    @w_mu.setter
    def w_mu(self, value):
        self._mu = value

    @w_sig.setter
    def w_sig(self, value):
        self._sig = value
