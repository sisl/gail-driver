import tensorflow as tf
import numpy as np

def gaussian_kl_regularizer(t, name= None):
    """
    Computes the kl divergence between distribution defined by tensor t
    and standard normal guassian.
    
    t is split in half. Top rows interpreted as mean, bottom rows as stdev.
    """
    mu, rho = tf.split(1,2,t)
    sig = tf.log(1.0 + tf.exp(rho))
    
    dist_a = tf.contrib.distributions.Normal(mu, sig)
    dist_b = tf.contrib.distributions.Normal(0.0,1.0)
    
    return tf.reduce_sum( tf.contrib.distributions.kl(dist_a, dist_b) )

#def approx_kl_regularizer(t, name= None, k= 5): # smaller seems better ...
    #"""
    #Computes the kl divergence between distribution defined by tensor t
    #and standard normal guassian.
    
    #t is split in half. Top rows interpreted as mean, bottom rows as stdev.
    #"""
    #mu, rho = tf.split(1,2,t)
    #sig = tf.log(1.0 + tf.exp(rho))
    
    #posterior = tf.contrib.distributions.Normal(mu, sig)
    #prior = tf.contrib.distributions.Normal(0.0,1.0)
    
    #samples= posterior.sample(k)
    #kl_d = tf.reduce_mean(posterior.log_pdf(samples),[1,2]) - tf.reduce_mean(prior.log_pdf(samples),[1,2])    
    
    #return tf.reduce_sum(kl_d)

def flatten(t):
    flat_dim = np.prod(t.get_shape().as_list())
    return tf.reshape(t, (-1, flat_dim))

class BayesRegularizer(object):
    def __init__(self, fan_in, fan_out,
                 hyperparams= dict(approx = True,
                                   muldiag = True,
                                   samples = 1,
                                   empirical= 0)
                 ):
        self.hyperparams = hyperparams
        self._mu = tf.Variable(np.zeros((fan_in, fan_out)), trainable= False, name= 'prior_mu', dtype= tf.float32)
        self._sig = tf.Variable(np.ones((fan_in, fan_out)), trainable= False, name= 'prior_sig', dtype= tf.float32)
        
    def approx_kl(self, t, name= None): # smaller seems better ...
        """
        Computes the kl divergence between distribution defined by tensor t
        and standard normal guassian.
        
        t is split in half. Top rows interpreted as mean, bottom rows as stdev.
        """
        # unpack hyperparams
        k = self.hyperparams['samples']
        muldiag = self.hyperparams['muldiag']
        approx = self.hyperparams['approx']
        
        # unpack posterior params
        mu, rho = tf.split(1,2,t)
        sig = tf.log(1.0 + tf.exp(rho))
        
        # flatten mu and stdev.
        mu, sig = flatten(mu), flatten(sig)
        mu_prior, sig_prior = flatten(self.mu), flatten(self.sig)
        
        if muldiag:
            posterior = tf.contrib.distributions.MultivariateNormalDiag(mu, sig, name="posterior")
            prior = tf.contrib.distributions.MultivariateNormalDiag(mu_prior, sig_prior, name="prior")
            
            if approx:
                kl_d = posterior.log_pdf(posterior.sample(k)) - prior.log_pdf(posterior.sample(k))
            else:
                kl_d = tf.contrib.distributions.kl(posterior,prior)
        else:
            posterior = tf.contrib.distributions.Normal(mu, sig, name = "posterior")
            prior = tf.contrib.distributions.Normal(mu_prior, sig_prior, name = "prior")
            
            if approx:
                kl_d = tf.reduce_mean(posterior.log_pdf(posterior.sample(k)),[1,2])\
                    - tf.reduce_mean(prior.log_pdf(posterior.sample(k)),[1,2])
            else:
                kl_d = tf.contrib.distributions.kl(posterior,prior)
        
        #samples= posterior.sample(k)
        #kl_d = tf.reduce_mean(posterior.log_pdf(samples),[1,2]) - tf.reduce_mean(prior.log_pdf(samples),[1,2])
        #kl_d = posterior.log_pdf(samples) - prior.log_pdf(samples)            
        
        return tf.reduce_sum(kl_d)
    
    @property
    def mu(self):
        return self._mu
    
    @property
    def sig(self):
        return self._sig    
    
    @mu.setter
    def mu(self, value):
        self._mu = value
        
    @sig.setter
    def sig(self, value):
        self._sig = value
        
    
    