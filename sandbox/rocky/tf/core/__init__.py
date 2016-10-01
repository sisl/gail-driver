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

def approx_kl_regularizer(t, name= None, k= 1): # smaller seems better ...
    """
    Computes the kl divergence between distribution defined by tensor t
    and standard normal guassian.
    
    t is split in half. Top rows interpreted as mean, bottom rows as stdev.
    """
    mu, rho = tf.split(1,2,t)
    sig = tf.log(1.0 + tf.exp(rho))
    
    posterior = tf.contrib.distributions.Normal(mu, sig)
    prior = tf.contrib.distributions.Normal(0.0,1.0)
    
    samples= posterior.sample(k)
    diff = tf.reduce_mean(posterior.log_pdf(samples),[1,2]) - tf.reduce_mean(prior.log_pdf(samples),[1,2])    
    
    return tf.reduce_sum(diff)