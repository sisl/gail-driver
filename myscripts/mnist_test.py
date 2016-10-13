import numpy as np
from data import mnist

import tensorflow as tf

from sandbox.rocky.tf.optimizers.first_order_optimizer import Solver

from myscripts import Visualizer, normalize, permute_and_split, rescale

from sandbox.rocky.tf.core.network import LatentMLP

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs',type=int,default=10)
parser.add_argument('--reg',type=float,default=1e-2)
parser.add_argument('--cmx',type=float,default=1.0)
parser.add_argument('--H_nonlin',type=str,default='tanh')
parser.add_argument('--O_nonlin',type=str,default='linear')
parser.add_argument('--visualize',type=bool,default=False)

args = parser.parse_args()

X, Y = mnist()
X_t, Y_t, X_v, Y_v = permute_and_split(*rescale(*mnist(),normalize_targets= False, binarize= True),p_train= 0.7)

X_t = X_t[:2000]
Y_t = Y_t[:2000]

## Create dataset
N, M = X.shape
H = 128
Z = 7
V = int(0.7 * N)

nonlinearities = {
    'tanh' : tf.nn.tanh,
    'relu' : tf.nn.relu,
    'softplus' : tf.nn.softplus,
    'linear' : tf.identity,
    'sigmoid' : tf.nn.sigmoid
}

BATCH_SIZE = 50

viz = Visualizer(enabled= args.visualize)
viz.display_image_batch(X_t[:BATCH_SIZE].reshape(BATCH_SIZE,28,28)) # this looks good when visualized ...
        
H_SPEC = [('dense',H),('dense',H),('latent',Z),('dense',H),('dense',H)]
vae = LatentMLP('llmlp_1', M, H_SPEC, nonlinearities[args.H_nonlin], 
                output_nonlinearity = nonlinearities[args.O_nonlin], batch_size= BATCH_SIZE,
                input_shape= (M,), reg_params={'approx':True,'muldiag':True,'empirical':0,'samples':1})
viz.add_model(vae)

def callback(**kwargs):
    viz.append_data(**kwargs)
    vae.save_params('vae', kwargs['itr'], overwrite= True)

optimizer = Solver(vae, 0.0, args.cmx, args.epochs, BATCH_SIZE, None, callback= callback)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    optimizer.train(X_t, X_t, X_v, X_v)
    
    L = vae.predict(X_t)[:BATCH_SIZE].reshape(BATCH_SIZE,28,28)
    viz.display_image_batch(L) # Afer 99 epochs, looks fuzzy, but like its on the right track.
    
    L = vae.generate().reshape(BATCH_SIZE,28,28)
    viz.display_image_batch(L) # looks wrong
    
    #z_batch = np.repeat(np.random.normal(size=(1,Z)),BATCH_SIZE,axis=0)
    z_batch = np.zeros((BATCH_SIZE, Z))
    L = vae.generate(Z = z_batch).reshape(BATCH_SIZE,28,28)
    viz.display_image_batch(L)
    
    viz.enabled = True
    viz.plot()
    
halt = True