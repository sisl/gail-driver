from sandbox.rocky.tf.core.network import MLP, BayesMLP, LatentMLP
import tensorflow as tf
import numpy as np
import argparse

import matplotlib.pyplot as plt

from myscripts import Visualizer, gaussians, wave, normalize, permute_and_split

from sandbox.rocky.tf.optimizers.first_order_optimizer import Solver

from sklearn.metrics import accuracy_score

SEED = 320
np.random.seed(SEED)
tf.set_random_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs',type=int,default= 500)
parser.add_argument('--classif',type=bool,default= False)
parser.add_argument('--regress',type=bool,default= False)
parser.add_argument('--one_hot',type=bool,default= True)
parser.add_argument('--batch_size',type=int,default= 20)

parser.add_argument('--reg',type=float,default=0.0) # weight used for parameter regularization
parser.add_argument('--cmx',type=float,default=1.0) # weight used for data-dependent regularization

args = parser.parse_args()

assert args.classif or args.regress and not (args.classif and args.regress)

## Create dataset
N, V = 600, 100
M, H, O = 10, 20, 3

if args.classif:
    mus = [np.ones((M,)) * -0.1, np.ones((M,)) * 2.0, np.ones((M,)) * 0.1]
    stds = [np.ones((M,)), np.ones((M,)), np.ones((M,))]
    
    X_t, Y_t, X_v, Y_v = permute_and_split(*normalize(*gaussians(N, M, mus, stds, one_hot= args.one_hot)), p_train = 0.7)
    output_nonlinearity= tf.nn.softmax
        
elif args.regress:
    X_t, Y_t, X_v, Y_v = permute_and_split(*normalize(*wave(N, M, O, 1.),normalize_targets=True),n_total=N,n_val=V)
    output_nonlinearity= tf.identity
        
mlp = MLP('mlp', O, [H], tf.nn.tanh, output_nonlinearity, input_shape= (M,)) # 0.884 - 0.72

bmlp = BayesMLP('bmlp_1', O, [H], tf.nn.tanh, output_nonlinearity, input_shape= (M,), reg_params={'approx':True,'muldiag':True,'empirical':0,'samples':1}) # 0.702 - 0.64

H_SPEC = [('dense',H/2),('latent',5),('dense',H/2)]
lmlp = LatentMLP('llmlp_1', O, H_SPEC, tf.nn.tanh, output_nonlinearity, batch_size= args.batch_size, input_shape= (M,), reg_params={'approx':False,'muldiag':True,'empirical':0,'samples':1})

models = [mlp, bmlp, lmlp]

viz = Visualizer()

acc= []
for model in models:
    
    viz.add_model(model.name)
    optimizer = Solver(model, args.reg, args.cmx, args.epochs, args.batch_size, tolerance= 1e-6, 
                      callback= viz.append_data)
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        optimizer.train(X_t, Y_t, X_v, Y_v)
        
        Y_p_t = model.predict(X_t)
        Y_p_v = model.predict(X_v)
        
    if args.classif: 
        if args.one_hot:
            Y_a_t = np.argmax(Y_t, axis = 1)
            Y_a_v = np.argmax(Y_v, axis = 1)
        else:
            Y_a_t = Y_t
            Y_a_v = Y_v
           
        acc.append((accuracy_score(Y_p_t,Y_a_t), accuracy_score(Y_p_v,Y_a_v)))
        
if args.classif:
    for model, (t, v) in zip(models,acc):
        print("Model: {} ({} parameters)".format(model.name,model.n_params))
        print("Training accuracy: {} == Validation accuracy: {}".format(t, v))
viz.plot()

halt= True

