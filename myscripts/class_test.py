from sandbox.rocky.tf.core.network import MLP, BayesMLP
import tensorflow as tf
import numpy as np
import argparse

import matplotlib.pyplot as plt

from myscripts import Visualizer, gaussians, wave, normalize, permute_and_split

from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer as FOO

from sklearn.metrics import accuracy_score

np.random.seed(456)
tf.set_random_seed(456)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs',type=int,default= 500)
parser.add_argument('--reg',type=float,default=0.0)
parser.add_argument('--classif',type=bool,default= False)
parser.add_argument('--regress',type=bool,default= False)
parser.add_argument('--one_hot',type=bool,default= True)

args = parser.parse_args()

assert args.classif or args.regress and not (args.classif and args.regress)

## Create dataset
N, V = 600, 100
M, H, O = 10, 20, 3

if args.classif:
    mus = [np.ones((M,)) * -0.1, np.ones((M,)) * 2.0, np.ones((M,)) * 0.1]
    stds = [np.ones((M,)), np.ones((M,)), np.ones((M,))]
    
    X_t, Y_t, X_v, Y_v = permute_and_split(*normalize(*gaussians(N, M, mus, stds, one_hot= args.one_hot)), n_total= N, n_val= V)
    output_nonlinearity= tf.nn.softmax
    
    y = tf.placeholder(dtype=tf.int32, shape= [None, O], name='labels')
    
    
elif args.regress:
    X_t, Y_t, X_v, Y_v = permute_and_split(*normalize(*wave(N, M, O, 1.),normalize_targets=True),n_total=N,n_val=V)
    output_nonlinearity= tf.identity
    
    y = tf.placeholder(dtype=tf.float32, shape= [None, O], name='labels')
    

mlp = MLP('mlp', O, [H], tf.nn.tanh, output_nonlinearity, input_shape= (M,))
bmlp = BayesMLP('bmlp', O, [H], tf.nn.tanh, output_nonlinearity, input_shape= (M,), approximate_kl= True)

models= [mlp, bmlp]

viz = Visualizer()

for model in models:
    
    x = model.input_var    
    viz.add_model(model.name)
    
    optimizer = FOO(max_epochs= args.epochs, tolerance= 1e-6)    
    optimizer.update_opt(model.loss(y, reg= args.reg), model, [x], extra_inputs= [y])
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        optimizer.optimize([X_t], extra_inputs=tuple([Y_t]), callback= viz.append_data,
                     val_inputs= [X_v], val_extra_inputs= tuple([Y_v]))
        
        Y_p_t = model.predict(X_t)
        Y_p_v = model.predict(X_v)
        
    if args.one_hot:
        Y_a_t = np.argmax(Y_t, axis = 1)
        Y_a_v = np.argmax(Y_v, axis = 1)
    else:
        Y_a_t = Y_t
        Y_a_v = Y_v
        
    if args.classif:
        print("Training accuracy: {} == Validation accuracy: {}".format(accuracy_score(Y_p_t,Y_a_t), accuracy_score(Y_p_v,Y_a_v)))
viz.plot()

halt= True

