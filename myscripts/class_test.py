from sandbox.rocky.tf.core.network import MLP, BayesMLP
import tensorflow as tf
import numpy as np
import argparse

import matplotlib.pyplot as plt

from myscripts import Visualizer, gaussians, normalize, permute_and_split

from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer as FOO

from sklearn.metrics import accuracy_score

np.random.seed(456)
tf.set_random_seed(456)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs',type=int,default= 500)
parser.add_argument('--reg',type=float,default=0.0)

args = parser.parse_args()

## Create dataset
N, V = 600, 100
M, H, O = 10, 20, 3

mus = [np.ones((M,)) * -0.1, np.ones((M,)) * 2.0, np.ones((M,)) * 0.1]
stds = [np.ones((M,)), np.ones((M,)), np.ones((M,))]
X_t, Y_t, X_v, Y_v = permute_and_split(*normalize(*gaussians(N, M, mus, stds)), n_total= N, n_val= V)

mlp = MLP('mlp', O, [H], tf.nn.tanh, tf.nn.softmax, input_shape= (M,))
bmlp = BayesMLP('bmlp', O, [H], tf.nn.tanh, tf.nn.softmax, input_shape= (M,), approximate_kl= True)

models= [mlp, bmlp]

viz = Visualizer()

for model in models:
    
    x = model.input_var
    y = tf.placeholder(dtype=tf.int32, shape= [None, 1], name='labels')
    
    viz.add_model(model.name)
    
    optimizer = FOO(max_epochs= args.epochs, tolerance= 1e-6)    
    optimizer.update_opt(model.loss(y, reg= args.reg), model, [x], extra_inputs= [y])
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        optimizer.optimize([X_t], extra_inputs=tuple([Y_t]), callback= viz.append_data,
                     val_inputs= [X_v], val_extra_inputs= tuple([Y_v]))
        
        Y_p_t = model.predict(X_t)
        Y_p_v = model.predict(X_v)
        
    print("Training accuracy: {} == Validation accuracy: {}".format(accuracy_score(Y_p_t,Y_t), accuracy_score(Y_p_v,Y_v)))
viz.plot()

halt= True

