from keras.datasets.mnist import load_data as load_mnist
import numpy as np

def mnist(flat= True, one_hot= True, as_float= True):
    data, _ = load_mnist()
    X, Y = data
    
    n_classes = len(np.unique(Y))
    N, W, H = X.shape
    
    if one_hot:
        Y = np.eye(n_classes)[Y]
    else:
        Y = Y[...,None]
    if flat:
        X = X.reshape((N, H * W))
    if as_float:
        X = X.astype('float32')
        Y = Y.astype('float32')
        
    return X, Y
