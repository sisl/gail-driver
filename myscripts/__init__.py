import matplotlib.pyplot as plt
import numpy as np

Xgen = lambda N, M : np.random.uniform(-5,5,(N,M))

def pringle(N, M, std):
    X = Xgen(N,M)
    Y = (X[:,0]**2 - X[:,1]**2 + np.random.normal(0,std,(N,)))[...,None]    
    return X, Y

def wave(N, M, O, std):
    X = Xgen(N,M)
    #Y = (np.cos(X[:,0]) + np.sin(X[:,1]) + np.random.normal(0,std,(N,)))[...,None]
    Y = np.column_stack([np.cos(X[:,i])[...,None] if i % 2 == 0 else np.sin(X[:,i])[...,None] for i in range(M)])
    lis = np.array_split(Y,O,axis=1)
    Y = np.column_stack([l.sum(axis=1)[...,None] for l in lis])
    return X, Y

def gaussians(N, M, mus, stds, one_hot= False):
    X, Y = [],[]
    
    K = len(mus)    
    assert len(mus) == len(stds), "Number of means and stdevs must match"
    assert N % K == 0, "{N} not divisible by number of Gaussians, {K}".format(N=N,K=K)
    
    for i, (mu, std) in enumerate(zip(mus,stds)):
        X.append(np.random.normal(mu, std, (N/K, M)))
        if one_hot:
            Y.append(
                np.repeat(np.eye(K)[i][None,...], N/K, axis= 0)
                )
        else:
            Y.append(
                np.repeat(np.array([[i]]), N/K, axis= 0)
            )
            
    X = np.row_stack(X)
    Y = np.row_stack(Y)
    return X, Y

def normalize(X, Y, normalize_targets= False, epsilon= 1e-6):
    ## Normalize data
    X -= X.mean(axis= 0)
    X /= (X.std(axis= 0) + epsilon)
    if normalize_targets:
        Y -= Y.mean(axis= 0)
        Y /= (Y.std(axis= 0) + epsilon)
    
    return X, Y

def rescale(X, Y, normalize_targets= False, binarize= False):
    ## Normalize data
    X = (X - X.min())/(X.max() - X.min())
    if binarize:
        X = (X > 0.5).astype('float32')
    
    return X, Y

def permute_and_split(X, Y, p_train= 0.7):
    ## retrieve dimensions
    N = X.shape[0]
    n_train = int(N * p_train)
    
    ## permute data
    p = np.random.permutation(N)
    X, Y = X[p], Y[p]
    
    ## Split data
    X_t, Y_t = X[:n_train], Y[:n_train]
    X_v, Y_v = X[n_train:], Y[n_train:]
    
    return X_t, Y_t, X_v, Y_v

class Visualizer(object):
    def __init__(self, enabled= True):
        self.tloss = {}
        self.vloss = {}
        self.closs = {}
        self.lloss = {}
        
        self.curr_model= ''
        self._enabled= enabled
        
    def add_model(self, model_name):
        if not self.tloss.has_key(model_name):
            self.tloss[model_name]= []
            self.vloss[model_name]= []
            self.closs[model_name]= []
            self.lloss[model_name]= []
            
        else:
            print("Warning: model already exists in visualizer.")
            
        self.curr_model= model_name
        
    def append_data(self, loss, params, itr, elapsed, val_loss= None, c_loss= None, l_loss= None, verbose= True):
        if verbose:
            print("Epoch: {:05d} == Loss: {:.5f} == Val Loss: {:.5f}".format(itr,loss,val_loss))
        self.tloss[self.curr_model].append(loss)
        self.vloss[self.curr_model].append(val_loss)
        self.closs[self.curr_model].append(c_loss)
        self.lloss[self.curr_model].append(l_loss)
        
    def display_image_batch(self, X):
        f, axs = plt.subplots(1, 5)
        for i, ax in enumerate(axs):
            ax.imshow(X[i])
          
        if self.enabled:  
            plt.show()
        
    def plot(self):
        names= self.tloss.keys()
        f, axs = plt.subplots(2,len(names))
        if axs.ndim == 1:
            axs= axs[...,None]
        
        for name, col in zip(names, axs.transpose()):
            ax1, ax2 = col[0], col[1]
            ax1.set_title(name)
            
            ax1.plot(self.tloss[name],'b')            
            ax1.plot(self.vloss[name],'r')
            
            ax2.plot(self.closs[name],'g')
            ax2.plot(self.lloss[name],'k')
            
        if self.enabled:
            plt.show()
            
    @property
    def enabled(self):
        return self._enabled
    
    @enabled.setter
    def enabled(self, value):
        return value