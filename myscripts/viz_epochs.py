import numpy as np
import matplotlib.pyplot as plt

import os
import argparse

import pickle
import joblib

headers = ('MaxReturn','LossAfter','raccpi','MeanKLBefore','dLoss','NumTrajs','Iteration','AverageDiscountedReturn',
'MinReturn','StdReturn','LossBefore','ItrTime','Entropy','AveragePolicyStd','Time','Perplexity','MeanKL',
'ExplainedVariance','raccex','AverageReturn','racc')

parser = argparse.ArgumentParser()
parser.add_argument('--exp_names',type=str,nargs='+')
parser.add_argument('--iters',type=int,nargs='+')
parser.add_argument('--max_iter',type=int,default=-1)

args = parser.parse_args()

max_iters = args.max_iter

if args.iters is None:
    iters = [-1 for _ in args.exp_names]
else:
    iters = args.iters

# plotting
f, axs = plt.subplots(nrows= len(args.exp_names), ncols= 3)

if axs.ndim == 1:
    axs = axs[None,...]

for i, exp_name in enumerate(args.exp_names):
    path = '../data/' + exp_name + '/'
    #pkls = [f for f in os.listdir(path) if f.split('.')[-1] == 'pkl']

    X = np.genfromtxt(path + 'tab.txt', delimiter=',')[1:]
    headers = [header.replace('"','') for header in np.genfromtxt(path + 'tab.txt', dtype=str, delimiter=',')[0]]

    #M = np.row_stack([np.array(list(line)) for line in f[1:]]).astype('float32')
    M = X

    axrow = axs[i]
    #axcol[0].plot(M[:,headers.index('AverageReturn')], 'r')
    #axcol[0].plot(M[:,headers.index('AverageDiscountedReturn')], 'b')
    axrow[0].set_title("Return")
    axrow[0].plot(M[:max_iters,headers.index('AverageReturn')], 'r')
    axrow[0].plot(M[:max_iters,headers.index('AverageDiscountedReturn')], 'b')

    axrow[0].set_xticklabels([])

    try:
        averageEnvReturn = M[:max_iters,headers.index('AverageEnvReturn')]
        #axrow[1]


        axrow[1].set_title("Env. Return ({low_epoch}:{low},{high_epoch}:{high})".format(
            low_epoch=averageEnvReturn.argmin(),
            low=int(averageEnvReturn.min()),
            high_epoch=averageEnvReturn.argmax(),
            high=int(averageEnvReturn.max())))


        axrow[1].plot(averageEnvReturn, 'r')
        #axrow[1].plot(M[:max_iters,headers.index('AverageDiscountedEnvReturn')], 'b')

        axrow[2].set_title("GAIL accuracy")
        axrow[2].plot(M[:max_iters,headers.index('raccex')], 'r')
        axrow[2].plot(M[:max_iters,headers.index('racc')], 'g')
        axrow[2].plot(M[:max_iters,headers.index('raccpi')], 'b')

        axrow[1].set_xticklabels([])
        axrow[2].set_xticklabels([])

    except:
        pass

plt.show()

halt= True