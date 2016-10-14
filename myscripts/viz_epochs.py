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

args = parser.parse_args()

if args.iters is None:
    iters = [-1 for _ in args.exp_names]
else:
    iters = args.iters

# plotting
f, axs = plt.subplots(nrows= 2, ncols= len(args.exp_names))

if axs.ndim == 1:
    axs = axs[None,...]

for i, exp_name in enumerate(args.exp_names):
    path = '../data/' + exp_name + '/'
    pkls = [f for f in os.listdir(path) if f.split('.')[-1] == 'pkl']

    f = np.genfromtxt(path + 'tab.txt', delimiter=',', dtype=None, names= headers)

    M = np.row_stack([np.array(list(line)) for line in f[1:]]).astype('float32')

    axcol = axs[i]
    axcol[0].plot(M[:,headers.index('AverageReturn')], 'r')
    axcol[0].plot(M[:,headers.index('AverageDiscountedReturn')], 'b')

    axcol[1].plot(M[:,headers.index('raccex')], 'r')
    axcol[1].plot(M[:,headers.index('racc')], 'g')
    axcol[1].plot(M[:,headers.index('raccpi')], 'b')

plt.show()

halt= True