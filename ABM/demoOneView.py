# Copyright (c) 2015, Andreas Damianou

import matplotlib as mp
# Use this backend for when the server updates plots through the X 
mp.use('TkAgg')
import numpy as np
import pylab as pb
import GPy
pb.ion()
default_seed = 123344
import pods
#from ABM import ABM
import ABM


# data = pods.datasets.brendan_faces()
Q = 2
Ntr = 100
Nts = 50

data = pods.datasets.oil()
Y = data['X'] # Data
L = data['Y'] # Labels

perm = np.random.permutation(Y.shape[0])
indTs = perm[0:Nts]
indTs.sort()
indTr = perm[Nts:Nts+Ntr]
indTr.sort()
Ytest = Y[indTs]
Ltest = L[indTs]
Y = Y[indTr]
L = L[indTr]

Yn = Y - Y.mean()
Yn /= Yn.std()

Y = {'Y':Yn}

a=ABM.LFM()
a.store(observed=Y, inputs=None, Q=Q, kernel=None, num_inducing=40)
a.add_labels(L.argmax(axis=1))
a.learn(optimizer='bfgs',max_iters=2000, verbose=True)
ret = a.visualise()

pred_mean, pred_var = a.pattern_completion(Ytest)
pb.plot(pred_mean[:,0],pred_mean[:,1],'bx')