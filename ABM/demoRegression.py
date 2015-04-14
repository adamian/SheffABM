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

data=dict()
N=80
Nts=20
X = np.array(range(N))*0.14
X=X[None,:].T
Y = np.sin(X) + np.random.randn(*X.shape) * 0.05
perm = np.random.permutation(X.shape[0])
indTs = perm[1:Nts]
indTs.sort()
indTr = perm[Nts:]
indTr.sort()
Xtest = X[indTs]
Ytest = Y[indTs]
X = X[indTr]
Y = Y[indTr]

#data = pods.datasets.olympic_marathon_men()
# Y = data['Y']
# X = data['X']


Ymean = Y.mean(0)
Y = Y - Ymean
Ystd = Y.std(0)
Y /= Ystd

Ydict = {'Y':Y}
a=ABM.LFM()
a.store(observed=Ydict, inputs=X, Q=None, kernel=None, num_inducing=20)
a.learn()
ret = a.visualise()

pred_mean, pred_var = a.model.predict(Xtest)#a.pattern_completion(Xtest)
pb.figure()
pb.plot(Xtest, Ytest, 'r-x')
pb.plot(Xtest, pred_mean, 'b-x')
pb.axis('equal')
pb.title('Pattern Completion given Novel Inputs')
pb.legend(('True Location', 'Predicted Location'))

sse = ((Ytest - pred_mean)**2).sum()
print('Sum of squares error on test data: ' + str(sse))