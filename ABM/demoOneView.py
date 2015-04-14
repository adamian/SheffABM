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


data = pods.datasets.brendan_faces()
Q = 2
data = pods.datasets.oil_100()
Y = data['X']

Yn = Y - Y.mean()
Yn /= Yn.std()

Y = {'Y':Yn}

a=ABM.LFM()
a.store(observed=Y, inputs=None, Q=15, kernel=None, num_inducing=20)
a.add_labels(data['Y'].argmax(axis=1))
a.learn('bfgs',True,1000)
ret = a.visualise()

