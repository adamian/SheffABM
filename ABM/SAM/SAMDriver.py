#!/usr/bin/python

import matplotlib.pyplot as plt
#import matplotlib as mp
import pylab as pb
import sys
#import pickle
import numpy
import os
#import yarp
#import cv2
import GPy
#import time
#from scipy.spatial import distance
#import operator
#from .SAMCore import *


#""""""""""""""""
#Class developed for the implementation of the face recognition task in real-time mode.
#""""""""""""""""

class SAMDriver:
    def __init__(self, isYarpRunning = False, dataDir=""):
        
        # self.inputImagePort=inputImagePort  # MOVE to FaceDriver.py
        
        self.SAMObject=SAMCore.LFM()        
        # self.imgHeight = imgH  # MOVE to FaceDriver.py
        # self.imgWidth = imgW  # MOVE to FaceDriver.py
        # self.imgHeightNew = imgHNew  # MOVE to FaceDriver.py
        # self.imgWidthNew = imgWNew  # MOVE to FaceDriver.py
        # self.image_suffix=".ppm"  # MOVE to FaceDriver.py

        self.Y = None
        self.L = None
        self.X = None
        self.Ytest = None
        self.Ltest = None
        self.Ytestn = None
        self.Ltestn = None
        self.Ymean = None
        self.Ystd = None
        self.Yn = None
        self.Ln = None
        self.data_labels = None
        # self.participant_index = None  # MOVE to FaceDriver.py

        self.model_num_inducing = 0
        self.model_num_iterations = 0
        self.model_init_iterations = 0

        # if( isYarpRunning == True ):  # MOVE to FaceDriver.py
        #     yarp.Network.init()  # MOVE to FaceDriver.py
        #     self.createPorts()  # MOVE to FaceDriver.py
        #     self.openPorts()  # MOVE to FaceDriver.py
        #     self.createImageArrays()  # MOVE to FaceDriver.py


#""""""""""""""""
#Methods to create the ports for reading images from iCub eyes
#Inputs: None
#Outputs: None
#""""""""""""""""
    def createPorts(self):
        raise NotImplementedError("this needs to be implemented to use the model class")

#""""""""""""""""
#Method to open the ports. It waits until the ports are connected
#Inputs: None
#Outputs: None
#""""""""""""""""
    def openPorts(self):
        raise NotImplementedError("this needs to be implemented to use the model class")

    #def readData(self, ...):
    #    raise NotImplementedError("this needs to be implemented to use the model class")

    # Perhaps the following method also has to be implemented by the subclass
    def prepareData(self, model='mrd', Ntr = 50):    
        # TODO: this should be ready from readData: self.Y = ...
        # TODO: this should be ready from readData: self.L = ...
        # TODO: this should be ready from readData: self.Ln = ...

        Nts=self.Y.shape[0]-Ntr
   
        perm = numpy.random.permutation(self.Y.shape[0])
        indTs = perm[0:Nts]
        indTs.sort()
        indTr = perm[Nts:Nts+Ntr]
        indTr.sort()
        self.Ytest = self.Y[indTs]
        self.Ltest = self.L[indTs]
        self.Y = self.Y[indTr]
        self.L = self.L[indTr]
    
        # Center data to zero mean and 1 std
        self.Ymean = self.Y.mean()
        self.Yn = self.Y - self.Ymean
        self.Ystd = self.Yn.std()
        self.Yn /= self.Ystd
        # Normalise test data similarly to training data
        self.Ytestn = self.Ytest - self.Ymean
        self.Ytestn /= self.Ystd

        # As above but for the labels
        self.Lmean = self.L.mean()
        self.Ln = self.L - self.Lmean
        self.Lstd = self.Ln.std()
        self.Ln /= self.Lstd
        self.Ltestn = self.Ltest - self.Lmean
        self.Ltestn /= self.Lstd

        if model == 'mrd':    
            self.X=None     
            self.Y = {'Y':self.Yn,'L':self.L}
            self.data_labels = self.L.copy()
        elif model == 'gp':
            self.X=self.Y.copy()
            self.Y = {'L':self.Ln.copy()+0.08*numpy.random.randn(self.Ln.shape[0],self.Ln.shape[1])}
            self.data_labels = None
        elif model == 'bgplvm':
            self.X=None     
            self.Y = {'Y':self.Yn}
            self.data_labels = self.L.copy()

#""""""""""""""""
#Method to train, store and load the learned model to be use for the face recognition task
#Inputs:
#    - modelNumInducing:
#    - modelNumIterations:
#    - modelInitIterations:
#    - fname: file name to store/load the learned model
#    - save_model: enable/disable to save the model
#
#Outputs: None
#""""""""""""""""
    def training(self, modelNumInducing, modelNumIterations, modelInitIterations, fname, save_model):
        self.model_num_inducing = modelNumInducing
        self.model_num_iterations = modelNumIterations
        self.model_init_iterations = modelInitIterations
    
        if not os.path.isfile(fname + '.pickle'):
            print "Training..."    
            if self.X is not None:
                Q = self.X.shape[1]
            else:
                Q=2

            if Q > 100:
                kernel = GPy.kern.RBF(Q, ARD=False) + GPy.kern.Bias(Q) + GPy.kern.White(Q)
            else:
                kernel = None
            # Simulate the function of storing a collection of events
            self.SAMObject.store(observed=self.Y, inputs=self.X, Q=Q, kernel=kernel, num_inducing=self.model_num_inducing)
            # If data are associated with labels (e.g. face identities), associate them with the event collection
            if self.data_labels is not None:
                self.SAMObject.add_labels(self.data_labels)
            # Simulate the function of learning from stored memories, e.g. while sleeping (consolidation).
            self.SAMObject.learn(optimizer='scg',max_iters=self.model_num_iterations, init_iters=self.model_init_iterations, verbose=True)
	
            print "Saving SAMObject"
            if save_model:
                SAMCore.save_pruned_model(self.SAMObject, fname)
        else:
	        print "Loading SAMOBject"
	        self.SAMObject = SAMCore.load_pruned_model(fname)



    def testing(self, testInstance, visualiseInfo=None):
        raise NotImplementedError("this needs to be implemented to use the model class")