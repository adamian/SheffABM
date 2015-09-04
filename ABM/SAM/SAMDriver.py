
from ABM.SAM import SAMCore
import matplotlib.pyplot as plt
#import matplotlib as mp
import pylab as pb
import sys
#import pickle
import numpy
import os
import yarp
import cv2
import GPy
import time
from scipy.spatial import distance
import operator
"""
try:
    from SAM import SAM
except ImportError:
    import SAM
"""
import time



#""""""""""""""""
#Class developed for the implementation of the face recognition task in real-time mode.
#""""""""""""""""

class SAMDriver:
    #""""""""""""""""
    #Initilization of the SAM class
    #Inputs:
    #    - isYarprunning: specifies if yarp is used (True) or not(False)
    #    - imgH, imgW: original image width and height
    #    - imgHNewm imgWNew: width and height values to resize the image
    #
    #Outputs: None
    #""""""""""""""""
    def __init__(self, isYarpRunning = False):
                
        self.SAMObject=SAMCore.LFM()        

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

        self.model_num_inducing = 0
        self.model_num_iterations = 0
        self.model_init_iterations = 0





    #""""""""""""""""
    #Methods to create the ports for reading images from iCub eyes
    #Inputs: None
    #Outputs: None
    #""""""""""""""""
    def createPorts(self):
        self.imageDataInputPort = yarp.BufferedPortImageRgb()
        self.outputFacePrection = yarp.Port()
        self.speakStatusPort = yarp.RpcClient();
        self.speakStatusOutBottle = yarp.Bottle()
        self.speakStatusInBottle = yarp.Bottle()
        self.imageInputBottle = yarp.Bottle()

    #""""""""""""""""
    #Method to open the ports. It waits until the ports are connected
    #Inputs: None
    #Outputs: None
    #""""""""""""""""
    def openPorts(self):
        print "open ports"
        self.imageDataInputPort.open("/sam/face/imageData:i");
        self.outputFacePrection.open("/sam/face/facePrediction:o")
        self.speakStatusPort.open("/sam/face/speakStatus:i")
        self.speakStatusOutBottle.addString("stat")

        #print "Waiting for connection with imageDataInputPort..."
#        while( not(yarp.Network.isConnected(self.inputImagePort,"/sam/imageData:i")) ):
#            print "Waiting for connection with imageDataInputPort..."
#            pass

    def closePorts(self):
        print "open ports"
        self.actionDataInputPort.close();
        self.outputActionPrediction.close()
        self.speakStatusPort.close()
        #self.speakStatusOutBottle.addString("stat")
        #print "Waiting for connection with actionDataInputPort..."
        #while( not(yarp.Network.isConnected(self.inputActionPort,"/sam/actionData:i")) ):
        #    print "Waiting for connection with actionDataInputPort..."
        #    pass

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


    def testing(self, testInstance, choice, visualiseInfo=None):
        raise NotImplementedError("this needs to be implemented to use the model class")
