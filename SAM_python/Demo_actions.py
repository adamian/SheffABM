#!/usr/bin/python

#
#The University of Sheffield
#WYSIWYD Project
#
#Example of implementation of SAMpy class
#
#Created on 29 May 2015
#
#@authors: Uriel Martinez, Luke Boorman, Andreas Damianou
#
#

import matplotlib.pyplot as plt
from SAMpy import SAMpy
import pylab as pb
import sys
import pickle
import os
import numpy
import time
import operator


# Creates a SAMpy object
# TODO: mySAMpy = SAMpy(True, imgH = 400, imgW = 400, imgHNew = 200, imgWNew = 200,inputImagePort="/visionDriver/image:o")

# Specification of the experiment number
experiment_number = 11

# Location of face data
root_data_dir="/home/icub/dataDump/faceImageData_11062015"
# Image format
image_suffix=".ppm"
# Use a subset of the data for training
Ntr=300

# Action selected for training
action_selection = 0

# Specification of model type and training parameters
model_type = 'mrd'
model_num_inducing = 35
model_num_iterations = 100
model_init_iterations = 800
fname = 'm_' + model_type + '_exp' + str(experiment_number) #+ '.pickle'

# Enable to save the model and visualise GP nearest neighbour matching
save_model=True
visualise_output=True

# Reading action data, preparation of data and training of the model
# TODO #mySAMpy.readFaceData(root_data_dir, participant_index, pose_index)
# TODO #mySAMpy.prepareFaceData(model_type, Ntr, pose_selection)
# TODO #mySAMpy.training(model_num_inducing, model_num_iterations, model_init_iterations, fname, save_model)

# TODO: The above will read the action data and transform them so that:
# We have A different actions in the dictionary. E.g. A=4, for pull, push, up, down.
# We have recorded N actions stored in a dictionary Y0. Y['n'] is a S_n x D matrix, saying that for
# the n-th action we recorded S_n D-dimensional steps. Here we'll take D=2, so that each step is the
# x-y coordinates (later add z) in the plane. 
# We also have X0 being a N x S_n matrix, with the times at which Y0['n'] were observed.
# Now for each pair { X0[n], Y['n']} we fit a linear model and obtain the parameters [a,b] (the slope and
# intercept of the line). We store these parameters in the N x 2 matrix in position Y[n,:].
# We finally create a matrix L which is N x A, which is saying what sort of action Y[n,:] corresponds to
# (1-of-K encoding).

#--- Move this to corresponding Driver:ActionDriver:read module and Driver:prepare
from scipy import stats
a, b, r_value, p_value, std_err = stats.linregress(x,y)
#---
#--- Move this to corresponding Driver:train module
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
            ABM.save_pruned_model(self.SAMObject, fname)
    else:
        print "Loading SAMOBject"
        self.SAMObject = ABM.load_pruned_model(fname)
#---














# This is for visualising the mapping of the test face back to the internal memory
if visualise_output: 
    ax = mySAMpy.SAMObject.visualise()
    visualiseInfo=dict()
    visualiseInfo['ax']=ax
    ytmp = mySAMpy.SAMObject.recall(0)
    ytmp = numpy.reshape(ytmp,(mySAMpy.imgHeightNew,mySAMpy.imgWidthNew))
    fig_nn = pb.figure()
    pb.title('Training NN')
    pl_nn = fig_nn.add_subplot(111)
    ax_nn = pl_nn.imshow(ytmp, cmap=plt.cm.Greys_r)
    pb.draw()
    pb.show()
    visualiseInfo['fig_nn']=fig_nn
else:
    visualiseInfo=None

# Read and test images from iCub eyes in real-time

#fig_input = pb.figure()
#subplt_input = fig_input.add_subplot(111)

while( True ):
    try:
        testFace = mySAMpy.readImageFromCamera()
        #subplt_input.imshow(testFace, cmap=plt.cm.Greys_r)
        pp = mySAMpy.testing(testFace, visualiseInfo)
        time.sleep(0.5)
        l = pp.pop(0)
        l.remove()
        pb.draw()
        del l
    except KeyboardInterrupt:
        print 'Interrupted'
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

