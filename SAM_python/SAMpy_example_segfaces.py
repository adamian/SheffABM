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
mySAMpy = SAMpy(True, imgH = 400, imgW = 400, imgHNew = 200, imgWNew = 200,inputImagePort="/visionDriver/image:o")

# Specification of the experiment number
experiment_number = 11

# Location of face data
root_data_dir="/home/icub/dataDump/faceImageData_11062015"
# Image format
image_suffix=".ppm"
# Array of participants to be recognised
participant_index=('Luke','Michael','Adriel','Emma')
# Poses used during the data collection
pose_index=['Seg']
# Use a subset of the data for training
Ntr=300

# Pose selected for training
pose_selection = 0

# Specification of model type and training parameters
model_type = 'mrd'
model_num_inducing = 35
model_num_iterations = 100
model_init_iterations = 800
fname = 'm_' + model_type + '_exp' + str(experiment_number) #+ '.pickle'

# Enable to save the model and visualise GP nearest neighbour matching
save_model=True
visualise_output=True

# Reading face data, preparation of data and training of the model
mySAMpy.readFaceData(root_data_dir, participant_index, pose_index)
mySAMpy.prepareFaceData(model_type, Ntr, pose_selection)
mySAMpy.training(model_num_inducing, model_num_iterations, model_init_iterations, fname, save_model)


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

