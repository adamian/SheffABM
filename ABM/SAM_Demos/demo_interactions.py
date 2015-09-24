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
from ABM.SAM_Drivers import SAMDriver_interaction
import pylab as pb
import sys
import pickle
import os
import numpy
import time
import operator
import yarp


# Creates and opens ports for interaction with speech module
yarp.Network.init()
inputInteractionPort = yarp.BufferedPortBottle()
inputInteractionPort.open("/sam/face/interaction:i");
choice = yarp.Bottle();

# Creates a SAMpy object
mySAMpy = SAMDriver_interaction(True, imgH = 400, imgW = 400, imgHNew = 200, imgWNew = 200,inputImagePort="/visionDriver/image:o")

# Specification of the experiment number
experiment_number = 1007#42

# Location of face data
#root_data_dir="/home/icub/dataDump/faceImageData_11062015"
root_data_dir="/home/icub/dataDump/actionFilm"

# Image format
image_suffix=".ppm"
# Array of participants to be recognised
#participant_index=('Uriel','Andreas','Daniel')#=('Luke','Michael','Adriel','Emma','Uriel','Daniel','Andreas')
#participant_index=('Andreas','Uriel','Tony','Daniel')
participant_index=('Andreas','Jordi','Gregoire','Uriel') #participant_index=('Michael','Uriel','Tony', 'Luke')

# Poses used during the data collection
pose_index=['Seg']
# Use a subset of the data for training
Ntr=300

# Pose selected for training
pose_selection = 0

# Specification of model type and training parameters
model_type = 'mrd'
model_num_inducing = 30
model_num_iterations = 150
model_init_iterations = 400
fname = '/home/icub/models/' + 'mActions_' + model_type + '_exp' + str(experiment_number) #+ '.pickle'

# Enable to save the model and visualise GP nearest neighbour matching
save_model=True
visualise_output=True

# Reading face data, preparation of data and training of the model
mySAMpy.readData(root_data_dir, participant_index, pose_index)
mySAMpy.prepareData(model_type, Ntr, pose_selection)
mySAMpy.training(model_num_inducing, model_num_iterations, model_init_iterations, fname, save_model)

while( not(yarp.Network.isConnected("/speechInteraction/behaviour:o","/sam/face/interaction:i")) ):
    print "Waiting for connection with behaviour port..."
    pass


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
    pass

    try:        
        choice = inputInteractionPort.read(True)
        testFace = mySAMpy.readImageFromCamera()
        pp = mySAMpy.testing(testFace, choice, visualiseInfo)
        #time.sleep(0.5)
        l = pp.pop()
        l.remove()
        pb.draw()
        pb.waitforbuttonpress(0.1)
        #del l
    except KeyboardInterrupt:
        print 'Interrupted'
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

