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
from SAMpy_actions import SAMpy_actions
import pylab as pb
import sys
import pickle
import os
import numpy
import time
import operator


# Creates a SAMpy object
mySAMpy = SAMpy_actions(True, imgH = 400, imgW = 400, imgHNew = 200, imgWNew = 200,inputActionPort="/visionDriver/image:o")

# Specification of the experiment number
experiment_number = 882

# Location of face data
#root_data_dir="/home/icub/dataDump/actionData"
#root_data_dir="D:/robotology/SheffABM/actionData"
#root_data_dir=r"//10.0.0.20/dataDump/actionData"
root_data_dir=r"//10.0.0.20/dataDump/actionDataNew"
# Image format
#image_suffix=".ppm"
# Array of participants to be recognised
participant_index=('Michael','Luke')#,'uriel')
# Poses used during the data collection
hand_index=('left','right')
action_index=('LR','UD' ,'waving')
# Use a subset of the data for training
Ntr=300

# Pose selected for training
#pose_selection = 0

# Specification of model type and training parameters
model_type = 'mrd'
model_num_inducing = 35
model_num_iterations = 100
model_init_iterations = 800
fname = 'mActions_' + model_type + '_exp' + str(experiment_number) #+ '.pickle'

# Enable to save the model and visualise GP nearest neighbour matching
save_model=True
visualise_output=True

#action_index = 1;
#hand_index = 2;



"""
# @@@@@@@@@ LOAD DATA AND TRAIN MODEL IF IT DOESNT EXIST @@@@@@@@@@@@@
# Reading face data, preparation of data and training of the model
mySAMpy.readData(root_data_dir,participant_index,hand_index,action_index)
mySAMpy.prepareActionData(model_type, Ntr)
#mySAMpy.prepareFaceData(model_type, Ntr, pose_selection)
mySAMpy.training(model_num_inducing, model_num_iterations, model_init_iterations, fname, save_model)


# This is for visualising the mapping of the test face back to the internal memory
if visualise_output: 
    ax = mySAMpy.SAMObject.visualise()
    visualiseInfo=dict()
    visualiseInfo['ax']=ax
    ytmp = mySAMpy.SAMObject.recall(0)
    #ytmp = numpy.reshape(ytmp,(mySAMpy.imgHeightNew,mySAMpy.imgWidthNew))
    fig_nn = pb.figure()
    pb.title('Training NN')
    pl_nn = fig_nn.add_subplot(111)
    #ax_nn = pl_nn.imshow(ytmp, cmap=plt.cm.Greys_r)
    ax_nn=pl_nn.plot(ytmp)    
    pb.draw()
    pb.show()
    visualiseInfo['fig_nn']=fig_nn
else:
    visualiseInfo=None

# Read and test images from iCub eyes in real-time

fig_input = pb.figure()
subplt_input = fig_input.add_subplot(111)
"""

# LB @@@@@@@@@@@@@@@@@@@@@@@ REAL TIME DATA SECTION -> get actions from robot
actionCount=0

pb.figure(111)
pb.ion()
pb.show()
pb.figure(112)
pb.ion()
pb.show()


while (actionCount<5):
    testAction, testActionZero, testTime = mySAMpy.readActionFromRobot()
    
    # Check action found!
    if (testAction.shape[1]!=1):
        pb.figure(111)
        color_rand=numpy.random.rand(3,1)
        for currentBP in range(numpy.shape(testAction)[1]):
            pb.subplot(numpy.shape(testAction)[1],1,currentBP+1)
            pb.hold(True)                                    
            pb.plot(testTime,testAction[:,currentBP],c=color_rand)
        pb.figure(112)
        for currentBP in range(numpy.shape(testAction)[1]):
            pb.subplot(numpy.shape(testAction)[1],1,currentBP+1)
            pb.hold(True)                                    
            pb.plot(testActionZero[:,currentBP],c=color_rand)
        actionCount+=1
        time.sleep(0.05)
        pb.draw()
        
        # Format data for model        
        testActionFormatted=numpy.reshape(testActionZero,(testActionZero.size))
        # Ned to zero mean and format        
        
        
        # Send data to model
        #pp = mySAMpy.testing(testActionFormatted, visualiseInfo)
        
        # TODO: REMOVE EXTRA non-moving data
        # TODO: ADD interpolation to file data loading -> need to match data length e.g. 3s*20Hz = 60 points
        # TODO: SPlit actions in loaded files e.g. left / right
        
pb.figure(111)
pb.subplot(numpy.shape(testAction)[1],1,1)                                   
pb.title('Actions with time')
pb.figure(112)
pb.subplot(numpy.shape(testAction)[1],1,1)                                   
pb.title('Actions Zeropad') 


"""
while( True ):
    try:
        #testFace = mySAMpy.readImageFromCamera()
        #subplt_input.imshow(testFace, cmap=plt.cm.Greys_r)
        #pp = mySAMpy.testing(testFace, visualiseInfo)
        #time.sleep(0.5)
        #l = pp.pop(0)
        #l.remove()
        #pb.draw()
        #del l
    except KeyboardInterrupt:
        print 'Interrupted'
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
"""

mySAMpy.closePorts()