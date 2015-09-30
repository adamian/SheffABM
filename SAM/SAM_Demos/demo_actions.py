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
#from SAMDriver_actions import SAMDriver_actions
#import ABM
from SAM.SAM_Drivers import SAMDriver_actions
from scipy.spatial import distance
import pylab as pb
import sys
import pickle
import os
import numpy
import time
import operator
import yarp


yarp.Network.init()
inputInteractionPort = yarp.BufferedPortBottle()
inputObjectPort = yarp.BufferedPortBottle()

inputInteractionPort.open("/sam/actions/interaction:i");
inputObjectPort.open("/sam/actions/objects:i");
choice = yarp.Bottle();
objectLocation = yarp.Bottle();

# Creates a SAMpy object
# YARP ON
mySAMpy = SAMDriver_actions(True,inputActionPort="/visionDriver/image:o")
# YARP OFF
#mySAMpy = SAMDriver_actions(False,inputActionPort="/visionDriver/image:o")

# Specification of the experiment number
experiment_number = 4044#1010

# Location of face data
#root_data_dir="/home/icub/dataDump/actionData"
#root_data_dir="D:/robotology/SheffABM/actionData"
#root_data_dir=r"//10.0.0.20/dataDump/actionData"
#root_data_dir=r"//10.0.0.20/dataDump/actionDataNew"
#root_data_dir=r"//10.0.0.20/dataDump/actionDataUpdated"
root_data_dir="/home/icub/dataDump/actionDataUpdated"

# Image format
#image_suffix=".ppm"

# Based on directories where files are held
# 1. Array of participants to be recognised
participant_index=['Luke','Michael', 'Uriel'] # 'Michael','Luke'
# 2. Poses used during the data collection
hand_index=('left','right')
# 3. actions
action_index=('LR','UD' ,'waving') # Based on directories where files are held

# Sub split training data -> e.g. left right into left and right
# split done taking gradient of greatest movement
# INCREASING GRADIENT FIRST! 0 = top left, so
# left -ve, Right +ve
# up -ve, down +ve 
# Take each action and enter the number of splits
action_splitting_index=[['left','right'],['down','up'],['waving']]

# LB Temp
#action_labels=('Left_LR','Left_UD' ,'Left_waving','Right_LR','Right_UD' ,'Right_waving')

# Use a subset of the data for training
Ntr=500

# Pose selected for training
#pose_selection = 0

# Specification of model type and training parameters
model_type = 'mrd'
model_num_inducing = 35
model_num_iterations = 100 #100
model_init_iterations = 300 #800
fname = '/home/icub/models/mActions_' + model_type + '_exp' + str(experiment_number) #+ '.pickle'
#fname = './models/mActions_' + model_type + '_exp' + str(experiment_number) #+ '.pickle'

# Enable to save the model and visualise GP nearest neighbour matching
save_model=True
visualise_output=True

#action_index = 1;
#hand_index = 2;

objectFlag = True   # True: objects ----- False: hands

# @@@@@@@@@ LOAD DATA AND TRAIN MODEL IF IT DOESNT EXIST @@@@@@@@@@@@@
# Reading face data, preparation of data and training of the model
mySAMpy.readData(root_data_dir,participant_index,hand_index,action_index,action_splitting_index)
mySAMpy.prepareActionData(model_type, Ntr)
#mySAMpy.prepareFaceData(model_type, Ntr, pose_selection)
mySAMpy.training(model_num_inducing, model_num_iterations, model_init_iterations, fname, save_model)

while( not(yarp.Network.isConnected("/speechInteraction/behaviour:o","/sam/actions/interaction:i")) ):
    print "Waiting for connection with behaviour port..."
    pass


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

#fig_input = pb.figure()
#subplt_input = fig_input.add_subplot(111)

# LB @@@@@@@@@@@@@@@@@@@@@@@ REAL TIME DATA SECTION -> get actions from robot
actionCount=0

pb.figure(111)
#pb.ion()
#pb.show()
pb.figure(112)
#pb.ion()
#pb.show()

while (True):

    testAction, testActionZero, actionFormattedTesting, testTime = mySAMpy.readActionFromRobot()
    
    choice = inputInteractionPort.read(True)
                
    # Send data to model
    pp = mySAMpy.testing(actionFormattedTesting, choice, objectFlag, visualiseInfo)
    l = pp.pop()
    l.remove()
    pb.draw()
        
    pb.waitforbuttonpress(0.1)


pb.figure(111)
pb.subplot(numpy.shape(testAction)[1],1,1)                                   
pb.title('Actions with time')
pb.figure(112)
pb.subplot(numpy.shape(testAction)[1],1,1)                                   
pb.title('Actions Zeropad') 


mySAMpy.closePorts()

