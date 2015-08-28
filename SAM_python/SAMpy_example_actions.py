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
mySAMpy = SAMpy_actions(True,inputActionPort="/visionDriver/image:o")
# YARP OFF
#mySAMpy = SAMpy_actions(False,inputActionPort="/visionDriver/image:o")

# Specification of the experiment number
experiment_number = 1004

# Location of face data
#root_data_dir="/home/icub/dataDump/actionData"
#root_data_dir="D:/robotology/SheffABM/actionData"
#root_data_dir=r"//10.0.0.20/dataDump/actionData"
#root_data_dir=r"//10.0.0.20/dataDump/actionDataNew"
root_data_dir=r"//10.0.0.20/dataDump/actionDataUpdated"
#root_data_dir="/home/icub/dataDump/actionDataUpdated"

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
fname = './models/mActions_' + model_type + '_exp' + str(experiment_number) #+ '.pickle'

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

print 'Got here 1'
while (True):
    testAction, testActionZero, actionFormattedTesting, testTime = mySAMpy.readActionFromRobot()
    # Check action found!
    """
    if (testAction.shape[1]!=1):
        
        #if (mySAMpy.plotFlag):
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
     """  
    if (testAction.shape[1]!=1):    
        # Waiting for interaction input from speech code...
        choice = inputInteractionPort.read(False)
        
        if not(choice == None ):
            choiceInt = choice.get(0).asInt()
            print "Choice found: " + str(choiceInt)
            # Find action - push / pull / lift / put down
            if (choiceInt==18):
                # Send data to model
                pp = mySAMpy.testing(actionFormattedTesting, choiceInt, objectFlag, visualiseInfo)
                l = pp.pop()
                l.remove()
                pb.draw()
            # Case - what am I pointing at
            elif (choiceInt==20):
                print "Looking for object!"
                # Get objects location
                # Waiting for interaction input from speech code...
                objectLocation = inputObjectPort.read(True) # non-blocking function
                
                if not(objectLocation == None ): # object reported
                    # Get length of bottle
                    #objectData = numpy.zeros((objectLocation.size(),1),dtype=int)
                    objectCount = int(numpy.floor(objectLocation.size()/4))
                    objectLabel = []
                    # Bottle as multiples of x,y,z name
                    for currentData in range(objectCount):
                        if (currentData==0):
                            objectData = numpy.array(objectLocation.get((currentData*4)).asInt())
                        else:
                            objectData = numpy.hstack((objectData,objectLocation.get((currentData*4)).asInt()))
                        objectData = numpy.hstack((objectData,objectLocation.get((currentData*4)+1).asInt()))
                        objectData = numpy.hstack((objectData,objectLocation.get((currentData*4)+2).asInt()))
                        objectLabel.append(objectLocation.get((currentData*4)+3).asString())
                        
                        
                    print "Objects, count=" + str(objectCount) + " " + objectLabel[0] + " " + objectLabel[1] + " " + objectLabel[2]   
                    # objectLocation = numpy.array([23,45,10,34,10,0]) # object 1 xyz, object 2 xyz etc...
                    # Find which hand moved most
                    # Left  hand
                    movementLeft=numpy.sum(numpy.sum(numpy.abs(numpy.diff(testAction[:,mySAMpy.bodyPartIndex[2]],1,axis=0)),axis=0))
                    # Right  hand
                    movementRight=numpy.sum(numpy.sum(numpy.abs(numpy.diff(testAction[:,mySAMpy.bodyPartIndex[3]],1,axis=0)),axis=0))
                    
                    dist=numpy.zeros(objectCount)
                    print objectData
                    print testAction[-1,:]
                    if (movementLeft>movementRight):
                        handUsed = 'Left'
                        # Find object nearest to moving handin 3D
                        for currentObject in range(objectCount):
                            dist[currentObject]=distance.euclidean(testAction[-1,mySAMpy.bodyPartIndex[2]],objectData[[currentObject*3,(currentObject*3)+1,(currentObject*3)+2]])
                    else:
                        handUsed = 'Right'
                        # Find object nearest to moving hand in 3D
                        for currentObject in range(objectCount):
                            #dist[currentObject]=(numpy.linalg.norm(testAction[-1,mySAMpy.bodyPartIndex[3]]-objectData[[currentObject*3,(currentObject*3)+1,(currentObject*3)+2]]))
                            dist[currentObject]=distance.euclidean(testAction[-1,mySAMpy.bodyPartIndex[3]],objectData[[currentObject*3,(currentObject*3)+1,(currentObject*3)+2]])
                    objectNearest=numpy.argmin(dist,axis=0)
                        
                    print dist
                    print "Hand used = " + handUsed
                    print "Nearest object is: " + str(objectNearest)
                    objectPredictionBottle = yarp.Bottle()        
                    objectPredictionBottle.addString("It is the " + objectLabel[objectNearest]) 
                    mySAMpy.speakStatusPort.write(mySAMpy.speakStatusOutBottle, mySAMpy.speakStatusInBottle)
                    if( mySAMpy.speakStatusInBottle.get(0).asString() == "quiet"):
                        mySAMpy.outputActionPrediction.write(objectPredictionBottle)
                    objectPredictionBottle.clear()  
                else:
                    print "Hand used = " + handUsed
                    print "No objects found!"
              
    
                    
            pb.waitforbuttonpress(0.05)
            
            #mySAMpy.testDebug(mySAMpy,actionFormattedTesting, numpy.zeros(actionFormattedTesting.shape[0],actionFormattedTesting.shape[1]), action_labels, 0)
            
            #testDebug(d.SAMObject, d.Ytest, d.Ltest, d.L)
            # TODO: SPlit actions in loaded files e.g. left / right
        
pb.figure(111)
pb.subplot(numpy.shape(testAction)[1],1,1)                                   
pb.title('Actions with time')
pb.figure(112)
pb.subplot(numpy.shape(testAction)[1],1,1)                                   
pb.title('Actions Zeropad') 


mySAMpy.closePorts()

