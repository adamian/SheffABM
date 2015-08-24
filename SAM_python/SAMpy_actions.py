#!/usr/bin/python

#""""""""""""""""""""""""""""""""""""""""""""""
#The University of Sheffield
#WYSIWYD Project
#
#SAMpy class for implementation of ABM module
#
#Created July 2015
#
#@authors: Luke Boorman, Uriel Martinez,  Andreas Damianou
#
#""""""""""""""""""""""""""""""""""""""""""""""

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
#from scipy.signal import savgol_filter
from scipy.signal import medfilt
import operator
from ABM import ABM


#""""""""""""""""
#Class developed for the implementation of the action recognition task in real-time mode.
#""""""""""""""""

class SAMpy_actions:

#""""""""""""""""
#Initilization of the SAM class
#Inputs:
#    - isYarprunning: specifies if yarp is used (True) or not(False)
#    - imgH, imgW: original image width and height
#    - imgHNewm imgWNew: width and height values to resize the image
#
#Outputs: None
#""""""""""""""""
    def __init__(self, isYarpRunning = False, imgH = 200, imgW = 200, imgHNew = 200, imgWNew = 200, inputActionPort="/visionDriver/bodyPartPosition:o"):
        
        self.inputActionPort=inputActionPort
        
        self.SAMObject=ABM.LFM()        
        self.imgHeight = imgH
        self.imgWidth = imgW
        self.imgHeightNew = imgHNew
        self.imgWidthNew = imgWNew
        self.image_suffix=".ppm"
        self.plotFlag=False

        self.dataFileName="data.log"
        self.bodyPartNames=['Face','Body','Left Arm','Right Arm','Left and Right Arms']
        self.bodyPartIndex=numpy.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])

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
        self.participant_index = None
        self.cutEnds=True  # cut off first and last values (remove start and end effects e.g. trailling zero)
        self.preProcessDataFlag = True # Zero mean and median filter data        
        self.plotPreProcessedData = True # plot preprocessed data

        # Tags columns from input file that will be processed, e.g smoothed and differentiated
        self.indToProcess=(2,3,4,5,6,7,8,9,10,11,12,13) # index to grab from file for processing
        self.indTim=1

        # ############## Parameters for movement segmentation ################
        self.actionStopTime = 1 # Time in s for splitting each new movement
        self.minActionTime = 0.4 # Minimum length for action in s
        self.minimumMovementThreshold = 5 # Equivalent number of steps or pixels or mm that triggers movement  
        self.maxMovementTime = 3 # Greatest duration of movement in s        
        self.filterWindow=5 # Median Filter window length
        # real time options
        #self.sampleRate = 0 # sample rate calculated from data!
        self.fixedSampleRate = 20  #Hz data will be interpolated up to this sample rate 
        self.minSampleRate = 5 #Hz data rejected is sample rate is below this
        
        # Calcs realtime time vector
        self.minActionSteps=int(self.fixedSampleRate*self.minActionTime)
        self.maxActionSteps=int(self.fixedSampleRate*self.maxMovementTime)
        self.timRT=numpy.arange(0.0,self.maxMovementTime,1.0/self.fixedSampleRate)
        
        # GPy SAM model option
        self.model_num_inducing = 0
        self.model_num_iterations = 0
        self.model_init_iterations = 0
        


        if( isYarpRunning == True ):
            yarp.Network.init()
            self.createPorts()
            self.openPorts()
            self.createImageArrays()


#""""""""""""""""
#Methods to create the ports for reading actions from iCub
#Inputs: None
#Outputs: None
#""""""""""""""""
    def createPorts(self):
        self.actionDataInputPort = yarp.BufferedPortBottle()#yarp.BufferedPortImageRgb()
        self.outputActionPrection = yarp.Port()
        self.speakStatusPort = yarp.RpcClient()
        self.speakStatusOutBottle = yarp.Bottle()
        self.speakStatusInBottle = yarp.Bottle()
        self.imageInputBottle = yarp.Bottle()

#""""""""""""""""
#Method to open the ports.
#Inputs: None
#Outputs: None
#""""""""""""""""
    def openPorts(self):
        print "open ports"
        self.actionDataInputPort.open("/sam/actionData:i");
        self.outputActionPrection.open("/sam/actionPrediction:o")
        self.speakStatusPort.open("/sam/speakStatus:i")
        self.speakStatusOutBottle.addString("stat")

#""""""""""""""""
#Method to close the ports.
#Inputs: None
#Outputs: None
#""""""""""""""""
    def closePorts(self):
        print "open ports"
        self.actionDataInputPort.close();
        self.outputActionPrection.close()
        self.speakStatusPort.close()
        #self.speakStatusOutBottle.addString("stat")
        #print "Waiting for connection with actionDataInputPort..."
        #while( not(yarp.Network.isConnected(self.inputActionPort,"/sam/actionData:i")) ):
        #    print "Waiting for connection with actionDataInputPort..."
        #    pass

#""""""""""""""""
#Method to prepare the arrays to receive the RBG images from yarp
#Inputs: None
#Outputs: None
#""""""""""""""""
    def createImageArrays(self):
        self.imageArray = numpy.zeros((self.imgHeight, self.imgWidth, 3), dtype=numpy.uint8)
        #self.newImage = yarp.ImageRgb()
        self.yarpImage = yarp.ImageRgb()
        self.yarpImage.resize(self.imgWidthNew,self.imgWidthNew)
        self.yarpImage.setExternal(self.imageArray, self.imageArray.shape[1], self.imageArray.shape[0])

    def getColumns(self, inFile, delim="\t", header=True):
        """
        Get columns of data from inFile. The order of the rows is respected
    
        :param inFile: column file separated by delim
        :param header: if True the first line will be considered a header line
        :returns: a tuple of 2 dicts (cols, indexToName). cols dict has keys that 
        are headings in the inFile, and values are a list of all the entries in that
        column. indexToName dict maps column index to names that are used as keys in 
        the cols dict. The names are the same as the headings used in inFile. If
        header is False, then column indices (starting from 0) are used for the 
        heading names (i.e. the keys in the cols dict)
        """
        cols = {}
        indexToName = {}
        for lineNum, line in enumerate(inFile):
            if lineNum == 0:
                headings = line.split(delim)
                i = 0
                for heading in headings:
                    heading = heading.strip()
                    if header:
                        cols[heading] = []
                        indexToName[i] = heading
                    else:
                        # in this case the heading is actually just a cell
                        cols[i] = [heading]
                        indexToName[i] = i
                    i += 1
            else:
                cells = line.split(delim)
                i = 0
                for cell in cells:
                    cell = cell.strip()
                    cols[indexToName[i]] += [cell]
                    i += 1
        # Convert cols to numpy
        # check its rectangular, go through each column
        
        for colsChk in range(len(cols)):
            if (colsChk==0):
                colsRows=len(cols[0])
            else:
                #print "colsRows: " + str(colsRows)
                if (colsRows!=len(cols[colsChk])):
                    print "MATRIX is not rectangular cannot convert to NUMPY array"
                    return numpy.empty((1,1))
        
        dataReturn = numpy.empty([colsRows,len(cols)],dtype=numpy.float)
        
        for colsColumns in range(len(cols)):
            for colsRows in range(colsRows):
                dataReturn[colsRows,colsColumns]=float(cols[colsColumns][colsRows])
        
        if (self.cutEnds):
            #print dataLog.shape
            dataReturn = dataReturn[10:-10,:]
            #print dataLog.shape
                
        
        
        return dataReturn #cols, indexToName

    def preProcessData(self,dataIn,indToProcess,indTim):
        # This will cut and filter data prior to splitting
        # dataIn as data points (ts) by each ts (e.g. face x y z, body x y z....) 
        #zeroMeanData=[2,3,4,5,6,7,8,9,10,11,12,13] # -1=off, index to columns to zero mean -> Should only OPERATE ON FACE AND BODY DATA
        #preProcessData= [2,3,4,5,6,7,8,9,10,11,12,13]# -1 off, index to colums to filt currently medfilt. TODO Changes of less than three values about zeros are removed e.g. [0 0 # # 0 0]    
        
        # Optional plot raw        
        if (self.plotPreProcessedData):   
            #plotInd=(6,7,9,10) # ind to plot from indTo process            
            plotInd=(0,1,2,3,4,5,6,7,8,9,10,11) # ind to plot from indTo process              
            plt.figure(self.test_count+500)
            for currentInd in range(len(plotInd)):
                plt.subplot(len(plotInd),1,currentInd+1)
                plt.hold(True)
                lineRaw, = plt.plot(range(0,dataIn.shape[0]),dataIn[:,indToProcess[plotInd[currentInd]]],'r',label='raw')
        
        dataOut=numpy.empty([dataIn.shape[0],len(indToProcess)],dtype=float)
        dataDiff=numpy.empty([dataIn.shape[0]-1,len(indToProcess)],dtype=float)        
        #dataDiff2nd=numpy.empty([dataIn.shape[0]-2,len(indToProcess)],dtype=float)        
        
        tim=dataIn[:,indTim]
        diffTim=tim[:-1]        
        
        # Check for corrupt data        
        maxVal=numpy.max(dataIn[:,indToProcess])
        print "Max:" + str(maxVal)
        
        
        
        
        ## Main processing here....
        for indCount,currentXYZ in enumerate(indToProcess):
            # Zero mean
            dataOut[:,indCount]=dataIn[:,currentXYZ]-numpy.mean(dataIn[:,currentXYZ])   
        #for indCount in range(len(indToProcess)):    
            # Median filt -> window 5            
            dataOut[:,indCount]=medfilt(dataOut[:,indCount],self.filterWindow)
            # Diff data to find action movement
            dataDiff[:,indCount]=numpy.diff(dataOut[:,indCount])                
            # Median filt -> window 5            
            dataDiff[:,indCount]=medfilt(dataDiff[:,indCount],self.filterWindow)
            # Diff data to find action movement
            #dataDiff2nd[:,indCount]=numpy.diff(dataDiff[:,indCount])                
            # Median filt -> window self.filterWindow            
            #dataDiff2nd[:,indCount]=medfilt(dataDiff2nd[:,indCount],self.filterWindow)
            
        # Optional now overlay processed data
        if (self.plotPreProcessedData):
            for currentInd in range(len(plotInd)):
                plt.subplot(len(plotInd),1,currentInd+1)
                lineProc, = plt.plot(range(0,dataOut.shape[0]),dataOut[:,plotInd[currentInd]],'b',label='processed')
                lineDiff, = plt.plot(numpy.arange(dataDiff.shape[0],dtype=float)+0.5,dataDiff[:,plotInd[currentInd]],'g',label='proc diff')
                #lineDiff2nd, = plt.plot(numpy.arange(dataDiff2nd.shape[0],dtype=float)+0.5,dataDiff2nd[:,plotInd[currentInd]],'m',label='proc 2nd diff')
                plt.title(str(plotInd[currentInd]))
            plt.legend(handles=[lineRaw, lineProc,lineDiff])#,lineDiff2nd])

            
        return dataOut, dataDiff, tim, diffTim

    def findMovements(self, data, dataDiff, tim, ind2Check):
        # Segment Body part data
        # Version 2 Luke August 2015
        # Looks for maximum movement from any body region and then sections the data by finding periods of no movement
        # self.actionStopTime = 2 # Time in s for splitting each movement
        # self.minimumMovementThreshold = 3 # Equivalent number of pixels that triggers movement  
        # self.minActionTime = shortest time for action to be accepted
        # self.maxMovementTime = 5 # greatest movement period in s
        # Calc steps for movement off (e.g. still time)        
        
        sampleRate=1/numpy.mean(numpy.diff(tim))
        minActionSteps=int(sampleRate*self.minActionTime)
        newActionSteps=int(sampleRate*self.actionStopTime)
        maxActionSteps=int(sampleRate*self.maxMovementTime)
        
        print "Min action steps:" + str(minActionSteps) 
        # Find maximum value across each time point (change in pos for all body parts and x,y,x)
        dataMax = numpy.max(numpy.abs(dataDiff[:,ind2Check]),axis=1)
        # Threshold data to find where the body part change is greater than the threshold
        #dataMoving = numpy.where(dataMax > self.minimumMovementThreshold)    
        dataMoving = numpy.where(dataMax > self.minimumMovementThreshold, dataMax, 0)
        # Now find consecutive zeros
        actionData = []
        timeData = []
        actionCount = 0
        actionFound = False
        zeroCount = 0
        # Separate actions depending on non-movement periods -> uses self.minActionTIme
        for currentInd in range(numpy.size(dataMoving)):
            # Check for movement (non-zero)
            if (dataMoving[currentInd]!=0):
                # New action, init whole thing
                if (not actionFound):
                    actionData.append(numpy.array(data[currentInd]))
                    timeData.append(numpy.array(tim[currentInd]))
                    actionFound = True
                else:
                # if action already found add values onto action
                    actionData[actionCount]=numpy.vstack((actionData[actionCount],data[currentInd]))
                    timeData[actionCount]=numpy.vstack((timeData[actionCount],tim[currentInd]))
                    #actionFound = True 
                # reset consecutive zeroCount
                zeroCount = 0
            # Case of zero... no movement
            else:
                # First check a new action has started, but we have not found too many zeros
                if (actionFound and zeroCount<newActionSteps):
                    zeroCount += 1 # increment zerocount
                    # Add data to action 
                    actionData[actionCount]=numpy.vstack((actionData[actionCount],data[currentInd]))
                    timeData[actionCount]=numpy.vstack((timeData[actionCount],tim[currentInd]))
                # Case for new action as enough zeros have been reached
                elif (actionFound and zeroCount>=newActionSteps):
                    # End of action as enough non-movement detected
                    actionFound = False
                    # Remove extra non-movement
                    actionData[actionCount]=actionData[actionCount][:-zeroCount,:]
                    timeData[actionCount]=timeData[actionCount][:-zeroCount]
                    
                    # Remove actions if too short
                    if (actionData[actionCount].shape[0]<int(minActionSteps)):
                        print "Removing action:" + str(actionCount) + " with length: " + str(actionData[actionCount].shape[0])  + " as too short"
                        actionData.pop(actionCount)
                        timeData.pop(actionCount)
                    else:
                        # Cut to largest movement allowed self.maxMovementTime
                        if (actionData[actionCount].shape[0]>maxActionSteps): 
                            actionData[actionCount]=actionData[actionCount][:maxActionSteps,:]
                            timeData[actionCount]=timeData[actionCount][:maxActionSteps]
                        # Report and increment
                        print "Added action no: " + str(actionCount) + " with length: " + str(actionData[actionCount].shape[0])  
                        actionCount += 1
            #print "Action count:" + str(actionCount)
            #print "Zero count:" + str(zeroCount)            
            #print "Action found:" + str(actionFound)                
            #print " "               
        #ttt=1       
        # Nothing found check
        if (actionCount==0 and not actionFound):
            print "Nothing found"
        # One action found - repeat above process that was missed out!
        elif (actionCount==0 and actionFound):
            print "Single action found"
            # Remove actions if too short
            if (actionData[actionCount].shape[0]<int(minActionSteps)):
                print "Removing action:" + str(actionCount) + " with length: " + str(actionData[actionCount].shape[0])  + " as too short"
                actionData.pop(actionCount)
                timeData.pop(actionCount)
            else:
                # Cut to largest movement allowed self.maxMovementTime
                if (actionData[actionCount].shape[0]>maxActionSteps): 
                    actionData[actionCount]=actionData[actionCount][:maxActionSteps,:]
                    timeData[actionCount]=timeData[actionCount][:maxActionSteps]
                print "Added action no: " + str(actionCount) + " with length: " + str(actionData[actionCount].shape[0])       
        
        # Expand all actions to full maxMovementTime e.g. 5s
        # Zero pad after action upto maxMovementTime
        # Loop through all actions
        actionDataZeroPad=[]
        for currentAction in range(len(actionData)):
            if (actionData[currentAction].shape[0]<maxActionSteps):
                zeroLength=maxActionSteps-actionData[currentAction].shape[0]
                #print actionData[currentAction].shape
                #print str(zeroLength) + " " + str(actionData[currentAction].shape[1])
                actionDataZeroPad.append(numpy.vstack((actionData[currentAction],numpy.zeros((zeroLength,actionData[currentAction].shape[1])))))
            elif (actionData[currentAction].shape[0]>maxActionSteps):
                actionDataZeroPad.append(actionData[currentAction][:maxActionSteps,:])
                print "WARNING DATA TOO LONG: " +  str(actionData[currentAction].shape[0])
            else:
                actionDataZeroPad.append(actionData[currentAction])

        return actionData, timeData, actionDataZeroPad
                
        
    
#""""""""""""""""
#Method to read action data previously collected to be used in the traning phase.
#Here the loaded data is preprocessed to have the correct image size and one action per image.
#Inputs:
#    - root_data_dir: location of action data
#    - participant_inde: array of participants names
#    - pose_index: array of poses from the action data collected
#
#Outputs: None
#""""""""""""""""(root_data_dir,participant_index,hand_index,action_index)
    def readData(self, root_data_dir,participant_index,hand_index,action_index):
        self.Y
        self.L
        self.action_index = action_index
        self.participant_index = participant_index
        self.hand_index = hand_index

        
        # Max action time -> 5s will be used to generate fixed array for each action
        self.maxActionTime = 5 # maximum action time in s        
        # Check if root dir exists
        if not os.path.exists(root_data_dir):
            print "CANNOT FIND: " + root_data_dir
        else:
            print "PATH FOUND: " + root_data_dir

        #data_file_database={}
        #dataFile = file(root_data_dir+"/data.log")

        # Generate directory names -> structure will likely change
        # 1. action index
        self.test_count=1  
        actionsbyType=[]
        actionsbyTypeLabel=[]
        
        # NEED TO CHECK FOR sucessful finding of actions from each sub index for building the final array.....        
        
        minActionsOverall = 10000 # find least number of actions for every context....
        minActionDuration = 100000 # find the shortest length for all actions
        actionTypesFoundCount=0 # count successful actions found
        
        for actionInd in range(len(self.action_index)):
            # 2. Participant index
            actionsbyParticipant=[]
            actionsbyParticipantLabel=[]
            minParticipant = 10000 # find min number of participants
            for partInd in range(len(self.participant_index)):
                # 3. hand index
                actionsbyHand=[]
                actionsbyHandLabel=[]
                minHands = 10000 # find min number of hands
                for handInd in range(len(self.hand_index)):
                    # Check if root dir exists
                    dir_string=os.path.join(root_data_dir,(self.participant_index[partInd] + "_" + self.hand_index[handInd]\
                    + "_" + self.action_index[actionInd]))
                    if not os.path.exists(dir_string):
                        print "CANNOT FIND: " + dir_string
                    else:
                        print "PATH FOUND: " + dir_string
                        # Check file in dir....
                        dataFilePath=os.path.join(dir_string,self.dataFileName)
                        if not os.path.exists(dataFilePath):
                            print "CANNOT FIND: " + dataFilePath
                        else:
                            print "PATH FOUND: " + dataFilePath
                            # Open file
                            dataFile = open(dataFilePath, 'rb')
                            logData = self.getColumns(dataFile, " ", False)
                            if (logData.size==0):
                                print "Log Data empty skipping"
                            else:
                                dataFile.close();
                                print "Sample rate: " + str(1/numpy.mean(numpy.diff(logData[:,1])))
                                if (self.preProcessDataFlag):
                                    dataProc, dataDiff, tim, diffTim = self.preProcessData(logData,self.indToProcess,self.indTim)
                                
                                self.test_count+=1
                                
                                # Segment body parts                                
                                #actionIndex, actionData, timeData=self.findMovements(dataProc, dataDiff, tim,range(dataDiff.shape[1]))
                                actionData, timeData, actionDataZeroPad=self.findMovements(dataProc, dataDiff, tim,range(dataDiff.shape[1]))
                                
                                # Check action found (only false if empty)
                                if actionData:
                                    # Sort counters here -> find minimum number of actions -> used later for cutting
                                    if len(actionDataZeroPad) < minActionsOverall:
                                        minActionsOverall = len(actionDataZeroPad)
                                    # Plot action output
                                    for currentAction in range(len(actionData)):
                                        # Generate and use same color throughout
                                        color_rand=numpy.random.rand(3,1)
                                        plt.figure(888+self.test_count)
                                        for currentBP in range(numpy.shape(actionData[currentAction])[1]):
                                            plt.subplot(numpy.shape(actionData[currentAction])[1],1,currentBP+1)
                                            plt.hold(True)                                    
                                            plt.plot(timeData[currentAction],actionData[currentAction][:,currentBP],c=color_rand)
                                        plt.subplot(numpy.shape(actionData[currentAction])[1],1,1)                                    
                                        plt.title(dataFilePath)
                                        
                                        plt.figure(777+self.test_count)
                                        for currentBP in range(numpy.shape(actionData[currentAction])[1]):
                                            plt.subplot(numpy.shape(actionData[currentAction])[1],1,currentBP+1)
                                            plt.hold(True)                                    
                                            plt.plot(actionDataZeroPad[currentAction][:,currentBP],c=color_rand)
                                        plt.subplot(numpy.shape(actionData[currentAction])[1],1,1)                                    
                                        plt.title(dataFilePath)                                        
                                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                    # ~~~~~~~~~~~~~ FORMATTING THE DATA FOR THE SAM ~~~~~~~~~~~~~~~
                                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
                                    # Make overall array
                                    # A. Data
                                    # Data as:
                                    # >> 1. Timeseries
                                    # >> 2. Body part and xyz
                                    # >> 3. Repeat actions
                                    # 4. Hand used
                                    # 5. Person
                                    # 6. Action Type
                                    actionsCombined=numpy.zeros((actionDataZeroPad[0].shape[0],actionDataZeroPad[0].shape[1],len(actionDataZeroPad)))
                                    for currentAction in range(len(actionDataZeroPad)):
                                        actionsCombined[:,:,currentAction]=actionDataZeroPad[currentAction]
                                        # Check for shortest action length
                                        if actionDataZeroPad[currentAction].shape[0]<minActionDuration:
                                            minActionDuration=actionDataZeroPad[currentAction].shape[0]
                                    # B. Labels                                
                                    # Initally built around the action name (based on action index)
                                    actionLabel=numpy.zeros((actionDataZeroPad[0].shape[0],actionDataZeroPad[0].shape[1],len(actionDataZeroPad)))+actionInd
                    # Combine arrays across actions hands
                    actionsbyHand.append(actionsCombined) 
                    actionsbyHandLabel.append(actionLabel)
                # Combine arrays across actions participant
                actionsbyParticipant.append(actionsbyHand) 
                actionsbyParticipantLabel.append(actionsbyHandLabel)
                # Check here for hand count should be 2 ->
                if len(actionsbyHand)<minHands:
                    minHands=len(actionsbyHand)
            # Combine arrays across actions participant
            actionsbyType.append(actionsbyParticipant)
            actionsbyTypeLabel.append(actionsbyParticipantLabel)
            if len(actionsbyParticipant)<minParticipant:
                minParticipant=len(actionsbyParticipant)
            if actionsbyType: # false if empty
                actionTypesFoundCount+=1
                
        # Print outputs
        print str(actionTypesFoundCount) + " different actions found"     
        print "Shortest action length found is " + str(minActionDuration) + " steps"        
        print "Found " + str(minActionsOverall) + " actions for every participant and hand used"
        print "Found minimum of data recorded from " + str(minHands) + " hands for all pariticipants involved"         
        print "Found " + str(minParticipant) + " participants completeing all actions"
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~ COMBINE AND CUT THE DATA FOR THE SAM ~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build the nump matrix
        # Data as:
        # 1. Timeseries         0
        # 2. Body part and xyz  1
        # 3. Repeat actions     2
        # 4. Hand used          3
        # 5. Person             4
        # 6. Action Type        5
        allActions=numpy.zeros((minActionDuration,len(self.indToProcess),minActionsOverall,minHands,minParticipant,actionTypesFoundCount))
        allLabels=numpy.zeros((minActionDuration,len(self.indToProcess),minActionsOverall,minHands,minParticipant,actionTypesFoundCount))        
        # 6. Action type
        for actionInd in range(actionTypesFoundCount):
            # 5. Participant index
            for partInd in range(minParticipant):
                # 4. hand index
                for handInd in range(minHands):
                    allActions[:,:,:,handInd,partInd,actionInd]=actionsbyType[actionInd][partInd][handInd][:minActionDuration,:len(self.indToProcess),:minActionsOverall]
                    allLabels[:,:,:,handInd,partInd,actionInd]=actionsbyTypeLabel[actionInd][partInd][handInd][:minActionDuration,:len(self.indToProcess),:minActionsOverall]
                            #strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
                
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~ ALTERNATIVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~ 1. Combines actions Type to include hand e.g. waving right , ud left
        # ~~~~~~~~~~~~~~ 2. flatten timeseries and body part xyz and participant
        # ~~~~~~~~~~~~~~ COMBINE AND CUT THE DATA FOR THE SAM ~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build the nump matrix
        # Data as:
        # 1. Timeseries (0) + Body part and xyz (1)
        # 2. Repeat actions (2) + person (4)
        # 3. Action Type (5) and Hand used (3)
        combinedActions=numpy.zeros((minActionDuration*len(self.indToProcess),minActionsOverall*minParticipant,minHands*actionTypesFoundCount))
        combinedLabels=numpy.zeros((minActionDuration*len(self.indToProcess),minActionsOverall*minParticipant,minHands*actionTypesFoundCount))        
        # Reshape timeseries, body part xyz
        temp1=numpy.reshape(numpy.transpose(allActions,(2,3,4,5,1,0)),(minActionsOverall,minHands,minParticipant,actionTypesFoundCount,minActionDuration*len(self.indToProcess)))      
        temp2=numpy.reshape(numpy.transpose(allLabels,(2,3,4,5,1,0)),(minActionsOverall,minHands,minParticipant,actionTypesFoundCount,minActionDuration*len(self.indToProcess)))      
        # !!!!!!UPDATE LABELS TO MARK BOTH Left and right hands for combined hands and action type 
        for handInd in range(minHands):        
            temp2[:,handInd,:,:]=temp2[:,handInd,:,:]+(handInd*100) # added differential index: original label = left hand, original label + 100 = right hand
        # Combine Repeat actions and participant
        temp1=numpy.reshape(numpy.transpose(temp1,(4,1,3,0,2)),(minActionDuration*len(self.indToProcess),minHands,actionTypesFoundCount,minActionsOverall*minParticipant))
        temp2=numpy.reshape(numpy.transpose(temp2,(4,1,3,0,2)),(minActionDuration*len(self.indToProcess),minHands,actionTypesFoundCount,minActionsOverall*minParticipant))
        # Combine hands and action type    
        combinedActions=numpy.reshape(numpy.transpose(temp1,(0,3,1,2)),(minActionDuration*len(self.indToProcess),minActionsOverall*minParticipant,minHands*actionTypesFoundCount))
        combinedLabels=numpy.reshape(numpy.transpose(temp2,(0,3,1,2)),(minActionDuration*len(self.indToProcess),minActionsOverall*minParticipant,minHands*actionTypesFoundCount))
   
        """
        combinedActions=numpy.zeros((minActionDuration*len(self.indToProcess)*minParticipant,minActionsOverall,minHands*actionTypesFoundCount))
        combinedLabels=numpy.zeros((minActionDuration*len(self.indToProcess)*minParticipant,minActionsOverall,minHands*actionTypesFoundCount))        
        # Reshape timeseries, body part xyz and persons
        temp1=numpy.reshape(numpy.transpose(allActions,(2,3,5,4,1,0)),(minActionsOverall,minHands,actionTypesFoundCount,minActionDuration*len(self.indToProcess)*minParticipant))      
        temp2=numpy.reshape(numpy.transpose(allLabels,(2,3,5,4,1,0)),(minActionsOverall,minHands,actionTypesFoundCount,minActionDuration*len(self.indToProcess)*minParticipant))      
        # Combined hands and action type -> !!!!!!UPDATE LABELS TO MARK BOTH!!!!!!!!
        combinedActions=numpy.reshape(numpy.transpose(temp1,(3,0,1,2)),(temp1.shape[3],temp1.shape[0],temp1.shape[1]*temp1.shape[2]))
        for handInd in range(minHands):        
            temp2[:,handInd,:,:]=temp2[:,handInd,:,:]+(handInd*100)
        combinedLabels=numpy.reshape(numpy.transpose(temp2,(3,0,1,2)),(temp2.shape[3],temp2.shape[0],temp2.shape[1]*temp2.shape[2]))
        """     
        
        """
        # 6. Action type
        for actionInd in range(actionTypesFoundCount):
            # 5. Participant index
            for partInd in range(minParticipant):
                # 4. hand index
                for handInd in range(minHands):
                    combinedActions[:,:,:,handInd,partInd,actionInd]=actionsbyType[actionInd][partInd][handInd][:minActionDuration,:len(self.indToProcess),:minActionsOverall]
                    combinedLabels[:,:,:,handInd,partInd,actionInd]=actionsbyTypeLabel[actionInd][partInd][handInd][:minActionDuration,:len(self.indToProcess),:minActionsOverall]
        """
        
        plt.show()#block=True)

        self.Y=combinedActions
        self.L=combinedLabels

#        for count_participant, current_participant in enumerate(self.participant_index):
#            data_file_database_part={}
#            for count_pose, current_pose in enumerate(pose_index):
#                current_data_dir=os.path.join(root_data_dir,current_participant+current_pose)
#                data_file_database_p=numpy.empty(0,dtype=[('orig_file_id','i2'),('file_id','i2'),('img_fname','a100')])
#                data_image_count=0
#                if os.path.exists(current_data_dir):
#                    for file in os.listdir(current_data_dir):
#	                    #parts = re.split("[-,\.]", file)
#                        fileName, fileExtension = os.path.splitext(file)
#                        if fileExtension==self.image_suffix: # Check for image file
#                            file_ttt=numpy.empty(1, dtype=[('orig_file_id','i2'),('file_id','i2'),('img_fname','a100')])
#                            file_ttt['orig_file_id'][0]=int(fileName)
#                            file_ttt['img_fname'][0]=file
#                            file_ttt['file_id'][0]=data_image_count
#                            data_file_database_p = numpy.append(data_file_database_p,file_ttt,axis=0)
#                            data_image_count += 1
#                    data_file_database_p=numpy.sort(data_file_database_p,order=['orig_file_id'])  
#                data_file_database_part[pose_index[count_pose]]=data_file_database_p
#                data_file_count[count_participant,count_pose]=len(data_file_database_p)
#            data_file_database[self.participant_index[count_participant]]=data_file_database_part
#
#	    # To access use both dictionaries data_file_database['Luke']['LR']
#	    # Cutting indexes to smllest number of available files -> Using file count
#        min_no_images=int(numpy.min(data_file_count))
#
#	    # Load image data into array......
#	    # Load first image to get sizes....
#        data_image=cv2.imread(os.path.join(root_data_dir,self.participant_index[0]+pose_index[0]+"/"+
#            data_file_database[self.participant_index[0]][pose_index[0]][0][2]))[:,:,(2,1,0)] # Convert BGR to RGB
#
#	    # Data size
#        print "Found minimum number of images:" + str(min_no_images)
#        print "Image count:", data_file_count
#        print "Found image with dimensions" + str(data_image.shape)
#	#    imgplot = plt.imshow(data_image)#[:,:,(2,1,0)]) # convert BGR to RGB
#
#	    # Load all images....
#	    #Data Dimensions:
#	    #1. Pixels (e.g. 200x200)
#	    #2. Images 
#	    #3. Person
#	    #4. Movement (Static. up/down. left / right) 
#        set_x=int(data_image.shape[0])
#        set_y=int(data_image.shape[1])
#        #no_rgb=int(data_image.shape[2])
#        no_pixels=self.imgWidthNew*self.imgHeightNew #set_x*set_y
#        img_data=numpy.zeros([no_pixels, min_no_images, len(self.participant_index),len(pose_index)])
#        img_label_data=numpy.zeros([no_pixels, min_no_images, len(self.participant_index),len(pose_index)],dtype=int)
#	    #cv2.imshow("test", data_image)
#	    #cv2.waitKey(50)              
#        for count_pose, current_pose in enumerate(pose_index):
#            for count_participant, current_participant in enumerate(self.participant_index):
#                for current_image in range(min_no_images): 
#                    current_image_path=os.path.join(os.path.join(root_data_dir,self.participant_index[count_participant]+pose_index[count_pose]+"/"+
#                        data_file_database[self.participant_index[count_participant]][pose_index[count_pose]][current_image][2]))
#                    data_image=cv2.imread(current_image_path)
#	                # Check image is the same size if not... cut or reject
#                    if data_image.shape[0]<set_x or data_image.shape[1]<set_y:
#                        print "Image too small... EXITING:"
#                        print "Found image with dimensions" + str(data_image.shape)
#                        sys.exit(0)
#                    if data_image.shape[0]>set_x or data_image.shape[1]>set_y:
#                        print "Found image with dimensions" + str(data_image.shape)
#                        print "Image too big cutting to: x="+ str(set_x) + " y=" + str(set_y)
#                        data_image=data_image[:set_x,:set_y]
#                    data_image=cv2.resize(data_image, (self.imgWidthNew, self.imgHeightNew)) # New
#                    data_image=cv2.cvtColor(data_image, cv2.COLOR_BGR2GRAY) 
#                    # Data is flattened into single vector (inside matrix of all images) -> (from images)        
#                    img_data[:,current_image,count_participant,count_pose] = data_image.flatten()
#	                # Labelling with participant            
#                    img_label_data[:,current_image,count_participant,count_pose]=numpy.zeros(no_pixels,dtype=int)+count_participant
#
#        self.Y=img_data
#        self.L=img_label_data


#""""""""""""""""
#Method to process some important features from the action data required for the classification model such as mean and variance.
#Inputs:
#    - model: type of model used for the ABM object
#    - Ntr: Number of training samples
#    - pose_selection: participants pose used for training of the ABM object
#
#Outputs: None
#""""""""""""""""
    def prepareActionData(self, model='mrd', Ntr = 50):    
        #""--- Now Y has 3 dimensions: 
        #1. time points (body part x,y,z and participant)
        #2. repeated actions 
        #3. Action type (waving. up/down. left / right)     
        #
        #We can prepare the action data using different scenarios about what to be perceived.
        #In each scenario, a different LFM is used. We have:
        #- gp scenario, where we regress from images to labels (inputs are images, outputs are labels)
        #- bgplvm scenario, where we are only perceiving images as outputs (no inputs, no labels)
        #- mrd scenario, where we have no inputs, but images and labels form two different views of the output space.
        #
        #The store module of the LFM automatically sees the structure of the assumed perceived data and 
        #decides on the LFM backbone to be used.
        #
        #! Important: The global variable Y is changed in this section. From the multi-dim. matrix of all
        #modalities, it turns into the training matrix of action data and then again it turns into the 
        #dictionary used for the LFM.
        #---""" 
        #OLD
        #0. Pixels = ts 0
        #1. Images = repeat actions 1
        #2. Person = actiontype 2
        #3. Movement (Static. up/down. left / right)
    
    
    
        # Take all poses if pose selection ==-1
        """  if pose_selection == -1:
        ttt=numpy.transpose(self.Y,(0,1,3,2))
        ttt=ttt.reshape((ttt.shape[0],ttt.shape[1]*ttt.shape[2],ttt.shape[3])) 
        else:
            ttt=self.Y[:,:,:,pose_selection]
        """    
        ttt=numpy.transpose(self.Y,(0,2,1))
        self.Y=ttt.reshape(ttt.shape[0],ttt.shape[2]*ttt.shape[1]) 
        self.Y=self.Y.T
        #N=self.Y.shape[0]
        """
        if pose_selection == -1:
            ttt=numpy.transpose(self.L,(0,1,3,2))
            ttt=ttt.reshape((ttt.shape[0],ttt.shape[1]*ttt.shape[2],ttt.shape[3]))
        else:
    		ttt=self.L[:,:,:,pose_selection]
        """ 
        
        ttt=numpy.transpose(self.L,(0,2,1))
        self.L=ttt.reshape(ttt.shape[0],ttt.shape[2]*ttt.shape[1]) 
        self.L=self.L.T
        self.L=self.L[:,:1]

        # 

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
#Method to train, store and load the learned model to be use for the action recognition task
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
            # If data are associated with labels (e.g. action identities), associate them with the event collection
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

#""""""""""""""""
#Method to test the learned model with actions read from the iCub eyes in real-time
#Inputs:
#    - testaction: image from iCub eyes to be recognized
#    - visualiseInfo: enable/disable the result from the testing process
#
#Outputs:
#    - pp: the axis of the latent space backwards mapping
#""""""""""""""""
    def testing(self, testAction, visualiseInfo=None):
        # Returns the predictive mean, the predictive variance and the axis (pp) of the latent space backwards mapping.            
        mm,vv,pp=self.SAMObject.pattern_completion(testAction, visualiseInfo=visualiseInfo)
                
        # find nearest neighbour of mm and SAMObject.model.X
        dists = numpy.zeros((self.SAMObject.model.X.shape[0],1))

        actionPredictionBottle = yarp.Bottle()
    
        for j in range(dists.shape[0]):
            dists[j,:] = distance.euclidean(self.SAMObject.model.X.mean[j,:], mm[0].values)
        nn, min_value = min(enumerate(dists), key=operator.itemgetter(1))
        if self.SAMObject.type == 'mrd':
            print "With " + str(vv.mean()) +" prob. error the new image is " + self.participant_index[int(self.SAMObject.model.bgplvms[1].Y[nn,:])]
            textStringOut=self.participant_index[int(self.SAMObject.model.bgplvms[1].Y[nn,:])]

        elif self.SAMObject.type == 'bgplvm':
            print "With " + str(vv.mean()) +" prob. error the new image is " + self.participant_index[int(self.L[nn,:])]
            textStringOut=self.participant_index[int(self.L[nn,:])]
        if (vv.mean()<0.00012):
            choice=numpy.random.randint(4)
            if (choice==0):
                 actionPredictionBottle.addString("Hello " + textStringOut)
            elif(choice==1):
                 actionPredictionBottle.addString("I am watching you " + textStringOut)
            elif(choice==2):
                 actionPredictionBottle.addString(textStringOut + " could you move a little you are blocking my view of the outside")
            else:
                 actionPredictionBottle.addString(textStringOut + " will you be my friend")                  
            # Otherwise ask for updated name... (TODO: add in updated name)
        else:
            actionPredictionBottle.addString("I think you are " + textStringOut + " but I am not sure, please confirm?")        
     
        # Plot the training NN of the test image (the NN is found in the INTERNAl, compressed (latent) memory space!!!)
        if visualiseInfo is not None:
            fig_nn = visualiseInfo['fig_nn']
            fig_nn = pb.figure(11)
            pb.title('Training NN')
            fig_nn.clf()
            pl_nn = fig_nn.add_subplot(111)
            pl_nn.imshow(numpy.reshape(self.SAMObject.recall(nn),(self.imgHeightNew, self.imgWidthNew)), cmap=plt.cm.Greys_r)
            pb.title('Training NN')
            pb.show()
            pb.draw()
            pb.waitforbuttonpress(0.1)
            
        self.speakStatusPort.write(self.speakStatusOutBottle, self.speakStatusInBottle)

        if( self.speakStatusInBottle.get(0).asString() == "quiet"):
            self.outputActionPrection.write(actionPredictionBottle)

        actionPredictionBottle.clear()

        return pp

#""""""""""""""""
#Method to read images from the iCub eyes used for the Action recognition task
#Inputs: None
#Outputs:
#    - imageFlatten_testing: image from iCub eyes in row format for testing by the ABM model
#""""""""""""""""
    def readImageFromCamera(self):
        while(True):
            try:
                self.newImage = self.actionDataInputPort.read(False)
            except KeyboardInterrupt:
                print 'Interrupted'
                try:
                    sys.exit(0)
                except SystemExit:
                    os._exit(0)

            if not( self.newImage == None ):
                self.yarpImage.copy(self.newImage)

                imageArrayOld=cv2.resize(self.imageArray,(self.imgHeightNew,self.imgWidthNew))
                imageArrayGray=cv2.cvtColor(imageArrayOld, cv2.COLOR_BGR2GRAY)

                plt.figure(10)
                plt.title('Image received')
                plt.imshow(imageArrayGray,cmap=plt.cm.Greys_r)
                plt.show()
                plt.waitforbuttonpress(0.1)

                imageFlatten_testing = imageArrayGray.flatten()
                imageFlatten_testing = imageFlatten_testing - self.Ymean
                imageFlatten_testing = imageFlatten_testing/self.Ystd#

                imageFlatten_testing = imageFlatten_testing[:,None].T
                
                break

        return imageFlatten_testing
        
#""""""""""""""""
#Method to read action data from the iCub used for the action recognition task
#Inputs: None
#Outputs:
#    - imageFlatten_testing: image from iCub eyes in row format for testing by the ABM model
# Detect action and send out detected actions......
# Loops here to find next action....        
#""""""""""""""""
    def readActionFromRobot(self):
        #self.sampleRate = 0 # sample rate calculated from data!
        #self.fixedSampleRate = 20  #Hz data will be interpolated up to this sample rate 
        #self.minSampleRate = 8 #Hz data rejected is sample rate is below this
        # Init variables 
        dataCount = 0
        
        zeroCount = 0
        actionFound = False
        
        actionData = []
        timeData = []
        

        
        while(True):
            
            
            try:
                newData = yarp.Bottle
                newData = self.actionDataInputPort.read(True) # read values from yarp port
                currentTime = time.time() # time stamp data as it arrives
 
            except KeyboardInterrupt:
                print 'Interrupted'
                try:
                    sys.exit(0)
                except SystemExit:
                    os._exit(0)

            if not(newData == None ):
                #self.yarpImage.copy(self.newImage)
                                
                # Get length of bottle
                data = numpy.zeros((newData.size(),1),dtype=int)
                for currentData in range(newData.size()):                
                    data[currentData] = newData.get(currentData).asInt()
                # Add to data store
                if (dataCount == 0):
                    dataStoreRT=data
                    baseTime = currentTime # baseline time to subtract
                    timeStoreRT = numpy.array([0]) # init time store
                    dataCount+=1
                else:
                    #print data
                    # Check appropiate sample rate!
                    if ((1/numpy.diff(timeStoreRT[-2:]))<self.minSampleRate):
                        print "CLEARING DATA: Sample rate too low: " + str(1/numpy.diff(timeStoreRT[-2:]))
                        actionFound = False
                        zeroCount = 0                        
                        dataCount = 0
                    else:
                        dataCount+=1
                        dataStoreRT=numpy.hstack((dataStoreRT,data))
                        timeStoreRT=numpy.hstack((timeStoreRT,currentTime-baseTime))
                        
                # Wait for at least 3 x filter window length data points to arrive before startin analysis
                if (dataCount>(2*self.filterWindow)):
                    # Initial sample rate from data and action detection times
                    #if (self.sampleRate == 0):
                    self.sampleRate = 1/numpy.mean(numpy.diff(timeStoreRT[-self.filterWindow:-1])) 
                    newActionSteps=int(self.sampleRate*self.actionStopTime) 
                    #else:
                    #    print "Real-time sampleRate = " + str(1/numpy.mean(numpy.diff(timeStoreRT[-self.filterWindow:-1]))) + "vs fixed rate " + str(self.sampleRate)    
                    # This runs in real time so we cannot use the normal file loading
                    # Apply median filter to data...
                    # 1. Preprocess data -> median filter and take first derivative then medin filter
                    # Iterate median filter over data 
                    dataMed=numpy.zeros((dataStoreRT.shape[0],self.filterWindow+1))
                    for currentWindow in range(self.filterWindow+1):
                        dataMed[:,-(currentWindow+1)]=numpy.median(dataStoreRT[:,-(self.filterWindow+currentWindow+1):-(currentWindow+1)],axis=1)
                    # Differentiate it
                    dataDiff=numpy.diff(dataMed,1,axis=1)
                    dataDiffMed=numpy.median(dataDiff,axis=1)
                
                    #2. Find action
                    # Find maximum action across all body parts x,y,z
                    dataMax=numpy.max(numpy.abs(dataDiffMed))
                    
                    # Separate actions depending on non-movement periods -> uses self.minActionTIme
                    # Check movement is above threshold
                    if (dataMax>self.minimumMovementThreshold):
                        #print "Action detected!!!!!!!!!!!!!! " #+ str(actionCount)
                        #actionCount+=1
                        if (not actionFound):
                            actionData=numpy.array(dataMed[:,-1])
                            timeData=numpy.array(timeStoreRT[-1])
                            actionFound = True
                        else:
                        # if action already found add values onto action
                            actionData=numpy.vstack((actionData,dataMed[:,-1]))
                            timeData=numpy.vstack((timeData,timeStoreRT[-1]))
                        # reset consecutive zeroCount as nonzero found
                        zeroCount = 0
                    else: # case of not enough movement
                        # First check a new action has started, but we have not found too many zeros
                        if (actionFound and zeroCount<newActionSteps):
                            zeroCount += 1 # increment zerocount
                            # Add data to action 
                            actionData=numpy.vstack((actionData,dataMed[:,-1]))
                            timeData=numpy.vstack((timeData,timeStoreRT[-1]))
                        # Case for new action as enough zeros have been reached
                        elif (actionFound and zeroCount>=newActionSteps):
                            # End of action as enough non-movement detected
                            actionFound = False
                            # Remove extra non-movement
                            actionDataTemp=actionData[:-zeroCount,:]
                            timeDataTemp=timeData[:-zeroCount]
                            
                            zeroCount = 0
                            
                            # Process before return!
                            # 1. Zero time to start
                            timeDataTemp=timeDataTemp-timeDataTemp[0]
                            # 2. Zero start of action                                
                            actionDataTemp=actionDataTemp-numpy.tile(actionDataTemp[0,:],(actionDataTemp.shape[0],1))                                                      
                            # Linearly interpolate data to 20Hz                                
                            timeData=numpy.arange(0,timeData[-1],1.0/self.fixedSampleRate)
                            actionData=numpy.zeros((timeData.shape[0],actionDataTemp.shape[1]))
                            
                            for currentBP in range(actionData.shape[1]):                            
                                actionData[:,currentBP]=numpy.interp(timeData,timeDataTemp[:,0],actionDataTemp[:,currentBP])
                            
                            # Remove actions if too short
                            if (actionData.shape[0]<int(self.minActionSteps)):
                                print "Removing action with length: " + str(actionData.shape[0])  + " as too short"
                                                                
                                #actionData.pop(actionCount)
                                #timeData.pop(actionCount)
                            else:

                                
                                
                                # Cut to largest movement allowed self.maxMovementTime
                                if (actionData.shape[0]>self.maxActionSteps): 
                                    actionData=actionData[:self.maxActionSteps,:]
                                    timeData=timeData[:self.maxActionSteps]
                                # Report and increment
                                print "Found action with length: " + str(actionData.shape[0])
                                
                                # 3. Zero pad data to max time
                                # Expand all actions to full maxMovementTime e.g. 5s
                                # Zero pad after action upto maxMovementTime
                                if (actionData.shape[0]<self.maxActionSteps):
                                    zeroLength=self.maxActionSteps-actionData.shape[0]
                                    #print actionData[currentAction].shape
                                    #print str(zeroLength) + " " + str(actionData[currentAction].shape[1])
                                    
                                    actionDataZeroPad=numpy.vstack((actionData,numpy.zeros((zeroLength,actionData.shape[1]))))
                                elif (actionData.shape[0]>self.maxActionSteps):
                                    actionDataZeroPad=actionData[:self.maxActionSteps,:]
                                    print "WARNING DATA TOO LONG: " +  str(actionData.shape[0])
                                else:
                                    actionDataZeroPad=actionData

                                return  actionData, actionDataZeroPad, timeData                              
                                #actionCount += 1
                    print "Zeros found: " + str(zeroCount)
                """        
                ## LUKE TEMP ADDITION
                    # Check for movement (non-zero)
                        if (dataMoving[currentInd]!=0):
                            # New action, init whole thing
                            if (not actionFound):
                                actionData.append(numpy.array(data[currentInd]))
                                timeData.append(numpy.array(tim[currentInd]))
                                actionFound = True
                            else:
                            # if action already found add values onto action
                                actionData[actionCount]=numpy.vstack((actionData[actionCount],data[currentInd]))
                                timeData[actionCount]=numpy.vstack((timeData[actionCount],tim[currentInd]))
                                #actionFound = True 
                            # reset consecutive zeroCount
                            zeroCount = 0
                        # Case of zero... no movement
                        else:
                            # First check a new action has started, but we have not found too many zeros
                            if (actionFound and zeroCount<newActionSteps):
                                zeroCount += 1 # increment zerocount
                                # Add data to action 
                                actionData[actionCount]=numpy.vstack((actionData[actionCount],data[currentInd]))
                                timeData[actionCount]=numpy.vstack((timeData[actionCount],tim[currentInd]))
                            # Case for new action as enough zeros have been reached
                            elif (actionFound and zeroCount>=newActionSteps):
                                # End of action as enough non-movement detected
                                actionFound = False
                                # Remove extra non-movement
                                actionData[actionCount]=actionData[actionCount][:-zeroCount,:]
                                timeData[actionCount]=timeData[actionCount][:-zeroCount]
                                
                                # Remove actions if too short
                                if (actionData[actionCount].shape[0]<int(self.minActionSteps)):
                                    print "Removing action:" + str(actionCount) + " with length: " + str(actionData[actionCount].shape[0])  + " as too short"
                                    actionData.pop(actionCount)
                                    timeData.pop(actionCount)
                                else:
                                    # Cut to largest movement allowed self.maxMovementTime
                                    if (actionData[actionCount].shape[0]>self.maxActionSteps): 
                                        actionData[actionCount]=actionData[actionCount][:self.maxActionSteps,:]
                                        timeData[actionCount]=timeData[actionCount][:self.maxActionSteps]
                                    # Report and increment
                                    print "Added action no: " + str(actionCount) + " with length: " + str(actionData[actionCount].shape[0])  
                                    actionCount += 1
                        # Expand all actions to full maxMovementTime e.g. 5s
                        # Zero pad after action upto maxMovementTime
                        # Loop through all actions
                        actionDataZeroPad=[]
                        for currentAction in range(len(actionData)):
                            if (actionData[currentAction].shape[0]<self.maxActionSteps):
                                zeroLength=self.maxActionSteps-actionData[currentAction].shape[0]
                                #print actionData[currentAction].shape
                                #print str(zeroLength) + " " + str(actionData[currentAction].shape[1])
                                actionDataZeroPad.append(numpy.vstack((actionData[currentAction],numpy.zeros((zeroLength,actionData[currentAction].shape[1])))))
                            elif (actionData[currentAction].shape[0]>self.maxActionSteps):
                                actionDataZeroPad.append(actionData[currentAction][:self.maxActionSteps,:])
                                print "WARNING DATA TOO LONG: " +  str(actionData[currentAction].shape[0])
                            else:
                                actionDataZeroPad.append(actionData[currentAction])                        
                        
                ## END OF ADDITION
                """        
                # Reject if data size gets too big!
                if (dataCount==10000):
                    print "WARNING!!!!!!! No actions found found for long period exiting"
                    return 0,0,0
       
  
                
                #if (data[0]==1000):
                #    break
        return 0,0,0
"""
                imageArrayOld=cv2.resize(self.imageArray,(self.imgHeightNew,self.imgWidthNew))
                imageArrayGray=cv2.cvtColor(imageArrayOld, cv2.COLOR_BGR2GRAY)

                plt.figure(10)
                plt.title('Image received')
                plt.imshow(imageArrayGray,cmap=plt.cm.Greys_r)
                plt.show()
                plt.waitforbuttonpress(0.1)

                imageFlatten_testing = imageArrayGray.flatten()
                imageFlatten_testing = imageFlatten_testing - self.Ymean
                imageFlatten_testing = imageFlatten_testing/self.Ystd#

                imageFlatten_testing = imageFlatten_testing[:,None].T
                
                #break
"""
    #return imageFlatten_testing
        
   #def smooth(x,window_len=11,window='hanning'):
"""smooth the data using a window with requested size.
        
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal 
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.
        
        input:
            x: the input signal 
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.
    
        output:
            the smoothed signal
            
        example:
    
        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)
        
        see also: 
        
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter
     
        TODO: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
"""
"""
        if x.ndim != 1:
            raise ValueError, "smooth only accepts 1 dimension arrays."
    
        if x.size < window_len:
            raise ValueError, "Input vector needs to be bigger than window size."
    
    
        if window_len<3:
            return x
    
    
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    
    
        s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w=numpy.ones(window_len,'d')
        else:
            w=eval('numpy.'+window+'(window_len)')
    
        y=numpy.convolve(w/w.sum(),s,mode='valid')
        return y
"""