#!/usr/bin/python

#""""""""""""""""""""""""""""""""""""""""""""""
#The University of Sheffield
#WYSIWYD Project
#
#SAMpy class for implementation of ABM module
#
#Created on 26 May 2015
#
#@authors: Uriel Martinez, Luke Boorman, Andreas Damianou
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
from scipy.signal import savgol_filter
import operator
from ABM import ABM


#""""""""""""""""
#Class developed for the implementation of the face recognition task in real-time mode.
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
    def __init__(self, isYarpRunning = False, imgH = 200, imgW = 200, imgHNew = 200, imgWNew = 200, inputImagePort="/visionDriver/image:o"):
        
        self.inputImagePort=inputImagePort
        
        self.SAMObject=ABM.LFM()        
        self.imgHeight = imgH
        self.imgWidth = imgW
        self.imgHeightNew = imgHNew
        self.imgWidthNew = imgWNew
        self.image_suffix=".ppm"

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

        self.model_num_inducing = 0
        self.model_num_iterations = 0
        self.model_init_iterations = 0

#        if( isYarpRunning == True ):
#            yarp.Network.init()
#            self.createPorts()
#            self.openPorts()
#            self.createImageArrays()


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
        self.imageDataInputPort.open("/sam/imageData:i");
        self.outputFacePrection.open("/sam/facePrediction:o")
        self.speakStatusPort.open("/sam/speakStatus:i")
        self.speakStatusOutBottle.addString("stat")

        #print "Waiting for connection with imageDataInputPort..."
        while( not(yarp.Network.isConnected(self.inputImagePort,"/sam/imageData:i")) ):
            print "Waiting for connection with imageDataInputPort..."
            pass

#""""""""""""""""
#Method to prepare the arrays to receive the RBG images from yarp
#Inputs: None
#Outputs: None
#""""""""""""""""
    def createImageArrays(self):
        self.imageArray = numpy.zeros((self.imgHeight, self.imgWidth, 3), dtype=numpy.uint8)
        self.newImage = yarp.ImageRgb()
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
                
        return cols, indexToName


    def splitBodyPartMovements(self, dataLog, bodyPartIndex):

        # Get data read from datalog
        # bodyPartIndex = body part index
        # e.g.
        # bodyPartIndex = 1 # Face pos x,y,z
        # bodyPartIndex = 2 # Body pos x,y,z
        # bodyPartIndex = 3 # Left Arm pos x,y,z
        # bodyPartIndex = 4 # Right Arm x,y,z                
        
        # Data log files is 14 cols
        #1 some sort of index...
        #2 time (gmtime)
        #3 to 14 head, body, left, right arms
        
        denoiseFlag=True # This will remove movements less than 4 steps and remove single movements (1 non-zero movement in vector)        
        smoothFlag=True # Smooth data if required
        
        samplesFound = False
        counterZeros = 0 #numpy.zeros((1,3)); # define zero test x,y,z
        tempData = numpy.empty((1,3),dtype=float)
        timeVectorTemp = numpy.empty((1,1),dtype=float)
        dataBlocks = [] # has to be list as variable size data
        timeVector = [] # has to be list as variable size data
        
#        print numpy.shape(dataLog)

        # Items to Separate by action (sectioned by zero periods (in x and y)..),  from log file....
        if (bodyPartIndex==1): # face
            dataInd=numpy.array([2,3,4])
        elif(bodyPartIndex==2): # body
            dataInd=numpy.array([5,6,7])
        elif(bodyPartIndex==3): # body
            dataInd=numpy.array([8,9,10])
        elif(bodyPartIndex==4): # body
            dataInd=numpy.array([11,12,13])
        
        # To record very first data block... set to relative time from very first block
        firstDataBlock = True
        timeBaseline = 0.0
        # Loop through whole of datalog
        for i in range(len(dataLog[dataInd[0]])):
            # Checking for x,y,z zeros ######################################
            #print "WARNING Z zero check switched off until we get the data" 
            if( float(dataLog[dataInd[0]][i]) == 0.0 and float(dataLog[dataInd[1]][i]) == 0.0): #LB DISABLED FOR NOW AS z always ==1and dataLog[dataInd[2]][i] == 0.0 ):
                counterZeros = counterZeros + 1;
                # Checking for three (x,y,z) zeros in a row ######################################
                if( counterZeros >= 3 and samplesFound == True):
                    # Take last zero value (for output vector) LB removed as isnt real!
                    # tempData=numpy.vstack([tempData,[float(dataLog[dataInd[0]][i]), float(dataLog[dataInd[1]][i]), float(dataLog[dataInd[2]][i])]]);
                    # Dump data to output vector
                    dataBlocks.append(tempData)
                    # Get time point LB removed as extra point isnt real!
                    #timeVectorTemp=numpy.vstack([timeVectorTemp,[float(dataLog[1][i])]]) 
                    # Dump data to output vector
                    timeVector.append(timeVectorTemp)
                    # Reset temp data for next block
                    tempData = numpy.empty((1,3),dtype=float)
                    # Reset timeVectorTemp
                    timeVectorTemp=numpy.empty((1,1),dtype=float)
                    # Reset counter
                    counterZeros = 0
                    # Reset samples found
                    samplesFound = False
                #print counterZeros        
            else: # There is some movement so record it  
                if (samplesFound == False): # case of first value
                    if (firstDataBlock ==  True):
                        timeBaseline=float(dataLog[1][i-1])
                        firstDataBlock = False
                    # Init array with previous (zero) values
                    if (i>1): # ignore very first value from log file 
                        tempData=numpy.array([float(dataLog[dataInd[0]][i-1]), float(dataLog[dataInd[1]][i-1]), float(dataLog[dataInd[2]][i-1])]);
                        # Get previous time point
                        timeVectorTemp=numpy.array([float(dataLog[1][i-1])-timeBaseline]) # Baseline time removed (from very first block)
                        samplesFound = True
                # add in latest non-zero value
                tempData=numpy.vstack([tempData,[float(dataLog[dataInd[0]][i]), float(dataLog[dataInd[1]][i]), float(dataLog[dataInd[2]][i])]]);
                # Get time point
                timeVectorTemp=numpy.vstack([timeVectorTemp,float(dataLog[1][i])-timeBaseline]) # Baseline time removed (from very first block)
            #print float(dataLog[dataInd[0]][i])
            #print float(dataLog[dataInd[1]][i])
            #print float(dataLog[dataInd[2]][i])
        
        ############### De-noising data
        dataBlocksTemp=dataBlocks        
        #timeVectorTemp=timeVector
        ### AGAIN WARNING NOT TESTING Z as always 1 -> needs implementing
        if (denoiseFlag):
            # Reverse counter for popping out to keep correct ind
            for currentAction in range(len(dataBlocksTemp)-1,-1,-1):
            # 1. Check values are above 3
                # print dataBlocksTemp[currentAction][:,:2]            
                if (numpy.max(numpy.abs(dataBlocksTemp[currentAction]))<=3):
                    blockRM=dataBlocks.pop(currentAction)
                    timeVector.pop(currentAction) # also remove from time vector
                    print "Removing block: " + str(blockRM) + " as too small change"
            # 2. Check if single non-zero vale
                elif (numpy.size(numpy.nonzero(numpy.sum(numpy.abs(dataBlocksTemp[currentAction][:,:2]),axis=1)))<=3): # 3 for x,y,z
                    blockRM=dataBlocks.pop(currentAction)
                    timeVector.pop(currentAction) # also remove from time vector
                    print "Removing block: " + str(blockRM) + " as one non-zero value"                
        # Optional smoothing of data
        if (smoothFlag):
            for currentAction in range(len(dataBlocks)):
                print dataBlocks[currentAction][:,0]
                dataBlocks[currentAction][:,0]=savgol_filter(dataBlocks[currentAction][:,0],5,3)
                dataBlocks[currentAction][:,1]=savgol_filter(dataBlocks[currentAction][:,1],5,3)
                dataBlocks[currentAction][:,2]=savgol_filter(dataBlocks[currentAction][:,2],5,3)

            


        
        return dataBlocks, timeVector # left Data, rightData

#""""""""""""""""
#Method to read face data previously collected to be used in the traning phase.
#Here the loaded data is preprocessed to have the correct image size and one face per image.
#Inputs:
#    - root_data_dir: location of face data
#    - participant_inde: array of participants names
#    - pose_index: array of poses from the face data collected
#
#Outputs: None
#""""""""""""""""(root_data_dir,participant_index,hand_index,action_index)
    def readData(self, root_data_dir,participant_index,hand_index,action_index):
        self.Y
        self.L
        self.action_index = action_index
        self.participant_index = participant_index
        self.hand_index = hand_index
        self.dataFileName="data.log"
        self.bodyPartNames=['Face','Body','Left Arm','Right Arm']
        
        # Max action time -> 5s will be used to generate fixed array for each action
        self.maxActionTime = 5 # maximum action time in s        

        dataLogLeftArm =[]
        # Check if root dir exists
        if not os.path.exists(root_data_dir):
            print "CANNOT FIND: " + root_data_dir
        else:
            print "PATH FOUND: " + root_data_dir

        #data_file_database={}
        #dataFile = file(root_data_dir+"/data.log")

        # Generate directory names -> structure will likely change
        # 1. Participant index
        plot_count=1
        for partInd in range(len(self.participant_index)):
            # 2. hand index            
            for handInd in range(len(self.hand_index)):
                # 3. action index
                for actionInd in range(len(self.action_index)):
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
                            cols, indexToName = self.getColumns(dataFile, " ", False)
                            dataFile.close();
                            #rows = numpy.size(cols[2][0:]);
#                            dataLog = numpy.zeros((len(cols)-2, rows))
#                            for i in range(len(cols)-2):
#                                dataLog[i] = cols #cols[i+2][0:]
#                                print dataLog[i] 
                    
                            dataLogAllBody = []
                            timeLogAllBody = []
                            partNameAllIndex=[]
                            # LB COMMENTED HERE AS Body and Face DO NOT HAVE ANY RUNS OF ZEROS...
                            # FACE                           
#                            dataLogFace, timeVecFace = self.splitBodyPartMovements(cols,1) # get face data
#                            dataLogAllBody.append(dataLogFace)
#                            timeLogAllBody.append(timeVecFace)
#                            partNameAllIndex.append(0)
                            # BODY                            
#                            dataLogBody, timeVecBody  = self.splitBodyPartMovements(cols,2) # get Body data
#                            dataLogAllBody.append(dataLogBody)
#                            timeLogAllBody.append(timeVecBody)
#                            partNameAllIndex.append(1)
                            # LEFT ARM                           
                            dataLogLeftArm, timeVecLeftArm  = self.splitBodyPartMovements(cols,3) # get Left arm data
                            dataLogAllBody.append(dataLogLeftArm)
                            timeLogAllBody.append(timeVecLeftArm)
                            partNameAllIndex.append(2)
                            # RIGHT ARM
                            dataLogRightArm, timeVecRightArm  = self.splitBodyPartMovements(cols,4) # get Right Arm data
                            dataLogAllBody.append(dataLogRightArm)
                            timeLogAllBody.append(timeVecRightArm)
                            partNameAllIndex.append(3)
                            
                            figAll = plt.figure(plot_count)
                            plt.hold(True)
                            figAll.canvas.set_window_title(self.participant_index[partInd] + " " + self.hand_index[handInd] + " " + self.action_index[actionInd])

                            figNorm = plt.figure(plot_count+10)
                            plt.hold(True)
                            figNorm.canvas.set_window_title(self.participant_index[partInd] + " " + self.hand_index[handInd] + " " + self.action_index[actionInd])

                            for bodyPart in range(len(dataLogAllBody)):
                                print "Found " + str(len(dataLogAllBody[bodyPart])) + " actions"
                                for currentAction in range(len(dataLogAllBody[bodyPart])):
                                    #  Plt data in time....
                                    plt.figure(plot_count)
                                    plt.subplot(len(dataLogAllBody),2,bodyPart+1)
                                    plt_y_data=dataLogAllBody[bodyPart][currentAction]
                                    plt_x_data=numpy.transpose(timeLogAllBody[bodyPart][currentAction][:,0])
                                    # Use same color throughout                                    
                                    color_rand=numpy.random.rand(3,1)
                                    plt.subplot(len(dataLogAllBody),2,((bodyPart+1)*2)-1)
                                    plt.title("Movement in " + self.bodyPartNames[partNameAllIndex[bodyPart]] + " x")
                                    plt.plot(plt_x_data, plt_y_data[:,0], c=color_rand)
                                    plt.subplot(len(dataLogAllBody),2,(bodyPart+1)*2)
                                    plt.title("Movement in " + self.bodyPartNames[partNameAllIndex[bodyPart]] + " y")
                                    plt.plot(plt_x_data, plt_y_data[:,1], c=color_rand)                                    
                                    
                                    # Plt data with zero time (all overlaid)
                                    plt.figure(plot_count+10)
                                    plt_y_data=dataLogAllBody[bodyPart][currentAction]
                                    plt_x_data=numpy.transpose(timeLogAllBody[bodyPart][currentAction][:,0])-timeLogAllBody[bodyPart][currentAction][0]
                                    plt.subplot(len(dataLogAllBody),2,((bodyPart+1)*2)-1)
                                    plt.title("Time norm, move in " + self.bodyPartNames[partNameAllIndex[bodyPart]] + " x")
                                    plt.plot(plt_x_data, plt_y_data[:,0], c=color_rand)
                                    plt.subplot(len(dataLogAllBody),2,(bodyPart+1)*2)
                                    plt.plot(plt_x_data, plt_y_data[:,1], c=color_rand)
                                    plt.title("Time norm, move in " + self.bodyPartNames[partNameAllIndex[bodyPart]] + " y")                                    
                                    
            plot_count+=1

            #plt.wait
                            #strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
        plt.show()#block=True)


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
#Method to process some important features from the face data required for the classification model such as mean and variance.
#Inputs:
#    - model: type of model used for the ABM object
#    - Ntr: Number of training samples
#    - pose_selection: participants pose used for training of the ABM object
#
#Outputs: None
#""""""""""""""""
    def prepareFaceData(self, model='mrd', Ntr = 50, pose_selection = 0):    
        #""--- Now Y has 4 dimensions: 
        #1. Pixels
        #2. Images
        #3. Person
        #4. Movement (Static. up/down. left / right)     
        #
        #We can prepare the face data using different scenarios about what to be perceived.
        #In each scenario, a different LFM is used. We have:
        #- gp scenario, where we regress from images to labels (inputs are images, outputs are labels)
        #- bgplvm scenario, where we are only perceiving images as outputs (no inputs, no labels)
        #- mrd scenario, where we have no inputs, but images and labels form two different views of the output space.
        #
        #The store module of the LFM automatically sees the structure of the assumed perceived data and 
        #decides on the LFM backbone to be used.
        #
        #! Important: The global variable Y is changed in this section. From the multi-dim. matrix of all
        #modalities, it turns into the training matrix of image data and then again it turns into the 
        #dictionary used for the LFM.
        #---""" 

    	# Take all poses if pose selection ==-1
        if pose_selection == -1:
            ttt=numpy.transpose(self.Y,(0,1,3,2))
            ttt=ttt.reshape((ttt.shape[0],ttt.shape[1]*ttt.shape[2],ttt.shape[3])) 
        else:
    		ttt=self.Y[:,:,:,pose_selection]
        ttt=numpy.transpose(ttt,(0,2,1))
        self.Y=ttt.reshape(ttt.shape[0],ttt.shape[2]*ttt.shape[1]) 
        self.Y=self.Y.T
        #N=self.Y.shape[0]

        if pose_selection == -1:
            ttt=numpy.transpose(self.L,(0,1,3,2))
            ttt=ttt.reshape((ttt.shape[0],ttt.shape[1]*ttt.shape[2],ttt.shape[3]))
        else:
    		ttt=self.L[:,:,:,pose_selection]
        ttt=numpy.transpose(ttt,(0,2,1))
        self.L=ttt.reshape(ttt.shape[0],ttt.shape[2]*ttt.shape[1]) 
        self.L=self.L.T
        self.L=self.L[:,:1]

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
                ABM.save_pruned_model(self.SAMObject, fname)
        else:
	        print "Loading SAMOBject"
	        self.SAMObject = ABM.load_pruned_model(fname)

#""""""""""""""""
#Method to test the learned model with faces read from the iCub eyes in real-time
#Inputs:
#    - testFace: image from iCub eyes to be recognized
#    - visualiseInfo: enable/disable the result from the testing process
#
#Outputs:
#    - pp: the axis of the latent space backwards mapping
#""""""""""""""""
    def testing(self, testFace, visualiseInfo=None):
        # Returns the predictive mean, the predictive variance and the axis (pp) of the latent space backwards mapping.            
        mm,vv,pp=self.SAMObject.pattern_completion(testFace, visualiseInfo=visualiseInfo)
                
        # find nearest neighbour of mm and SAMObject.model.X
        dists = numpy.zeros((self.SAMObject.model.X.shape[0],1))

        facePredictionBottle = yarp.Bottle()
    
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
                 facePredictionBottle.addString("Hello " + textStringOut)
            elif(choice==1):
                 facePredictionBottle.addString("I am watching you " + textStringOut)
            elif(choice==2):
                 facePredictionBottle.addString(textStringOut + " could you move a little you are blocking my view of the outside")
            else:
                 facePredictionBottle.addString(textStringOut + " will you be my friend")                  
            # Otherwise ask for updated name... (TODO: add in updated name)
        else:
            facePredictionBottle.addString("I think you are " + textStringOut + " but I am not sure, please confirm?")        
     
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
            self.outputFacePrection.write(facePredictionBottle)

        facePredictionBottle.clear()

        return pp

#""""""""""""""""
#Method to read images from the iCub eyes used for the face recognition task
#Inputs: None
#Outputs:
#    - imageFlatten_testing: image from iCub eyes in row format for testing by the ABM model
#""""""""""""""""
    def readImageFromCamera(self):
        while(True):
            try:
                self.newImage = self.imageDataInputPort.read(False)
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

    def smooth(x,window_len=11,window='hanning'):
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
