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
import operator
from ABM import ABM


#""""""""""""""""
#Class developed for the implementation of the face recognition task in real-time mode.
#""""""""""""""""

class SAMpy:

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

        if( isYarpRunning == True ):
            yarp.Network.init()
            self.createPorts()
            self.openPorts()
            self.createImageArrays()


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

#""""""""""""""""
#Method to read face data previously collected to be used in the traning phase.
#Here the loaded data is preprocessed to have the correct image size and one face per image.
#Inputs:
#    - root_data_dir: location of face data
#    - participant_inde: array of participants names
#    - pose_index: array of poses from the face data collected
#
#Outputs: None
#""""""""""""""""
    def readFaceData(self, root_data_dir, participant_index, pose_index):
        self.Y
        self.L
        self.participant_index = participant_index

        if not os.path.exists(root_data_dir):
            print "CANNOT FIND:" + root_data_dir
        else:
            print "PATH FOUND"

	    ## Find and build index of available images.......
        data_file_count=numpy.zeros([len(self.participant_index),len(pose_index)])
        data_file_database={}
        for count_participant, current_participant in enumerate(self.participant_index):
            data_file_database_part={}
            for count_pose, current_pose in enumerate(pose_index):
                current_data_dir=os.path.join(root_data_dir,current_participant+current_pose)
                data_file_database_p=numpy.empty(0,dtype=[('orig_file_id','i2'),('file_id','i2'),('img_fname','a100')])
                data_image_count=0
                if os.path.exists(current_data_dir):
                    for file in os.listdir(current_data_dir):
	                    #parts = re.split("[-,\.]", file)
                        fileName, fileExtension = os.path.splitext(file)
                        if fileExtension==self.image_suffix: # Check for image file
                            file_ttt=numpy.empty(1, dtype=[('orig_file_id','i2'),('file_id','i2'),('img_fname','a100')])
                            file_ttt['orig_file_id'][0]=int(fileName)
                            file_ttt['img_fname'][0]=file
                            file_ttt['file_id'][0]=data_image_count
                            data_file_database_p = numpy.append(data_file_database_p,file_ttt,axis=0)
                            data_image_count += 1
                    data_file_database_p=numpy.sort(data_file_database_p,order=['orig_file_id'])  
                data_file_database_part[pose_index[count_pose]]=data_file_database_p
                data_file_count[count_participant,count_pose]=len(data_file_database_p)
            data_file_database[self.participant_index[count_participant]]=data_file_database_part

	    # To access use both dictionaries data_file_database['Luke']['LR']
	    # Cutting indexes to smllest number of available files -> Using file count
        min_no_images=int(numpy.min(data_file_count))

	    # Load image data into array......
	    # Load first image to get sizes....
        data_image=cv2.imread(os.path.join(root_data_dir,self.participant_index[0]+pose_index[0]+"/"+
            data_file_database[self.participant_index[0]][pose_index[0]][0][2]))[:,:,(2,1,0)] # Convert BGR to RGB

	    # Data size
        print "Found minimum number of images:" + str(min_no_images)
        print "Image count:", data_file_count
        print "Found image with dimensions" + str(data_image.shape)
	#    imgplot = plt.imshow(data_image)#[:,:,(2,1,0)]) # convert BGR to RGB

	    # Load all images....
	    #Data Dimensions:
	    #1. Pixels (e.g. 200x200)
	    #2. Images 
	    #3. Person
	    #4. Movement (Static. up/down. left / right) 
        set_x=int(data_image.shape[0])
        set_y=int(data_image.shape[1])
        #no_rgb=int(data_image.shape[2])
        no_pixels=self.imgWidthNew*self.imgHeightNew #set_x*set_y
        img_data=numpy.zeros([no_pixels, min_no_images, len(self.participant_index),len(pose_index)])
        img_label_data=numpy.zeros([no_pixels, min_no_images, len(self.participant_index),len(pose_index)],dtype=int)
	    #cv2.imshow("test", data_image)
	    #cv2.waitKey(50)              
        for count_pose, current_pose in enumerate(pose_index):
            for count_participant, current_participant in enumerate(self.participant_index):
                for current_image in range(min_no_images): 
                    current_image_path=os.path.join(os.path.join(root_data_dir,self.participant_index[count_participant]+pose_index[count_pose]+"/"+
                        data_file_database[self.participant_index[count_participant]][pose_index[count_pose]][current_image][2]))
                    data_image=cv2.imread(current_image_path)
	                # Check image is the same size if not... cut or reject
                    if data_image.shape[0]<set_x or data_image.shape[1]<set_y:
                        print "Image too small... EXITING:"
                        print "Found image with dimensions" + str(data_image.shape)
                        sys.exit(0)
                    if data_image.shape[0]>set_x or data_image.shape[1]>set_y:
                        print "Found image with dimensions" + str(data_image.shape)
                        print "Image too big cutting to: x="+ str(set_x) + " y=" + str(set_y)
                        data_image=data_image[:set_x,:set_y]
                    data_image=cv2.resize(data_image, (self.imgWidthNew, self.imgHeightNew)) # New
                    data_image=cv2.cvtColor(data_image, cv2.COLOR_BGR2GRAY)         
                    img_data[:,current_image,count_participant,count_pose] = data_image.flatten()
	                # Labelling with participant            
                    img_label_data[:,current_image,count_participant,count_pose]=numpy.zeros(no_pixels,dtype=int)+count_participant

        self.Y=img_data
        self.L=img_label_data

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




