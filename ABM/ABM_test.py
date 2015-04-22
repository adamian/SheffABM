# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:00:20 2015

@authors: uriel

"""

# Overall data dir
root_data_dir="/home/uriel/Packages/dataDump/faceImageData"
image_suffix=".ppm"
participant_index=('Luke','Uriel','Michael')
pose_index=('Straight','LR','UD')

save_data=False
pickled_save_data_name="Saved_face_Data"

run_abm=True

import matplotlib.pyplot as plt
import os
import cv2
import numpy
import sys
import pickle
#import yarp

import matplotlib as mp
# Use this backend for when the server updates plots through the X 
mp.use('TkAgg')
import pylab as pb
import GPy
# To display figures once they're called
pb.ion()
default_seed = 123344
import ABM


global imageDataInputPort
global imageModeInputPort
global imageInputBottle

global imgWidth
global imgHeight
global Y
global Ytest
global X
global SAMObject


# Read Face Data
def readFaceData():
    global Y

    if not os.path.exists(root_data_dir):
        print "CANNOT FIND:" + root_data_dir

    #load_file = os.path.join(root_data_dir, image_suffix ) 
    #data_image=cv2.imread(os.path.join(data_directory_list[0],data_file_database[0]['img_fname'][0]))                

    ## Find and build index of available images.......
    data_file_count=numpy.zeros([len(participant_index),len(pose_index)])
    data_file_database={}
    for count_participant, current_participant in enumerate(participant_index):
        data_file_database_part={}
        for count_pose, current_pose in enumerate(pose_index):
            current_data_dir=os.path.join(root_data_dir,current_participant+current_pose)
            data_file_database_p=numpy.empty(0,dtype=[('orig_file_id','i2'),('file_id','i2'),('img_fname','a100')])
            data_image_count=0
            if os.path.exists(current_data_dir):
                for file in os.listdir(current_data_dir):
                    #parts = re.split("[-,\.]", file)
                    fileName, fileExtension = os.path.splitext(file)
                    if fileExtension==image_suffix: # Check for image file
                        file_ttt=numpy.empty(1, dtype=[('orig_file_id','i2'),('file_id','i2'),('img_fname','a100')])
                        file_ttt['orig_file_id'][0]=int(fileName)
                        file_ttt['img_fname'][0]=file
                        file_ttt['file_id'][0]=data_image_count
                        data_file_database_p = numpy.append(data_file_database_p,file_ttt,axis=0)
                        data_image_count += 1
                data_file_database_p=numpy.sort(data_file_database_p,order=['orig_file_id'])  
            data_file_database_part[pose_index[count_pose]]=data_file_database_p
            data_file_count[count_participant,count_pose]=len(data_file_database_p)
        data_file_database[participant_index[count_participant]]=data_file_database_part

    # To access use both dictionaries data_file_database['Luke']['LR']
    # Cutting indexes to smllest number of available files -> Using file count
    min_no_images=int(numpy.min(data_file_count))

    # Load image data into array......
    # Load first image to get sizes....
    data_image=cv2.imread(os.path.join(root_data_dir,participant_index[0]+pose_index[0]+"/"+
        data_file_database[participant_index[0]][pose_index[0]][0][2]))[:,:,(2,1,0)] # Convert BGR to RGB

    # Data size
    print "Found image with dimensions" + str(data_image.shape)
    imgplot = plt.imshow(data_image)#[:,:,(2,1,0)]) # convert BGR to RGB

    # Load all images....
    #Data Dimensions:
    #1. Pixels (e.g. 200x200)
    #2. Images 
    #3. Person
    #4. Movement (Static. up/down. left / right) 
    set_x=int(data_image.shape[0])
    set_y=int(data_image.shape[1])
    no_rgb=int(data_image.shape[2])
    no_pixels=set_x*set_y
    img_data=numpy.zeros([no_pixels, min_no_images, len(participant_index),len(pose_index)])
    img_label_data=numpy.zeros([no_pixels, min_no_images, len(participant_index),len(pose_index)],dtype=int)
    #cv2.imshow("test", data_image)
    #cv2.waitKey(50)              
    for count_pose, current_pose in enumerate(pose_index):
        for count_participant, current_participant in enumerate(participant_index):
            for current_image in range(min_no_images): 
                current_image_path=os.path.join(os.path.join(root_data_dir,participant_index[count_participant]+pose_index[count_pose]+"/"+
                    data_file_database[participant_index[count_participant]][pose_index[count_pose]][current_image][2]))
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
                data_image=cv2.cvtColor(data_image, cv2.COLOR_BGR2GRAY)         
                img_data[:,current_image,count_participant,count_pose] = data_image.flatten()
                # Labelling with participant            
                img_label_data[:,current_image,count_participant,count_pose]=numpy.zeros(no_pixels,dtype=int)+count_participant
    Y=img_data

def prepareFaceData():    
    """--- Now Y has 4 dimensions: 
    1. Pixels
    2. Images
    3. Person
    4. Movement (Static. up/down. left / right)     
    """ 
    global Y
    global X
    K = len(participant_index)   
    # We can do differen scenarios.
    # Y = img_data[:,:,0,1] ; # Load one face, one pose . In this case, set also X=None 
    ttt=Y[:,:,:,1]
    Y=ttt.reshape(ttt.shape[0],K*ttt.shape[1]) 
    Y=Y.T
    N=Y.shape[0]
    L = numpy.zeros((N,1))
    L[0::N/3]=0
    L[N/3:2*N/3:]=1
    L[2*N/3::]=2    
    X=None
    
    #---TEMP
    #Y=Y[0::3,:]
    #----    
    
    Ymean = Y.mean()
    Yn = Y - Ymean
    Ystd = Yn.std()
    Yn /= Ystd
       
    Y = {'Y':Yn,'L':L}

# training
def prepareTraining():
    global SAMObject
    global Y
    global X
    

    
    if X is not None:
        Q = X.shape[1]
    else:
        Q=10

    # Instantiate object
    SAMObject=ABM.LFM()
    if Q > 100:
        kernel = GPy.kern.RBF(Q, ARD=False) + GPy.kern.Bias(Q) + GPy.kern.White(Q)
    else:
        kernel = None

    SAMObject.store(observed=Y, inputs=X, Q=Q, kernel=kernel, num_inducing=40)
    #SAMObject.add_labels(L.argmax(axis=1))

    SAMObject.learn(optimizer='bfgs',max_iters=2, verbose=True)

#    ret = SAMObject.visualise()



# testing
def testingImage():
    global SAMObject
    global Ytest
    
    pred_mean, pred_var = SAMObject.pattern_completion(Ytest)
    print pred_mean
    print pred_var
    # Visualise the predictive point estimates for the test data
    pb.plot(pred_mean[:,0],pred_mean[:,1],'bx')


# create ports
def createPorts():
    global imageDataInputPort
    global imageModeInputPort
    global imageInputBottle

    imageDataInputPort = yarp.Port()
    imageDataInputPort.open("/sam/abm/imageData:i")
#    imageModeInputPort.open("/sam/raw/imageMode:i")

    imageInputBottle = yarp.Bottle()

    return True


def readImagesFromCameras():
    global imgWidth
    global imgHeight
    global imageInputBottle
    global Ytest

    imageInputBottle.clear()
    imageDataInputPort.read(imageInputBottle)

    Ytest = numpy.zeros(shape=(imageInputBottle.size()))

    for i in range(imageInputBottle.size()):
        Ytest[i] = imageInputBottle.get(i).asInt();
    
    Ytest = Ytest[:,None].T
    print Ytest.shape

    print "-----------------"

# initialise Yarp
yarp.Network.init()


imgWidth=200
imgHeight=200


print "Creating ports..."
createPorts()

print "Reading Face Data"
readFaceData()

print "Creating Face scenario"
prepareFaceData()

print "Runnin training..."
prepareTraining()

while(1):
    print "Reading image for testing..."
    readImagesFromCameras()
    print "Testing..."
    testingImage()


print "Images process finish"

