# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:00:20 2015

@authors: uriel, luke, andreas

"""

#------------------- CONFIGURATION --------------
yarp_running = False
# Overall data dir
root_data_dir="/home/icub/dataDump/faceImageData_1_05_2015"
image_suffix=".ppm"
participant_index=('Luke','Uriel','Michael','Andreas')
pose_index=('Straight','LR','Natural') # 'UD')
Ntr=50 # Use a subset of the data for training (and leave the rest for testing)

save_data=False
pickled_save_data_name="Saved_face_Data"

run_abm=True

pose_selection = 1 # Select a pose to train and test.... e.g. LR = 1

# Globals for model
global model_type
global model_num_inducing
global model_num_iterations
model_type = 'bgplvm'
model_num_inducing = 40
model_num_iterations = 300
#---------------------------------------------------------------------------


import matplotlib.pyplot as plt
import os
import cv2
import numpy
import sys
import pickle
if yarp_running:
    import yarp

import matplotlib as mp
# Use this backend for when the server updates plots through the X 
mp.use('TkAgg')
import pylab as pb
import GPy
# To display figures once they're called
pb.ion()
default_seed = 123344
import ABM
import time

global imageDataInputPort
global imageModeInputPort
global imageInputBottle

global imgWidth
global imgHeight
global imgWidthNew
global imgHeightNew
global Y
global Ytest
global X
global L
global Ltest
global SAMObject


# Read Face Data
def readFaceData():
    global Y
    global L

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
    no_pixels=imgWidthNew*imgHeightNew #set_x*set_y
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
                data_image=cv2.resize(data_image, (imgWidthNew, imgHeightNew)) # New
                data_image=cv2.cvtColor(data_image, cv2.COLOR_BGR2GRAY)         
                img_data[:,current_image,count_participant,count_pose] = data_image.flatten()
                # Labelling with participant            
                img_label_data[:,current_image,count_participant,count_pose]=numpy.zeros(no_pixels,dtype=int)+count_participant
    Y=img_data
    L=img_label_data

def prepareFaceData(model='mrd'):    
    """--- Now Y has 4 dimensions: 
    1. Pixels
    2. Images
    3. Person
    4. Movement (Static. up/down. left / right)     

    We can prepare the face data using different scenarios about what to be perceived.
    In each scenario, a different LFM is used. We have:
    - gp scenario, where we regress from images to labels (inputs are images, outputs are labels)
    - bgplvm scenario, where we are only perceiving images as outputs (no inputs, no labels)
    - mrd scenario, where we have no inputs, but images and labels form two different views of the output space.

    The store module of the LFM automatically sees the structure of the assumed perceived data and 
    decides on the LFM backbone to be used.

    ! Important: The global variable Y is changed in this section. From the multi-dim. matrix of all
    modalities, it turns into the training matrix of image data and then again it turns into the 
    dictionary used for the LFM.
    """ 
    global Y
    global X
    global L
    global Ytest
    global Ltest 
    global Ytestn
    global Ltestn
    global data_labels
    
    #--- Config (TODO: move higher up)
    #pose_selection=1
    #---

    K = len(participant_index)   
    # We can do differen scenarios.
    # Y = img_data[:,:,0,1] ; # Load one face, one pose . In this case, set also X=None 
    ttt=Y[:,:,:,pose_selection]
    ttt=numpy.transpose(ttt,(0,2,1))
    Y=ttt.reshape(ttt.shape[0],ttt.shape[2]*ttt.shape[1]) 
    Y=Y.T
    N=Y.shape[0]

    ttt=L[:,:,:,pose_selection]
    ttt=numpy.transpose(ttt,(0,2,1))
    L=ttt.reshape(ttt.shape[0],ttt.shape[2]*ttt.shape[1]) 
    L=L.T
    L=L[:,:1]

    # L = numpy.zeros((N,1))
    # L[0::N/3]=0
    # L[N/3:2*N/3:]=1
    # L[2*N/3::]=2

    Nts=Y.shape[0]-Ntr
   
    perm = numpy.random.permutation(Y.shape[0])
    indTs = perm[0:Nts]
    indTs.sort()
    indTr = perm[Nts:Nts+Ntr]
    indTr.sort()
    Ytest = Y[indTs]
    Ltest = L[indTs]
    Y = Y[indTr]
    L = L[indTr]
    
    # Center data to zero mean and 1 std
    Ymean = Y.mean()
    Yn = Y - Ymean
    Ystd = Yn.std()
    Yn /= Ystd
    # Normalise test data similarly to training data
    Ytestn = Ytest - Ymean
    Ytestn /= Ystd

    # As above but for the labels
    Lmean = L.mean()
    Ln = L - Lmean
    Lstd = Ln.std()
    Ln /= Lstd
    Ltestn = Ltest - Lmean
    Ltestn /= Lstd

    if model == 'mrd':    
        X=None     
        Y = {'Y':Yn,'L':L}
        data_labels = L.copy()
    elif model == 'gp':
        X=Y.copy()
        Y = {'L':Ln.copy()+0.08*numpy.random.randn(Ln.shape[0],Ln.shape[1])}
        data_labels = None
    elif model == 'bgplvm':
        X=None     
        Y = {'Y':Yn}
        data_labels = L.copy()


# training
def prepareTraining(learn=True):
    global SAMObject
    global Y
    global X
    global model_num_inducing
    global model_num_iterations
    global data_labels
    
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

    SAMObject.store(observed=Y, inputs=X, Q=Q, kernel=kernel, num_inducing=model_num_inducing)
    #-- TEMP
    #SAMObject.model['.*rbf.variance'].constrain_bounded(0.8,100)
    #SAMObject.model['.*noise']=SAMObject.model.Y.var()/100
    #SAMObject.model['.*noise'].fix()
    #SAMObject.model.optimize(optimizer='scg',max_iters=100, messages=True)
    

    #---
    if data_labels is not None:
        SAMObject.add_labels(data_labels)
    if learn:
        SAMObject.learn(optimizer='scg',max_iters=model_num_iterations, verbose=True)

#    ret = SAMObject.visualise()



# testing
def testDebug(i=None):
    from scipy.spatial import distance
    import operator
    global Ytestn
    global Ltest
    global SAMObject
    
    ax = SAMObject.visualise()
    visualiseInfo=dict()
    visualiseInfo['ax']=ax
    if i is None:
        for i in range(Ytestn.shape[0]):
            mm,vv,pp=SAMObject.pattern_completion(Ytestn[i,:][None,:],visualiseInfo=visualiseInfo)
            # find nearest neighbour of mm and SAMObject.model.X
            dists = numpy.zeros((SAMObject.model.X.shape[0],1))
 
            # print mm[0].values
         
            for j in range(dists.shape[0]):
                dists[j,:] = distance.euclidean(SAMObject.model.X.mean[j,:], mm[0].values)
            nn, min_value = min(enumerate(dists), key=operator.itemgetter(1))
            if SAMObject.type == 'mrd':
                print "With " + str(vv.mean()) +" prob. error " +participant_index[int(Ltest[i,:])] +" is " + participant_index[int(SAMObject.model.bgplvms[1].Y[nn,:])]
            elif SAMObject.type == 'bgplvm':
                print "With " + str(vv.mean()) +" prob. error " +participant_index[int(Ltest[i,:])] +" is " + participant_index[int(L[nn,:])]
            time.sleep(1)
            l = pp.pop(0)
            l.remove()
            pb.draw()
            del l
    else:
        mm,vv,pp=SAMObject.pattern_completion(Ytestn[i,:][None,:],visualiseInfo=visualiseInfo)
        # find nearest neighbour of mm and SAMObject.model.X
        dists = numpy.zeros((SAMObject.model.X.shape[0],1))

        print "MM (1)"
        print mm[0].values
     
        for j in range(dists.shape[0]):
            dists[j,:] = distance.euclidean(SAMObject.model.X.mean[j,:], mm[0].values)
        nn, min_value = min(enumerate(dists), key=operator.itemgetter(1))
        if SAMObject.type == 'mrd':
            print "With " + str(vv.mean()) +" prob. error " +participant_index[int(Ltest[i,:])] +" is " + participant_index[int(SAMObject.model.bgplvms[1].Y[nn,:])]
        elif SAMObject.type == 'bgplvm':
            print "With " + str(vv.mean()) +" prob. error " +participant_index[int(Ltest[i,:])] +" is " + participant_index[int(L[nn,:])]




def testingImage():
    global SAMObject
    global Ytestn
    
    pred_mean, pred_var = SAMObject.pattern_completion(Ytestn)
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


imgWidth=200
imgHeight=200
imgWidthNew=200#50
imgHeightNew=200#50

if yarp_running:
    # initialise Yarp
    yarp.Network.init()
            
    print "Creating ports..."
    createPorts()

print "Reading Face Data"
readFaceData()

print "Creating Face scenario"
prepareFaceData(model=model_type)

print "Training"
prepareTraining()

print "Debug"
testDebug()

#print "Saving SAMObject"
#ABM.save_model(SAMObject, "saved_SAMObject.txt")

#print "Loading SAMOBject"
#SAMObject2 = ABM.load_model("saved_SAMObject.txt")

"""
print "Show training data"
yy=numpy.reshape(SAMObject.model.Ylist[0],(SAMObject.model.Ylist[0].shape[0],imgHeightNew,imgWidthNew))
for i in range(SAMObject.model.Ylist[0].shape[0]):
    #pb.imshow(numpy.reshape(SAMObject.model.Ylist[0][i,:],(imgHeightNew,imgWidthNew)))
    pb.clf()
    pb.imshow(yy[i,:,:])    
    pb.title(participant_index[int(SAMObject.model.Ylist[1][i,:][0])])
    pb.draw()
    #pb.show()
    time.sleep(0.001)
"""

#print "Visualisation of results"
#SAMObject.visualise()

#print "Interactive visualisation of results"
#SAMObject.visualise_interactive(dimensions=(imgWidthNew, imgHeightNew),view=0)

#while(1):
#    print "Reading image for testing..."
#    readImagesFromCameras()
#    print "Testing..."
#    testingImage()


#print "Images process finish"


