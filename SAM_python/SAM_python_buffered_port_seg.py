# -*- coding: utf-8 -*-
"""
Created on 5 May 2015

@authors: uriel, luke, andreas

"""

#------------------- CONFIGURATION --------------
experiment_number=17
save_model=True
yarp_running = True
# Overall data dir
root_data_dir="/home/icub/dataDump/faceImageData_11062015"#"/home/icub/dataDump/faceImageData_1_05_2015"
image_suffix=".ppm"
participant_index=('Luke','Michael','Adriel','Emma')
pose_index=['Seg']#['A'] #('Straight','LR','Natural') # 'UD')
Ntr=300 # Use a subset of the data for training (and leave the rest for testing)

save_data=False
pickled_save_data_name="Saved_face_Data"
#### Set pose selection to -1 to use all poses....
pose_selection = 0 # 0 # Select a pose to train and test.... e.g. LR = 1

run_abm=True

# Globals for model
global model_type
global model_num_inducing
global model_num_iterations
global model_init_iterations
global model_type
model_type = 'mrd'
model_num_inducing = 35
model_num_iterations = 100
model_init_iterations = 800
#---------------------------------------------------------------------------


import matplotlib.pyplot as plt
import os
import cv2
import numpy
import sys
import pickle
import yarp

import matplotlib as mp
# Use this backend for when the server updates plots through the X 
mp.use('TkAgg')
import pylab as pb
import GPy
# To display figures once they're called
pb.ion()
default_seed = 123344
numpy.random.seed(seed=default_seed)
#import ABM
from ABM import ABM
import time
import matplotlib.cm as cm
from scipy.spatial import distance
import operator

global imageDataInputPort
global imageModeInputPort
global outputFacePrection
global imageInputBottle
global speakStatusPort
global speakStatusOutBottle
global speakStatusInBottle

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
global commands
global predict
global Ystd
global Ymean
global Ytestn
global Ltest
global L
global imageArray
global yarpImage
global imageFlatten
global newImage

global pp

global syncPort
global syncPortBottle

class imageDataProcessor(yarp.PortReader):
    
    def read(self, connection):
        global imageDataInputPort
        global imageInputBottle
        print "In dataProcessor.read"
        if not(connection.isValid()):
            print "Connection shutting down"
            return False
        
        print "callback 1"
        #bottle_in = yarp.Bottle()
        bottle_in = yarp.Port()
        
        print "Trying to read from connection"
        
        ok = bottle_in.read(connection)
#        imageDataInputPort.read(connection)

        print "callback 2"

        #imageArray=cv2.resize(imageArray,(imgHeightNew,imgWidthNew))

        #print "callback 3"

        #imageFlatten = imageArray.flatten()

        #print "callback 4"
        
        if not(ok):
            print "Failed to read input"
            return False

        print "callback 3"

#        print "Bottle_In size: ", bottle_in.size()

        #if( bottle_in.size() > 0 ):
        #    imageInputBottle = bottle_in
        #else:
        #    imageInputBottle.clear()

        #for i in range(imgWidth*imgHeight):
        #    imageInputBottle.addInt(int(imageFlatten[i]))


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
    global Ymean
    global Ystd
    global data_labels

    #--- Config (TODO: move higher up)
#    motion=0
    #---

    K = len(participant_index)   
    # We can do differen scenarios.
    # Y = img_data[:,:,0,1] ; # Load one face, one pose . In this case, set also X=None

	# Take all poses if pose selection ==-1
    if pose_selection == -1:
        ttt=numpy.transpose(Y,(0,1,3,2))
        ttt=ttt.reshape((ttt.shape[0],ttt.shape[1]*ttt.shape[2],ttt.shape[3])) 
    else:
		ttt=Y[:,:,:,pose_selection]
    ttt=numpy.transpose(ttt,(0,2,1))
    Y=ttt.reshape(ttt.shape[0],ttt.shape[2]*ttt.shape[1]) 
    Y=Y.T
    N=Y.shape[0]

    if pose_selection == -1:
        ttt=numpy.transpose(L,(0,1,3,2))
        ttt=ttt.reshape((ttt.shape[0],ttt.shape[1]*ttt.shape[2],ttt.shape[3]))
    else:
		ttt=L[:,:,:,pose_selection]
    ttt=numpy.transpose(ttt,(0,2,1))
    L=ttt.reshape(ttt.shape[0],ttt.shape[2]*ttt.shape[1]) 
    L=L.T
    L=L[:,:1]

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

    #print "====================== Y shape in training ", Y['Y'].shape()


# training
def prepareTraining():
    global SAMObject
    global Y
    global X
    global model_num_inducing
    global model_num_iterations
    global model_init_iterations
    global data_labels
    
    if X is not None:
        Q = X.shape[1]
    else:
        Q=2

    # Instantiate object
    SAMObject=ABM.LFM()
    if Q > 100:
        kernel = GPy.kern.RBF(Q, ARD=False) + GPy.kern.Bias(Q) + GPy.kern.White(Q)
    else:
        kernel = None
    # Simulate the function of storing a collection of events
    SAMObject.store(observed=Y, inputs=X, Q=Q, kernel=kernel, num_inducing=model_num_inducing)
    # If data are associated with labels (e.g. face identities), associate them with the event collection
    if data_labels is not None:
        SAMObject.add_labels(data_labels)
    # Simulate the function of learning from stored memories, e.g. while sleeping (consolidation).
    SAMObject.learn(optimizer='scg',max_iters=model_num_iterations, init_iters=model_init_iterations, verbose=True)


# testing for new images
def testingImage(testFace, visualiseInfo=None):
    global Ytest
    global Ltest
    global SAMObject
    global pp

#    print "PREDICT debug 1"

    #print "====================== testFace shape in testing ",  testFace.shape

    #print "IMAGE TESTING: ", numpy.shape(testFace)
    
    # Returns the predictive mean, the predictive variance and the axis (pp) of the latent space backwards mapping.
    mm,vv,pp=SAMObject.pattern_completion(testFace, visualiseInfo=visualiseInfo)
    fig_nn = visualiseInfo['fig_nn']
#    print "PREDICT debug 2"

    # find nearest neighbour of mm and SAMObject.model.X
    dists = numpy.zeros((SAMObject.model.X.shape[0],1))

#    print "PREDICT debug 3"

    # print mm[0].values

    facePredictionBottle = yarp.Bottle()
    
    for j in range(dists.shape[0]):
        dists[j,:] = distance.euclidean(SAMObject.model.X.mean[j,:], mm[0].values)
    nn, min_value = min(enumerate(dists), key=operator.itemgetter(1))
    if SAMObject.type == 'mrd':
        print "With " + str(vv.mean()) +" prob. error the new image is " + participant_index[int(SAMObject.model.bgplvms[1].Y[nn,:])]
        #if vv > 1e-4: ##################### CHANGE THE THRESHOLD BY OPTIMISING IT
        #    facePredictionBottle.addString("I do not know you.")
        #else:
        #facePredictionBottle.addString("Hello " + participant_index[int(SAMObject.model.bgplvms[1].Y[nn,:])])
        textStringOut=participant_index[int(SAMObject.model.bgplvms[1].Y[nn,:])]
    elif SAMObject.type == 'bgplvm':
        print "With " + str(vv.mean()) +" prob. error the new image is " + participant_index[int(L[nn,:])]
        #facePredictionBottle.addString("Hello " + participant_index[int(L[nn,:])])
        textStringOut=participant_index[int(L[nn,:])]
    
    # If confidence is higher enough (low error)
    if (vv.mean()<0.00012):
        facePredictionBottle.addString("Hello " + textStringOut)
    # Otherwise ask for updated name... (TODO: add in updated name)
    else:
        facePredictionBottle.addString("I think you are " + textStringOut + " but im not sure, please confirm?")        
    
    # Plot the training NN of the test image (the NN is found in the INTERNAl, compressed (latent) memory space!!!)
    if fig_nn is not None:
        fig_nn = pb.figure(11)
        pb.title('Training NN')
        fig_nn.clf()
        pl_nn = fig_nn.add_subplot(111)
        pl_nn.imshow(numpy.reshape(SAMObject.recall(nn),(imgHeightNew, imgWidthNew)), cmap=cm.Greys_r)
        pb.title('Training NN')
        pb.show()
        pb.draw()
        #pb.waitforbuttonpress(0.2)
    speakStatusPort.write(speakStatusOutBottle, speakStatusInBottle)

    print "======== iSpeak status: ", speakStatusInBottle.get(0).asString(), " ==========="

    if( speakStatusInBottle.get(0).asString() == "quiet"):
        outputFacePrection.write(facePredictionBottle)

    facePredictionBottle.clear()

#    time.sleep(2)

# testing in pre-stored test data - FOR DEBUG
def testDebug(SAMObject, Ytestn, Ltest, L, i=None):
    ax = SAMObject.visualise()
    visualiseInfo=dict()
    visualiseInfo['ax']=ax
    if i is None:
        Nstar = Ytestn.shape[0]
        success_arr = numpy.repeat(False, Nstar)
        uncert_arr = numpy.repeat(-numpy.inf, Nstar)
        for i in range(Nstar):
            mm,vv,pp=SAMObject.pattern_completion(Ytestn[i,:][None,:],visualiseInfo=visualiseInfo)
            uncert_arr[i] = vv
            # find nearest neighbour of mm and SAMObject.model.X
            dists = numpy.zeros((SAMObject.model.X.shape[0],1))
 
            # print mm[0].values
         
            for j in range(dists.shape[0]):
                dists[j,:] = distance.euclidean(SAMObject.model.X.mean[j,:], mm[0].values)
            nn, min_value = min(enumerate(dists), key=operator.itemgetter(1))
            label_true = int(Ltest[i,:])
            if SAMObject.type == 'mrd':
                label_pred = int(SAMObject.model.bgplvms[1].Y[nn,:])
            elif SAMObject.type == 'bgplvm':
                label_pred = int(L[nn,:])
            success_arr = (label_pred == label_true)
            print "With " + str(vv.mean()) +" prob. error " +participant_index[label_true] +" is " + participant_index[label_pred]
            time.sleep(1)
            l = pp.pop(0)
            l.remove()
            pb.draw()
            del l
        return success_arr, uncert_arr
    else:
        mm,vv,pp=SAMObject.pattern_completion(Ytestn[i,:][None,:],visualiseInfo=visualiseInfo)
        # find nearest neighbour of mm and SAMObject.model.X
        dists = numpy.zeros((SAMObject.model.X.shape[0],1))

        print "MM (1)"
        print mm[0].values
     
        for j in range(dists.shape[0]):
            dists[j,:] = distance.euclidean(SAMObject.model.X.mean[j,:], mm[0].values)
        nn, min_value = min(enumerate(dists), key=operator.itemgetter(1))
        label_true = int(Ltest[i,:])
        if SAMObject.type == 'mrd':
            label_pred = int(SAMObject.model.bgplvms[1].Y[nn,:])
        elif SAMObject.type == 'bgplvm':
            label_pred = int(L[nn,:])
        print "With " + str(vv.mean()) +" prob. error " +participant_index[label_true] +" is " + participant_index[label_pred]


def optUncertaintyThreshold(YTestn, Ln, L):
    global SAMObject
    # Get error rates for the test data
    success_tr_arr, uncertainty_tr_arr = testDebug(SAMObject, Ytestn, Ltest, L, i=None)
    # Get error rates for the training data (ie reconstruction)
    success_ts_arr, uncertainty_ts_arr = testDebug(SAMObject, SAMObject.recall(-1), L, L, i=None)
    success_tr_ind = numpy.where(success_tr_arr == True)
    success_ts_ind = numpy.where(success_ts_ind == True)
    mean_suc_unc = numpy.average(np.hstack((uncertainty_ts_arr[succes_ts_ind,:].mean(), uncertainty_tr_arr[succes_tr_ind,:].mean())), axis=0, weights=[0.1,0.9])

    # TODO



# create ports
def createPorts():
    global imageDataInputPort
    global imageModeInputPort
    global outputFacePrection
    global speakStatusPort
    global speakStatusBottle
    global imageInputBottle
    global speakStatusOutBottle
    global speakStatusInBottle
    global syncPort
    global syncPortBottle

#    imageDataInputPort = yarp.Port()
#    imageDataInputPort.open("/sam/imageData:i")

    imageDataInputPort = yarp.BufferedPortImageRgb()
    imageDataInputPort.open("/sam/imageData:i");

    outputFacePrection = yarp.Port()
    outputFacePrection.open("/sam/facePrediction:o")

    speakStatusPort = yarp.RpcClient();
    speakStatusPort.open("/sam/speakStatus:i")

    speakStatusOutBottle = yarp.Bottle()
    speakStatusOutBottle.addString("stat")

    speakStatusInBottle = yarp.Bottle()

    imageInputBottle = yarp.Bottle()


#    syncPort = yarp.Port()
#    syncPort.open("/sam/syncPort:o");

#    syncPortBottle = yarp.Bottle()
#    syncPortBottle.addString("sam_ready")



def createImageArrays():
    global imageArray
    global yarpImage
    global imgWidth
    global imgHeight  
    global newImage


    imageArray = numpy.zeros((imgHeight, imgWidth, 3), dtype=numpy.uint8)
    newImage = yarp.ImageRgb()
    yarpImage = yarp.ImageRgb()
    yarpImage.resize(imgWidthNew,imgWidthNew)
    yarpImage.setExternal(imageArray, imageArray.shape[1], imageArray.shape[0])



def readImagesFromCameras():
    global imgWidth
    global imgHeight
    global imageInputBottle
    global Ytest
    global predict
    global yarpImage
    global imageArray
    global Ystd
    global Ymean
    global newImage

    predict = True

#    print "READ IMAGE DEBUG 1"
    newImage = imageDataInputPort.read()

    yarpImage.copy(newImage)

    # here image has to be resized
    #plt.imshow(imageArray)

#    print "READ IMAGE DEBUG 2"
    imageArrayOld=cv2.resize(imageArray,(imgHeightNew,imgWidthNew))
    imageArrayGray=cv2.cvtColor(imageArrayOld, cv2.COLOR_BGR2GRAY)         

    plt.figure(10)
    plt.title('Image received')
    plt.imshow(imageArrayGray, cmap=cm.Greys_r)
    plt.show()
    plt.waitforbuttonpress(0.1)

#    print "READ IMAGE DEBUG 3"
    #imageFlatten_testing = numpy.zeros((imgHeightNew*imgWidthNew,1))

    #Ytest = numpy.zeros(imgHeightNew*imgWidthNew)
    imageFlatten_testing = imageArrayGray.flatten()
    imageFlatten_testing = imageFlatten_testing - Ymean
    imageFlatten_testing = imageFlatten_testing/Ystd
    #imageFlatten_testing = imageFlatten_testing/imageFlatten_testing.std()
    
#    print "READ IMAGE DEBUG 4"
    #Ytest = numpy.zeros((imageFlatten.size()))
    #Ytest = imageFlatten

    #print "Size (readImagesFromCameras function): ", Ytest.size


    #for i in range(imageInputBottle.size()):
    #    Ytest[i] = imageInputBottle.get(i).asInt();

    #print "READ IMAGE DEBUG 5"
    #for i in range(imageFlatten.size()):
    #    Ytest[i] = imageFlatten[i]


    imageFlatten_testing = imageFlatten_testing[:,None].T

#    print "READ IMAGE CAMERAS: ", numpy.shape(imageFlatten_testing)

    #L = numpy.zeros(numpy.shape(imageFlatten_testing))

    #temp = {'Y':imageFlatten_testing,'L':L}

    #print "READ IMAGE DEBUG 6"
    #if( Ytest.size > 0 ):
    #    Ytest = Ytest[:,None].T
    ##    predict = True
    #    print "Ytest content: ", Ytest


    imageInputBottle.clear()

    return imageFlatten_testing



# initialise Yarp
yarp.Network.init()

print "WARNING LB CHANGED IMGWIDTH AND HEIGHT HERE TO 400x400"
imgWidth=400
imgHeight=400
imgWidthNew=150
imgHeightNew=150
fname = 'm_' + model_type + '_exp' + str(experiment_number) #+ '.pickle'

predict = False

#port_read = imageDataProcessor()

print "Creating ports..."
createPorts()

print "Creating image arrays..."
createImageArrays()

print "Reading Face Data..."
readFaceData()

print "Creating Face scenario..."
prepareFaceData(model=model_type)

import os.path

if not os.path.isfile(fname+ '.pickle') :
	print "Training..."
	prepareTraining()
	
	print "Saving SAMObject"
	if save_model:
		ABM.save_pruned_model(SAMObject, fname)
else:
	print "Loading SAMOBject"
	SAMObject = ABM.load_pruned_model(fname) #ABM.load_model(fname)
#SAMObject = ABM.load_model('m_good2.pickle') #######TMP

print "Optimising uncertainty threshold..."
#optUncertaintyThreshold(YTestn, Ln, L)

print "Waiting for connection with imageDataInputPort..."
#while( not(yarp.Network.isConnected("/sam/faceTracker:o","/sam/imageData:i")) ):
while( not(yarp.Network.isConnected("/faceTrackerImg:o","/sam/imageData:i")) ):
    pass

#while( not(yarp.Network.isConnected("/sam/syncPort:o","/faceTracker/syncPort:i")) ):
#    pass

print "Connection ready"
#imageDataInputPort.setReader(port_read)

# This is for visualising the mapping of the test face back to the internal memory
ax = SAMObject.visualise()
visualiseInfo=dict()
visualiseInfo['ax']=ax
ytmp = SAMObject.recall(0)
ytmp = numpy.reshape(ytmp,(imgHeightNew,imgWidthNew))
fig_nn = pb.figure(11)
pb.title('Training NN')
pl_nn = fig_nn.add_subplot(111)
ax_nn = pl_nn.imshow(ytmp, cmap=cm.Greys_r)
pb.draw()
pb.show()
visualiseInfo['fig_nn']=fig_nn



while 1:
    try:
        if not(imageDataInputPort == None):
    #        syncPort.write(syncPortBottle)
    #        print "+++++++++++++++ syncPort: sam_ready +++++++++++++++++++"

            testFace = readImagesFromCameras()
            if( predict ):
                testingImage(testFace, visualiseInfo)
                predict=False


        time.sleep(1)
        # Delete the newly added point in the internal memory representation
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

