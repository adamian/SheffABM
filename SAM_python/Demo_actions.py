#!/usr/bin/python

#
#The University of Sheffield
#WYSIWYD Project
#
#
#

import matplotlib.pyplot as plt
#from SAMpy import SAMpy
import pylab as pb
import sys
import pickle
import os
import numpy as np
import time
import operator

# This will be moved to the driver
from scipy import stats
from scipy.spatial import distance
import Driver



action_index=('down','left','right','up')#'Michael','Andreas')

# testing in pre-stored test data - FOR DEBUG
def testDebug(SAMObject, Ytestn, Ltest, L, i=None):
    ax = SAMObject.visualise()
    visualiseInfo=dict()
    visualiseInfo['ax']=ax
    if i is None:
        Nstar = Ytestn.shape[0]
        success_arr = np.repeat(False, Nstar)
        uncert_arr = np.repeat(-np.inf, Nstar)
        for i in range(Nstar):
            mm,vv,pp=SAMObject.pattern_completion(Ytestn[i,:][None,:],visualiseInfo=visualiseInfo)
            #uncert_arr[i] = vv
            # find nearest neighbour of mm and SAMObject.model.X
            dists = np.zeros((SAMObject.model.X.shape[0],1))
 
            # print mm[0].values
         
            for j in range(dists.shape[0]):
                dists[j,:] = distance.euclidean(SAMObject.model.X.mean[j,:], mm[0].values)
            nn, min_value = min(enumerate(dists), key=operator.itemgetter(1))
            label_true = int(Ltest[i,:])
            if SAMObject.type == 'mrd':
                label_pred = int(np.argmax(SAMObject.model.bgplvms[1].Y[nn,:],0))####
            elif SAMObject.type == 'bgplvm':
                label_pred = int(L[nn,:])
            success_arr = (label_pred == label_true)
            print "With " + str(vv.mean()) +" prob. error " +action_index[label_true] +" is " + action_index[label_pred]
            time.sleep(1)
            l = pp.pop(0)
            l.remove()
            pb.draw()
            del l
        return success_arr, uncert_arr
    else:
        mm,vv,pp=SAMObject.pattern_completion(Ytestn[i,:][None,:],visualiseInfo=visualiseInfo)
        # find nearest neighbour of mm and SAMObject.model.X
        dists = np.zeros((SAMObject.model.X.shape[0],1))

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
        print "With " + str(vv.mean()) +" prob. error " +action_index[label_true] +" is " + action_index[label_pred]



# Creates a SAMpy object
# TODO: mySAMpy = SAMpy(True, imgH = 400, imgW = 400, imgHNew = 200, imgWNew = 200,inputImagePort="/visionDriver/image:o")

# Specification of the experiment number
experiment_number = 404

# Location of face data
#root_data_dir="/home/icub/dataDump/faceImageData_11062015"
# Image format
image_suffix=".ppm"
# Use a subset of the data for training
Ntr=25

# Action selected for training
action_selection = 0

# Specification of model type and training parameters
model_type = 'mrd'
model_num_inducing = 15
model_num_iterations = 20
model_init_iterations = 80
fname = 'm_' + model_type + '_exp' + str(experiment_number) #+ '.pickle'

# Enable to save the model and visualise GP nearest neighbour matching
save_model=True
visualise_output=True

# Reading action data, preparation of data and training of the model
# TODO #mySAMpy.readFaceData(root_data_dir, action_index, pose_index)
# TODO #mySAMpy.prepareFaceData(model_type, Ntr, pose_selection)
# TODO #mySAMpy.training(model_num_inducing, model_num_iterations, model_init_iterations, fname, save_model)

# TODO: The above will read the action data and transform them so that:
# We have A different actions in the dictionary. E.g. A=4, for pull, push, up, down.
# We have recorded N actions stored in a dictionary Y0. Y['n'] is a S_n x D matrix, saying that for
# the n-th action we recorded S_n D-dimensional steps. Here we'll take D=2, so that each step is the
# x-y coordinates (later add z) in the plane. 
# We also have X0 being a N x S_n matrix, with the times at which Y0['n'] were observed.
# Now for each pair { X0[n], Y['n']} we fit a linear model and obtain the parameters [a,b] (the slope and
# intercept of the line). We store these parameters in the N x 2 matrix in position Y[n,:].
# We finally create a matrix L which is N x A, which is saying what sort of action Y[n,:] corresponds to
# (1-of-K encoding).
"""  ------ The driver should be called to load and prepare data. For the moment,
we will use toy data and later incorporate the driver.
"""
N=60
Dmean=20
# yy is the final data matrix, where each row is a trial and each trial is represented
# by 4 numbers: 2 for the linregress parameters for x axis, 2 for y-axis
yy = np.zeros((N, 4))
# L contains the labels
L = np.zeros((N, 4))
for i in range(N):
    # The "frames" captured for the i-th point
    Dcur = Dmean + np.floor((np.random.rand()-0.5)*10)
    # The time vector for the i-th point
    x = np.random.rand(1,Dcur)
    x.sort()
    x=x.T
    # The root (base) coordinates of the i-th point. Should be around 0
    # Random between -0.5 and 0.5
    rootx = (np.random.rand()-0.5)
    rooty = (np.random.rand()-0.5)
    # Type of action. Available is: right, left, up, down
    p = np.random.rand()
    if p <= 0.25:
        action = 'up'
        L[i,0] = 1
    elif p <= 0.5:
        action = 'down'
        L[i,1] = 1
    elif p <= 0.75:
        action = 'right'
        L[i,2] = 1
    else:
        action = 'left'
        L[i,3] = 1
    # The coordinates of the i-th point (currently only x,y coords)
    y = np.zeros((Dcur, 2))
    y[0,0] = rootx
    y[0,1] = rooty

    for j in np.arange(1,Dcur,1):
        if action == 'up':
            # random noise to x
            y[j,0] = rootx + (np.random.rand()-0.5)
            # Increment y and add random noise
            y[j,1] = y[j-1,1] + np.abs(np.random.rand()*5) + (np.random.rand()-0.5)
        elif action == 'down':
            # random noise to x
            y[j,0] = rootx + (np.random.rand()-0.5)
            # Decrement y and add random noise
            y[j,1] = y[j-1,1] - np.abs(np.random.rand()*5) + (np.random.rand()-0.5)
        elif action == 'right':
            # Increment x
            y[j,0] = y[j-1,0] + np.abs(np.random.rand()*5) + (np.random.rand()-0.5)
            # Random noise to y
            y[j,1] = rooty + (np.random.rand()-0.5)
        else:
            # Decrement x
            y[j,0] = y[j-1,0] - np.abs(np.random.rand()*5) + (np.random.rand()-0.5)
            # Random noise to y
            y[j,1] = rooty + (np.random.rand()-0.5)           
        #pb.close('all'); pb.plot(x[0,:],y,'x-'); pb.legend(('x','y'))


    #--- Move this to corresponding Driver:ActionDriver:read module and Driver:prepare
    #from scipy import stats
    a0, b0, r_value, p_value, std_err = stats.linregress(x[:,0],y[:,0])
    a1, b1, r_value, p_value, std_err = stats.linregress(x[:,0],y[:,1])
    yy[i,:] = [a0, b0, a1, b1]

L = np.argmax(L,1)[:,None]
""" ------------------ """
# DEBUG: Check if we find correct classes
from scipy.cluster.vq import kmeans2
cent, lbls = kmeans2(yy,4)
#print np.argmax(L,1)
#print lbls


d = Driver.Driver()

""" This will be replaced by d.readData() """
d.Y = yy
d.X = x
d.L = L

Nts=d.Y.shape[0]-Ntr

perm = np.random.permutation(d.Y.shape[0])
indTs = perm[0:Nts]
indTs.sort()
indTr = perm[Nts:Nts+Ntr]
indTr.sort()
d.Ytest = d.Y[indTs]
d.Ltest = d.L[indTs]
d.Y = d.Y[indTr]
d.L = d.L[indTr]

# Center data to zero mean and 1 std
d.Ymean = d.Y.mean()
d.Yn = d.Y - d.Ymean
d.Ystd = d.Yn.std()
d.Yn /= d.Ystd
# Normalise test data similarly to training data
d.Ytestn = d.Ytest - d.Ymean
d.Ytestn /= d.Ystd

# As above but for the labels
d.Lmean = d.L.mean()
d.Ln = d.L - d.Lmean
d.Lstd = d.Ln.std()
d.Ln /= d.Lstd
d.Ltestn = d.Ltest - d.Lmean
d.Ltestn /= d.Lstd

if model_type == 'mrd':    
    d.X=None     
    d.Y = {'Y':d.Yn,'L':d.L}
    d.data_labels = d.L.copy()
elif model_type == 'gp':
    d.X=d.Y.copy()
    d.Y = {'L':d.Ln.copy()+0.08*np.random.randn(d.Ln.shape[0],d.Ln.shape[1])}
    d.data_labels = None
elif model_type == 'bgplvm':
    d.X=None     
    d.Y = {'Y':d.Yn}
    d.data_labels = d.L.copy()
""" --------------------- """


d.training(model_num_inducing, model_num_iterations, model_init_iterations, fname, save_model)



if visualise_output: 
    ax = d.SAMObject.visualise()
    visualiseInfo=dict()
    visualiseInfo['ax']=ax
else:
    visualiseInfo=None

testDebug(d.SAMObject, d.Ytest, d.Ltest, d.L)


# # This is for visualising the mapping of the test face back to the internal memory
# if visualise_output: 
#     ax = mySAMpy.SAMObject.visualise()
#     visualiseInfo=dict()
#     visualiseInfo['ax']=ax
#     ytmp = mySAMpy.SAMObject.recall(0)
#     ytmp = np.reshape(ytmp,(mySAMpy.imgHeightNew,mySAMpy.imgWidthNew))
#     fig_nn = pb.figure()
#     pb.title('Training NN')
#     pl_nn = fig_nn.add_subplot(111)
#     ax_nn = pl_nn.imshow(ytmp, cmap=plt.cm.Greys_r)
#     pb.draw()
#     pb.show()
#     visualiseInfo['fig_nn']=fig_nn
# else:
#     visualiseInfo=None

# # Read and test images from iCub eyes in real-time

# #fig_input = pb.figure()
# #subplt_input = fig_input.add_subplot(111)

# while( True ):
#     try:
#         testFace = mySAMpy.readImageFromCamera()
#         #subplt_input.imshow(testFace, cmap=plt.cm.Greys_r)
#         pp = mySAMpy.testing(testFace, visualiseInfo)
#         time.sleep(0.5)
#         l = pp.pop(0)
#         l.remove()
#         pb.draw()
#         del l
#     except KeyboardInterrupt:
#         print 'Interrupted'
#         try:
#             sys.exit(0)
#         except SystemExit:
#             os._exit(0)

