# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:00:20 2015

@author: luke
Load face data and convert for ABM

Data Dimensions:
1. Pixels
2. Images
3. Person
4. Movement (Static. up/down. left / right) 

"""

# Overall data dir
#root_data_dir="D:/robotology/SheffABM/dataDump/faceImageData/"
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
import numpy as np
import sys
import pickle

if not os.path.exists(root_data_dir):
    print "CANNOT FIND:" + root_data_dir

#load_file = os.path.join(root_data_dir, image_suffix ) 
#data_image=cv2.imread(os.path.join(data_directory_list[0],data_file_database[0]['img_fname'][0]))                

## Find and build index of available images.......
data_file_count=np.zeros([len(participant_index),len(pose_index)])
data_file_database={}
for count_participant, current_participant in enumerate(participant_index):
    data_file_database_part={}
    for count_pose, current_pose in enumerate(pose_index):
        current_data_dir=os.path.join(root_data_dir,current_participant+current_pose)
        data_file_database_p=np.empty(0,dtype=[('orig_file_id','i2'),('file_id','i2'),('img_fname','a100')])
        data_image_count=0
        if os.path.exists(current_data_dir):
            for file in os.listdir(current_data_dir):
                #parts = re.split("[-,\.]", file)
                fileName, fileExtension = os.path.splitext(file)
                if fileExtension==image_suffix: # Check for image file
                    file_ttt=np.empty(1, dtype=[('orig_file_id','i2'),('file_id','i2'),('img_fname','a100')])
                    file_ttt['orig_file_id'][0]=int(fileName)
                    file_ttt['img_fname'][0]=file
                    file_ttt['file_id'][0]=data_image_count
                    data_file_database_p = np.append(data_file_database_p,file_ttt,axis=0)
                    data_image_count += 1
            data_file_database_p=np.sort(data_file_database_p,order=['orig_file_id'])  
        data_file_database_part[pose_index[count_pose]]=data_file_database_p
        data_file_count[count_participant,count_pose]=len(data_file_database_p)
    data_file_database[participant_index[count_participant]]=data_file_database_part

# To access use both dictionaries data_file_database['Luke']['LR']
# Cutting indexes to smllest number of available files -> Using file count
min_no_images=int(np.min(data_file_count))

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
img_data=np.zeros([no_pixels, min_no_images, len(participant_index),len(pose_index)])
img_label_data=np.zeros([no_pixels, min_no_images, len(participant_index),len(pose_index)],dtype=int)
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
            img_label_data[:,current_image,count_participant,count_pose]=np.zeros(no_pixels,dtype=int)+count_participant


### Save data
if save_data:
    saved_data = [img_data, img_label_data, participant_index, pose_index]
    pickle.dump( saved_data, open( os.path.join(root_data_dir,pickled_save_data_name+".pickle"), "wb" ) )


if run_abm:
    
    # Copyright (c) 2015, Andreas Damianou
    
    """
    This is a demo for unsupervised learning, ie the robot
    just perceives the world. In this case, we only have observables
    (and no inputs).
    """
    
    """ 
    Import necessary libraries
    """
    import matplotlib as mp
    # Use this backend for when the server updates plots through the X 
    mp.use('TkAgg')
    import numpy as np
    import pylab as pb
    import GPy
    # To display figures once they're called
    pb.ion()
    default_seed = 123344
    #import pods
    #from ABM import ABM
    import ABM
    
    """
    Prepare some data. This is NOT needed in the final demo,
    since the data will come from the iCub through the
    drivers. So, the next section is to just test the code
    in standalone mode.
    """
    # data = pods.datasets.brendan_faces()
    Ntr = 100
    Nts = 50
    
    #data = pods.datasets.oil()
#    Y = data['X'] # Data
#    L = data['Y'] # Labels
    # Just using faces from pose=straight
    Y = img_data[:,:,0,1]    
    L = img_label_data[:,:,0,1]
    
    Y=Y.T    
    
    perm = np.random.permutation(Y.shape[0])
    indTs = perm[0:Nts]
    indTs.sort()
    indTr = perm[Nts:Nts+Ntr]
    indTr.sort()
    Ytest = Y[indTs]
    Ltest = L[indTs]
    Y = Y[indTr]
    #L = L[indTr]
    
    
    """
    This needs to go to the code, since it's what happens after data collection
    """
    
    #--- Observables (outputs) - no inputs in this demo
    # Normalise data (zero mean, std=1)
    Ymean = Y.mean()
    Yn = Y - Ymean
    Ystd = Yn.std()
    Yn /= Ystd
    # Normalise test data similarly to training data
    Ytest-= Ymean
    Ytest /= Ystd
    
    # Build a dictionary. Every element in the dictionary is one modality
    # (one view of the data). Here we have only one modality.
    Y = {'Y':Yn}
    
    # The dimensionality of the compressed space (how many features does
    # each original point is compressed to)
    Q = 2
    
#    for i in range(Y['Y'].shape[0]-1):
#        pb.imshow(Y['Y'][i,:].reshape(200,200))
#        time.sleep(0.1)
#        pb.draw()
    
    # Instantiate object
    a=ABM.LFM()
    
    # Store the events Y.
    # ARG: Y: A N x D matrix, where N is the number of points and D the number
    # of features needed to describe each point.
    # ARG: Q: See above 
    # ARG: kernel: can be left as None for default.
    # ARG: num_inducing: says how many inducing points to use. Inducing points are
    # a fixed number of variables through which all memory is filtered, to achieve
    # full compression. E.g. it can correspond to the number of neurons.
    # Of course, this is not absolutely fixed, but it also doesn't grow necessarily
    # proportionally to the data, since synapses can make more complicated combinations
    # of the existing neurons. The GP is here playing the role of "synapses", by learning
    # non-linear and rich combinations of the inducing points.
    a.store(observed=Y, inputs=None, Q=Q, kernel=None, num_inducing=40)
    
    # In this problem each of the observables (each row of Y) also has a label.
    # This can be added through the next line via L, where L is a N x K matrix,
    # where K is the total number of labels. 
    #a.add_labels(L.argmax(axis=1))
    
    # Learn from the data, (analogous to forming synapses)
    a.learn(optimizer='bfgs',max_iters=300, verbose=True)
    
    # This is an important function: It visualises the internal state/representation
    #  of the memory.
    ret = a.visualise()
    
    # Pattern completion. In this case, we give a new set of test observables 
    # (i.e. events not experienced before) and we want to infer the internal/compressed
    # representation of those. We can then perform inference in this compressed representation.
    # pred_mean is the point estimates of the inernal representation adn pred_Var is the variance
    # (cenrtainty) with which they were predicted (low variance = high certainty)
    
    #pred_mean, pred_var = a.pattern_completion(Ytest)
    
    # Visualise the predictive point estimates for the test data
    
    #pb.plot(pred_mean[:,0],pred_mean[:,1],'bx')
