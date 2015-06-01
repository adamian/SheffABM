
import matplotlib.pyplot as plt
from SAMpy import SAMpy
import pylab as pb
import sys
import pickle
import os
import numpy
import time
import operator

mySAMpy = SAMpy(True)
experiment_number = 10
#root_data_dir="/home/uriel/Downloads/dataDump"
#participant_index=('Luke','Uriel','Michael')

root_data_dir="/home/icub/dataDump/faceImageData_13_05_2015"
image_suffix=".ppm"
participant_index=('Luke','Uriel','Andreas')#'Michael','Andreas')
pose_index=['A'] #('Straight','LR','Natural') # 'UD')
Ntr=300 # Use a subset of the data for training (and leave the rest for testing)

pose_selection = 0
model_type = 'mrd'
model_num_inducing = 35
model_num_iterations = 100
model_init_iterations = 800
fname = 'm_' + model_type + '_exp' + str(experiment_number) #+ '.pickle'

save_model=True
### Visualise GP nearest neighbour matching
visualise_output=True

mySAMpy.readFaceData(root_data_dir, participant_index, pose_index)
mySAMpy.prepareFaceData(model_type, Ntr, pose_selection)
mySAMpy.training(model_num_inducing, model_num_iterations, model_init_iterations, fname, save_model)

if visualise_output: 
    # This is for visualising the mapping of the test face back to the internal memory
    ax = mySAMpy.SAMObject.visualise()
    visualiseInfo=dict()
    visualiseInfo['ax']=ax
    ytmp = mySAMpy.SAMObject.recall(0)
    ytmp = numpy.reshape(ytmp,(mySAMpy.imgHeightNew,mySAMpy.imgWidthNew))
    fig_nn = pb.figure()
    pb.title('Training NN')
    pl_nn = fig_nn.add_subplot(111)
    ax_nn = pl_nn.imshow(ytmp, cmap=plt.cm.Greys_r)
    pb.draw()
    pb.show()
    visualiseInfo['fig_nn']=fig_nn
else:
    visualiseInfo=None

while( True ):
    try:
        testFace = mySAMpy.readImageFromCamera()
        pp = mySAMpy.testing(testFace, visualiseInfo)
        time.sleep(0.5)
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

