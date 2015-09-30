# Read images from iCub cameras

import numpy
import yarp
import time
import Image
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
#import pylab

global leftInputPort
global rightInputPort
global interfaceProcess
global leftImageArray
global leftYarpImage
global rightImageArray
global rightYarpImage
global outputCamerasRealTime
global camerasInfoBottle

# create ports
def createPorts():
    global leftInputPort
    global rightInputPort

    leftInputPort = yarp.Port()
    rightInputPort = yarp.Port()
    leftInputPort.open("/interface/cam/left")
    rightInputPort.open("/interface/cam/right")

    global outputCamerasRealTime

    outputCamerasRealTime = yarp.Port()
    outputCamerasRealTime.open("/realTime/cam")    # this port send the path and name of current image from left camera

    global camerasInfoBottle
    camerasInfoBottle = yarp.Bottle()

    return True

# connect ports to iCub cameras
def connectPorts():
    if not( yarp.Network.connect("/icub/camSim/left", "/interface/cam/left") ):
        return False

    if not( yarp.Network.connect("/icub/camSim/right", "/interface/cam/right") ):
        return False

    return True

# disconnect ports from iCub cameras
def disconnectPorts():
    yarp.Network.disconnect("/icub/camSim/left", "/interface/cam/left")
    yarp.Network.disconnect("/icub/camSim/right", "/interface/cam/right")
    return True

def createImageArrays():
    global leftImageArray
    global leftYarpImage
    global rightImageArray
    global rightYarpImage
    leftImageArray = numpy.zeros((240, 320, 3), dtype=numpy.uint8)
    leftYarpImage = yarp.ImageRgb()
    leftYarpImage.resize(320,240)
    leftYarpImage.setExternal(leftImageArray, leftImageArray.shape[1], leftImageArray.shape[0])

    rightImageArray = numpy.zeros((240, 320, 3), dtype=numpy.uint8)
    rightYarpImage = yarp.ImageRgb()
    rightYarpImage.resize(320,240)
    rightYarpImage.setExternal(rightImageArray, rightImageArray.shape[1], rightImageArray.shape[0])


def readImagesFromCameras( imageCounter ):
    plt.cla()
    plt.axis('off')
    leftInputPort.read(leftYarpImage)
    plt.imshow(leftImageArray)
    filename = '../database/left_cam/png/left_cam_' + str(imageCounter) + '.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=50)
    Image.open('../database/left_cam/png/left_cam_' + str(imageCounter) + '.png').save('../database/left_cam/jpg/left_cam_' + str(imageCounter) + '.jpg', 'JPEG')
    camerasInfoBottle.clear()
    camerasInfoBottle.addString(filename)
    
    plt.cla()
    plt.axis('off')
    rightInputPort.read(rightYarpImage)
    plt.imshow(rightImageArray)
    filename = '../database/right_cam/png/right_cam_' + str(imageCounter) + '.png'
    plt.savefig(filename,bbox_inches='tight', pad_inches=0, dpi=50)
    Image.open('../database/right_cam/png/right_cam_' + str(imageCounter) + '.png').save('../database/right_cam/jpg/right_cam_' + str(imageCounter) + '.jpg', 'JPEG')
    camerasInfoBottle.addString(filename)

    outputCamerasRealTime.write(camerasInfoBottle)
#    time.sleep(1)


# initialise Yarp
yarp.Network.init()

interfaceProcess = False;

if( createPorts() ):
    print "Ports successfully created"
    if( connectPorts() ):
        print "Ports successfully connected"
        interfaceProcess = True;
    else:
        print "Error in connecting ports"
        interfaceProcess = False;
else:
    print "Error in creating ports"
    interfaceProcess = False;

if( interfaceProcess ):
    createImageArrays()

counter = 0
while( interfaceProcess ):
    print "Starting reading images"
    readImagesFromCameras(counter)
    counter = counter + 1

print "Images process finish"


disconnectPorts()

