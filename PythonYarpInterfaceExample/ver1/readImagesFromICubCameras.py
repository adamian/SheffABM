# Read images from iCub cameras

import numpy
import yarp
#import time
import Image
import matplotlib.pyplot as plt
import cv2 as cv
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

global imgWidth
global imgHeight

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
    if not( yarp.Network.connect("/icub/cam/left", "/interface/cam/left") ):
        return False

    if not( yarp.Network.connect("/icub/cam/right", "/interface/cam/right") ):
        return False

    return True

# disconnect ports from iCub cameras
def disconnectPorts():
    yarp.Network.disconnect("/icub/cam/left", "/interface/cam/left")
    yarp.Network.disconnect("/icub/cam/right", "/interface/cam/right")
    return True

def createImageArrays():
    global leftImageArray
    global leftYarpImage
    global rightImageArray
    global rightYarpImage
    global imgWidth
    global imgHeight  
    

    leftImageArray = numpy.zeros((imgHeight, imgWidth, 3), dtype=numpy.uint8)
    leftYarpImage = yarp.ImageRgb()
    leftYarpImage.resize(imgWidth,imgHeight)
    leftYarpImage.setExternal(leftImageArray, leftImageArray.shape[1], leftImageArray.shape[0])

    rightImageArray = numpy.zeros((imgHeight, imgWidth, 3), dtype=numpy.uint8)
    rightYarpImage = yarp.ImageRgb()
    rightYarpImage.resize(imgWidth,imgHeight)
    rightYarpImage.setExternal(rightImageArray, rightImageArray.shape[1], rightImageArray.shape[0])


def readImagesFromCameras( imageCounter ):
    global imgWidth
    global imgHeight
#    global leftImageArray
#    global leftYarpImage
#    global rightImageArray
#    global rightYarpImage
#    global leftInputPort
#    global rightInputPort
    plt.cla()
    plt.axis('off')
    leftInputPort.read(leftYarpImage)
    plt.imshow(leftImageArray)
    # filename = '../database/left_cam/png/left_cam_' + str(imageCounter) + '.png'
    filename = '../database/left_cam/jpg/left_cam_' + str(imageCounter) + '.jpg'
    cv.imwrite(filename, leftImageArray[:,:,(2,1,0)]) 
    #plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=50)
    #Image.open('../database/left_cam/png/left_cam_' + str(imageCounter) + '.png').save('../database/left_cam/jpg/left_cam_' + str(imageCounter) + '.jpg', 'JPEG')
    camerasInfoBottle.clear()
    camerasInfoBottle.addString(filename)
    
    plt.cla()
    plt.axis('off')
    rightInputPort.read(rightYarpImage)
    plt.imshow(rightImageArray)
    # filename = '../database/right_cam/png/right_cam_' + str(imageCounter) + '.png'
    filename = '../database/right_cam/jpg/right_cam_' + str(imageCounter) + '.jpg'
    
#    cvImage = cv.cvCreateImage(cv.cvSize(imgWidth,imgHeight),cv.IPL_DEPTH_8U, 3 );
#    cv.cvCvtColor(rightYarpImage, cvImage, cv.CV_RGB2BGR);    
    cv.imwrite(filename, rightImageArray[:,:,(2,1,0)])
    #plt.savefig(filename,bbox_inches='tight', pad_inches=0, dpi=50)
    #Image.open('../database/right_cam/png/right_cam_' + str(imageCounter) + '.png').save('../database/right_cam/jpg/right_cam_' + str(imageCounter) + '.jpg', 'JPEG')
    camerasInfoBottle.addString(filename)

    outputCamerasRealTime.write(camerasInfoBottle)
#    time.sleep(1)


# initialise Yarp
yarp.Network.init()

interfaceProcess = False;

imgWidth=320
imgHeight=240


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
#while( interfaceProcess ):
while(counter<50):
    print "Starting reading images"
    readImagesFromCameras(counter)
    counter = counter + 1

print "Images process finish"


disconnectPorts()

