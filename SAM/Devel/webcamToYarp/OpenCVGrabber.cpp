 // -*- mode:C++; tab-width:4; c-basic-offset:4; indent-tabs-mode:nil -*-

/*
* Copyright (C) 2006 Eric Mislivec and RobotCub Consortium
* Authors: Eric Mislivec and Paul Fitzpatrick
* CopyPolicy: Released under the terms of the LGPLv2.1 or later, see LGPL.TXT
*
*/


/*
* A Yarp 2 frame grabber device driver using OpenCV to implement
* image capture from cameras and AVI files.
*
* Eric Mislivec
*/


// This define prevents Yarp from declaring its own copy of IplImage
// which OpenCV provides as well. Since Yarp's Image class depends on
// IplImage, we need to define this, then include the OpenCV headers
// before Yarp's.
#define YARP_CVTYPES_H_

#include <yarp/dev/Drivers.h>
#include <yarp/dev/FrameGrabberInterfaces.h>
#include <yarp/dev/PolyDriver.h>

#include <yarp/os/ConstString.h>
#include <yarp/os/Property.h>
#include <yarp/os/Searchable.h>
#include <yarp/os/Value.h>
#include <yarp/sig/Image.h>

#include <stdio.h>

#ifdef YARP2_WINDOWS
#include <cv.h>
#include <highgui.h>
#else
#include <opencv/cv.h>
#include <opencv/highgui.h>
#endif

 #include "OpenCVGrabber.h"


 using yarp::dev::DeviceDriver;
 using yarp::dev::DriverCreatorOf;
 using yarp::dev::Drivers;
 using yarp::dev::IFrameGrabberImage;
 using yarp::dev::PolyDriver;

 using yarp::os::ConstString;
 using yarp::os::Property;
 using yarp::os::Searchable;
 using yarp::os::Value;

 using yarp::sig::ImageOf;
 using yarp::sig::PixelRgb;

 using namespace yarp::os;
 using namespace yarp::sig;
 using namespace yarp::dev;


 #define DBG if (0)

 #ifndef CV_CAP_ANY
 #define CV_CAP_ANY (-1)
 #endif

 bool OpenCVGrabber::open(Searchable & config) {
     // Release any previously allocated resources, just in case
     close();

     m_saidSize = false;
     m_saidResize = false;

     // Are we capturing from a file or a camera ?
     ConstString file = config.check("movie", Value(""),
                                     "if present, read from specified file rather than camera").asString();
     fromFile = (file!="");
     if (fromFile) {

         // Try to open a capture object for the file
         m_capture = (void*)cvCaptureFromAVI(file.c_str());
         if (0 == m_capture) {
             printf("Unable to open file '%s' for capture!\n",
                    file.c_str());
             return false;
         }

         // Should we loop?
         m_loop = config.check("loop","if present, loop movie");

     } else {

         m_loop = false;

         int camera_idx =
             config.check("camera",
                          Value(CV_CAP_ANY),
                          "if present, read from camera identified by this index").asInt();

         // Try to open a capture object for the first camera
         m_capture = (void*)cvCaptureFromCAM(camera_idx);
         if (0 == m_capture) {
             printf("Unable to open camera for capture!\n");
             return false;
         }

     }


     // Extract the desired image size from the configuration if
     // present, otherwise query the capture device
     if (config.check("width","if present, specifies desired image width")) {
         m_w = config.check("width", Value(-1)).asInt();
         if (!fromFile && m_w>0) {
             cvSetCaptureProperty((CvCapture*)m_capture,
                                  CV_CAP_PROP_FRAME_WIDTH, m_w);
         }
     } else {
         m_w = (int)cvGetCaptureProperty((CvCapture*)m_capture,
                                         CV_CAP_PROP_FRAME_WIDTH);
     }

     if (config.check("height","if present, specifies desired image height")) {
         m_h = config.check("height", Value(-1)).asInt();
         if (!fromFile && m_h>0) {
             cvSetCaptureProperty((CvCapture*)m_capture,
                                  CV_CAP_PROP_FRAME_HEIGHT, m_h);
         }
     } else {
         m_h = (int)cvGetCaptureProperty((CvCapture*)m_capture,
                                         CV_CAP_PROP_FRAME_HEIGHT);
     }

	 cvSetCaptureProperty((CvCapture*)m_capture, CV_CAP_PROP_CONVERT_RGB, false);
	 cvSetCaptureProperty((CvCapture*)m_capture, CV_CAP_PROP_FPS , 30);

     // Ignore capture properties - they are unreliable
     //    printf("Capture properties: %ld x %ld pixels @ %lf frames/sec.\n",
     //        m_w, m_h, cvGetCaptureProperty(m_capture, CV_CAP_PROP_FPS));

     fprintf(stderr, "-->OpenCVGrabber opened\n");
     // Success!

     // save our configuration for future reference
     m_config.fromString(config.toString());

     return true;

 }


 bool OpenCVGrabber::close() {
     // Release the capture object, the pointer should be set null
     if (0 != m_capture) cvReleaseCapture((CvCapture**)(&m_capture));
     if (0 != m_capture) {
         m_capture = 0; return false;
     } else return true;
 }



 bool OpenCVGrabber::getImage(ImageOf<PixelRgb> & image) {

     //fprintf(stderr, "-->getImage123\n");

     // Must have a capture object
     if (0 == m_capture) {
         image.zero(); return false;
     }

     //fprintf(stderr, "-->HERE1\n");
     // Grab and retrieve a frame, OpenCV owns the returned image
     IplImage * iplFrame = cvQueryFrame((CvCapture*)m_capture);
     //fprintf(stderr, "-->HERE2\n");

     if (0 == iplFrame && m_loop) {
         bool ok = open(m_config);
         if (!ok) return false;
         iplFrame = cvQueryFrame((CvCapture*)m_capture);
     }

     if (0 == iplFrame) {
         image.zero(); return false;
     }

     //fprintf(stderr, "-->HERE3\n");

     // Resize the output image, this should not result in new
     // memory allocation if the image is already the correct size
     image.resize(iplFrame->width, iplFrame->height);

     if (!m_saidSize) {
         printf("Received image of size %dx%d\n",
                image.width(), image.height());
         m_saidSize = true;
     }

     // Get an IplImage, the Yarp Image owns the memory pointed to
     IplImage * iplImage = (IplImage*)image.getIplImage();

     // Copy the captured image to the output image, flipping it if
     // the coordinate origin is not the top left
     if (IPL_ORIGIN_TL == iplFrame->origin)
         cvCopy(iplFrame, iplImage, 0);
     else
         cvFlip(iplFrame, iplImage, 0);

     if (iplFrame->channelSeq[0]=='B') {
         cvCvtColor(iplImage, iplImage, CV_BGR2RGB);
     }

     if (m_w<=0) {
         m_w = image.width();
     }
     if (m_h<=0) {
         m_h = image.height();
     }
     if (fromFile) {
         if (m_w>0&&m_h>0) {
             if (image.width()!=m_w || image.height()!=m_h) {
                 if (!m_saidResize) {
                     printf("Software scaling from %dx%d to %dx%d\n",
                            image.width(), image.height(),
                            m_w, m_h);
                     m_saidResize = true;
                 }
                 image.copy(image,m_w,m_h);
             }
         }
     }

     //DBG printf("%d by %d %s image\n", image.width(), image.height(),
     //           iplFrame->channelSeq);
     // That's it
     return true;

 }


 // End: OpenCVGrabber.cpp