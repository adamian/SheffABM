// -*- mode:C++; tab-width:4; c-basic-offset:4; indent-tabs-mode:nil -*-
 
 /*
  * Copyright (C) 2006  Eric Mislivec and RobotCub Consortium
  * Authors:  Eric Mislivec and Paul Fitzpatrick
  * CopyPolicy: Released under the terms of the LGPLv2.1 or later, see LGPL.TXT
  *
  */
 
 #define OpenCVGrabber_INC
 
 /*
  * A Yarp 2 frame grabber device driver using OpenCV to implement
  * image capture from cameras and AVI files.
  *
  * written by Eric Mislivec
  *
  * edited by paulfitz
  *
  */
 
 namespace yarp {
     namespace dev {
         class OpenCVGrabber;
     }
 }
 
 #include <yarp/os/Property.h>
 #include <yarp/dev/FrameGrabberInterfaces.h>
 #include <yarp/dev/DeviceDriver.h>
 
 class yarp::dev::OpenCVGrabber : public IFrameGrabberImage, public DeviceDriver
 {
 public:
 
     OpenCVGrabber() : IFrameGrabberImage(), DeviceDriver(),
                       m_w(0), m_h(0), m_capture(0) { ; }
   
     virtual ~OpenCVGrabber() { ; }
   
   
   
     virtual bool open(yarp::os::Searchable & config);
   
     virtual bool close();
   
     virtual bool getImage(yarp::sig::ImageOf<yarp::sig::PixelRgb> & image);
   
   
     inline virtual int height() const { return m_h; }
   
     inline virtual int width() const { return m_w; }
   
 
 protected:
   
     int m_w;
     int m_h;
 
     bool m_loop;
 
     bool m_saidSize;
     bool m_saidResize;
 
     /* reading from file or camera */
     bool fromFile;
 
     void * m_capture;
 
     yarp::os::Property m_config;
 };