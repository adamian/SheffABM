/*
 *  SAM skeleton
 *  author: Uriel Martinez
 *  date: April, 2015
 */

#ifndef __SAM_H__
#define __SAM_H__

#include <iostream>

// Include YARP
#include <yarp/os/Network.h>
#include <yarp/os/RFModule.h>
#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <yarp/dev/all.h>

using namespace std;
using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::dev;

//class SAM: public RFModule
class SAM: public RFModule
{
    private:
        Port    rcpPort;

        // input ports
        BufferedPort<Bottle>    languageInputPort;
        BufferedPort<Bottle>    faceInputPort;
        BufferedPort<Bottle>    actionsInputPort;
        BufferedPort<Bottle>    visualInputPort;
        BufferedPort<Bottle>    audioInputPort;
        BufferedPort<Bottle>    touchInputPort;
        BufferedPort<Bottle>    motorsInputPort;

        // output ports
        Port    genericOutputPort;
        Port    uncertaintyOutputPort;
        Port    languageOutputPort;
        Port    actionsOutputPort;

    protected:

    public:
        SAM();
        ~SAM();
        bool updateModule();
        bool configure(ResourceFinder &);
        double getPeriod();
        bool close();
        bool interruptModule();
};

#endif  /*__SAM_H__*/
