#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>
#include <yarp/sig/all.h>
#include <yarp/os/all.h>
#include <yarp/dev/all.h>
#include <ctime>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <boost/algorithm/string.hpp>

using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::sig::draw;
using namespace yarp::sig::file;
using namespace yarp::dev;
using namespace std;


class speechInteraction: public RFModule
{
    private:
        vector<string> inputVocabs;
        vector<string> outputVocabs;
        
	    BufferedPort<Bottle> outputPort;
	    BufferedPort<Bottle> inputPort;
	    BufferedPort<Bottle> triggerBehaviourPort;

	    string inputPortName;	   	    
	    string outputPortName;	   	    
        string triggerBehaviourPortName;
        bool outputOpen;
        bool inputOpen;
        bool behaviourPortOpen;
        int nVocabs;

    public:
        speechInteraction();
        ~speechInteraction();
        bool updateModule();
        bool configure(ResourceFinder &);
        bool interruptModule();
        double getPeriod();
        bool matchVocab(string, int *);
        void sendSpeech(int);
        void triggerBehaviour(int);
};

