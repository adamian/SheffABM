
#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <yarp/dev/all.h>
#include "visionDriver.h"

using namespace std;
using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::dev;


int main(int argc, char **argv)
{
    Network yarp;

    if( !yarp.checkNetwork() )
    {
        cout << "yarp server not found..." << endl;
        return 1;
    }

    ResourceFinder rf;
    rf.setVerbose(true);
    rf.configure(argc,argv);
    
    visionDriver mod;

    return mod.runModule(rf);
}

