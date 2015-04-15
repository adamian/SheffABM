/*
 *  SAM skeleton
 *  author: Uriel Martinez
 *  date: April, 2015
 */


#include "sam.h"

int main(int argc, char **argv)
{
    Network yarp;

    if (!yarp.checkNetwork())
    {
        cout << "ERROR: yarpserver not found" << endl;
        return false;
    }

    ResourceFinder rf;
    rf.setVerbose(true);
    rf.setDefaultContext("wysiwyd_sheffield");
    rf.setDefaultConfigFile("config.ini");
    rf.setDefault("name","sam");
    rf.configure("wysiwyd_sheffield_root",argc,argv);

    SAM mod;

    return mod.runModule(rf);

}
