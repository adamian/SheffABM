/*
 *  SAM skeleton
 *  author: Uriel Martinez
 *  date: April, 2015
 */


#include "sam.h"

using namespace std;
using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::dev;

SAM::SAM()
{
}

SAM::~SAM()
{
}

bool SAM::configure(ResourceFinder &rf)
{
    string name = rf.find("name").asString().c_str();
    setName(name.c_str());

    cout << "Name: " << name << endl;

    Property config;
    config.fromConfigFile(rf.findFile("from").c_str());

    Bottle &bGeneral = config.findGroup("general");
    Bottle &bInputPorts = config.findGroup("input_ports");
    Bottle &bOutputPorts = config.findGroup("output_ports");

    if( bGeneral.isNull() || bInputPorts.isNull() || bOutputPorts.isNull() )
    {
        cout << "Error: one or more groups of parameters are missing!" << endl;
        return false;
    }

    // opening input ports
    languageInputPort.open(("/"+name+"/"+bInputPorts.find("language").asString().c_str()+"/i:").c_str());
    faceInputPort.open(("/"+name+"/"+bInputPorts.find("face").asString().c_str()+"/i:").c_str());
    actionsInputPort.open(("/"+name+"/"+bInputPorts.find("actions").asString().c_str()+"/i:").c_str());
    visualInputPort.open(("/"+name+"/"+bInputPorts.find("visual").asString().c_str()+"/i:").c_str());
    audioInputPort.open(("/"+name+"/"+bInputPorts.find("audio").asString().c_str()+"/i:").c_str());
    touchInputPort.open(("/"+name+"/"+bInputPorts.find("touch").asString().c_str()+"/i:").c_str());
    motorsInputPort.open(("/"+name+"/"+bInputPorts.find("motors").asString().c_str()+"/i:").c_str());
    
    // opening output ports
    genericOutputPort.open(("/"+name+"/generic/o:").c_str());
    languageOutputPort.open(("/"+name+"/"+bOutputPorts.find("language").asString().c_str()+"/o:").c_str());
    uncertaintyOutputPort.open(("/"+name+"/"+bOutputPorts.find("uncertainty").asString().c_str()+"/o:").c_str());
    actionsOutputPort.open(("/"+name+"/"+bOutputPorts.find("actions").asString().c_str()+"/o:").c_str());    

    return true;
}

bool SAM::updateModule()
{
    Bottle *languageBottle = languageInputPort.read(false);
    Bottle *faceBottle = faceInputPort.read(false);
    Bottle *actionsBottle = actionsInputPort.read(false);
    Bottle *visualBottle = visualInputPort.read(false);
    Bottle *audioBottle = audioInputPort.read(false);
    Bottle *touchBottle = touchInputPort.read(false);
    Bottle *motorsBottle = motorsInputPort.read(false);

    Bottle genericOutputBottle;

    if( languageBottle != NULL )
    {
        genericOutputBottle.clear();
        genericOutputBottle.add("language command received...");
        genericOutputPort.write(genericOutputBottle);
    }
    if( faceBottle != NULL )
    {
        genericOutputBottle.clear();
        genericOutputBottle.add("face command received...");
        genericOutputPort.write(genericOutputBottle);
    }
    if( actionsBottle != NULL )
    {
        genericOutputBottle.clear();
        genericOutputBottle.add("actions command received...");
        genericOutputPort.write(genericOutputBottle);
    }
    if( visualBottle != NULL )
    {
        genericOutputBottle.clear();
        genericOutputBottle.add("visual command received...");
        genericOutputPort.write(genericOutputBottle);
    }
    if( audioBottle != NULL )
    {
        genericOutputBottle.clear();
        genericOutputBottle.add("audio command received...");
        genericOutputPort.write(genericOutputBottle);
    }
    if( touchBottle != NULL )
    {
        genericOutputBottle.clear();
        genericOutputBottle.add("touch command received...");
        genericOutputPort.write(genericOutputBottle);
    }
    if( motorsBottle != NULL )
    {
        genericOutputBottle.clear();
        genericOutputBottle.add("motors command received...");
        genericOutputPort.write(genericOutputBottle);
    }
      
    return true;
}

double SAM::getPeriod()
{
    return 0.01;
}

bool SAM::close()
{
    languageInputPort.close();
    faceInputPort.close();
    actionsInputPort.close();
    visualInputPort.close();
    audioInputPort.close();
    touchInputPort.close();
    motorsInputPort.close();
    
    genericOutputPort.close();
    languageOutputPort.close();
    uncertaintyOutputPort.close();
    actionsOutputPort.close();

    return true;
}

bool SAM::interruptModule()
{
    languageInputPort.interrupt();
    faceInputPort.interrupt();
    actionsInputPort.interrupt();
    visualInputPort.interrupt();
    audioInputPort.interrupt();
    touchInputPort.interrupt();
    motorsInputPort.interrupt();
    
    genericOutputPort.interrupt();
    languageOutputPort.interrupt();
    uncertaintyOutputPort.interrupt();
    actionsOutputPort.interrupt();

    return true;
}

