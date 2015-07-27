
#include "speechInteraction.h"


speechInteraction::speechInteraction()
{
}

speechInteraction::~speechInteraction()
{
}

void speechInteraction::triggerBehaviour(int index)
{
    Bottle &outputBottle = triggerBehaviourPort.prepare();
    outputBottle.clear();
    outputBottle.addInt(index+1);
    triggerBehaviourPort.write();	
}

void speechInteraction::sendSpeech(int index)
{
    string outputString = outputVocabs.at(index);
    
    Bottle &outputBottle = outputPort.prepare();
    outputBottle.clear();
    outputBottle.addString(outputString);
    outputPort.write();	
}

bool speechInteraction::matchVocab(string vocab, int *index)
{    
    if( boost::iequals(vocab, "!SIL") )
        return false;
        
    for( int i = 0; i < nVocabs; i++ )
    {
        if( boost::iequals(vocab, inputVocabs.at(i).c_str()) )
        {
            *index = i;
            return true;
        }
    }

    return false;
}

bool speechInteraction::updateModule()
{
    Bottle *inputBottle = inputPort.read();
    string inputString = inputBottle->toString();

    cout << "RECEIVED TEXT: " << inputString << endl;

    inputString.erase(remove(inputString.begin(), inputString.end(), '\"'), inputString.end());
    
    int index;
    if( matchVocab(inputString, &index) )
    {
        if( index == 15 )
            triggerBehaviour(index);
        else
            sendSpeech(index);

    }
    
    return true;
}

bool speechInteraction::configure(ResourceFinder &rf)
{
    Property config;
    config.fromConfigFile(rf.findFile("from").c_str());


    Bottle &nGeneral = config.findGroup("number_of_vocabs");
    nVocabs = nGeneral.find("nvocabs").asInt();
    
    Bottle &inputGeneral = config.findGroup("input_vocabs");
    
    string findVocab = "vocab_";
    ostringstream convert;    

    cout << "INPUT VOCABS" << endl;
    for( int i = 0; i < nVocabs; i++ )
    {
        convert << (i+1);
        findVocab = findVocab + convert.str();
        inputVocabs.push_back(inputGeneral.find(findVocab).asString().c_str());
        cout << findVocab << ": " << inputGeneral.find(findVocab).asString().c_str() << endl;
        findVocab = "vocab_";
        convert.str("");
    }


    Bottle &outputGeneral = config.findGroup("output_vocabs");

    cout << "OUTPUT VOCABS" << endl;
    for( int i = 0; i < nVocabs; i++ )
    {
        convert << (i+1);
        findVocab = findVocab + convert.str();
        outputVocabs.push_back(outputGeneral.find(findVocab).asString().c_str());
        cout << findVocab << ": " << outputGeneral.find(findVocab).asString().c_str() << endl;
        findVocab = "vocab_";
        convert.str("");
    }
           

    Bottle &portsGeneral = config.findGroup("ports");
    
    inputPortName = portsGeneral.find("input_port").asString().c_str();
    outputPortName = portsGeneral.find("output_port").asString().c_str();
    triggerBehaviourPortName = portsGeneral.find("behaviour_port").asString().c_str();

	inputOpen = inputPort.open(inputPortName.c_str());
	outputOpen = outputPort.open(outputPortName.c_str());
	behaviourPortOpen = triggerBehaviourPort.open(triggerBehaviourPortName.c_str());

	if(!outputOpen || !inputOpen || !behaviourPortOpen)
	{
		cout << "Could not open ports. Exiting" << endl;
		return false;
	}

    return true;
}

bool speechInteraction::interruptModule()
{
    return true;
}

double speechInteraction::getPeriod()
{
    return 0.1;
}


