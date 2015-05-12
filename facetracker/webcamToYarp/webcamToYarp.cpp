#include "webcamToYarp.h"

using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::sig::draw;
using namespace yarp::sig::file;
using namespace cv;
using namespace yarp::dev;
using namespace std;

int main(int argc, char** argv) 
{	
	Network yarp;

	std::string destName;
	std::string mode;
	std::string srcName;
	int checkConnections = 0;
	int poll = 1000;

	if(argc >= 2)
	{
		srcName = argv[1];
	}
	else
	{	
		cout << "Not enough input arguments" << endl;
		return 0;
	}

	//Open an port to transmit images from
	BufferedPort< ImageOf<PixelRgb> > imagePort;
    bool portOpen  =  imagePort.open(srcName.c_str());

	if(portOpen == false)
	{
		cout<<"No yarp server detected. Exiting ..." <<endl;
		return -2;
	}
	
	//Set properties for which camera is to be opened
	Property camProps;
	camProps.put("camera",0);

	OpenCVGrabber webcam;

	//Open camera set in the Properties field to check if it is accessible
	bool camAvailable = webcam.open(camProps);

	if(camAvailable != true)
	{
		cout << "Camera not available" << endl;
		return -1;
	}
	else
	{
		webcam.close();
		cout << "Camera check: Done" << endl;
	}

	
	bool webcamStatus = false; //webcam status flag: false = closed, true = open
	bool acq;
	bool acqStatus = true;
	int count = 0;
	//This is a slow outer loop checking for connections to the port and
	//releasing hardware and processing resources accordingly
	//Currently loops continuously. To add control statement possibly
	while(acqStatus) 
	{
		//Check for outgoing connections so as not to occupy hardware
		//and resources when there is no receiver.
		//Poll every 100ms
		checkConnections = imagePort.getOutputCount();
		if(checkConnections == 0)
		{
			if(webcamStatus == true)
			{
				webcam.close(); //release webcam if there are no connections
				cout << "Webcam released" << endl;
				webcamStatus = false;
			}
			cout << "Awaiting connection" << endl;
			waitKey(poll);
		}
		else
		{
			if(webcamStatus == false) //Open webcam and check it is available if not already open
			{
				camAvailable = webcam.open(camProps);
				if(camAvailable != true)
				{
					cout << "Camera not available" << endl;
					return -1;
				}
				else
				{
					webcamStatus = true;
					cout << "Webcam connected" << endl;
				}
			}
			//This is a fast inner loop for image acquisition.
			//Loop and get images for 100 frames then check output connections again
			for(int i=0;i<=100;i++) 
			{
				ImageOf<PixelRgb>& captureFrame = imagePort.prepare();

				//Acquire image from webcam
				acq = webcam.getImage(captureFrame);

				if(acq == true)
				{
					//Write image to port
					imagePort.write();
					count = 0;
					acqStatus = true;
				}
				else
				{
					count++;
					if(count > 20)
					{
						acqStatus = false;
					}
				}
			}
		}
	}
	webcam.close();
	cout << "Webcam unexpectedly unavailable. Exiting" << endl;
	
    return 0;
}
