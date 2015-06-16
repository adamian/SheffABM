#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>
#include <yarp/sig/all.h>
#include <yarp/os/all.h>
#include <yarp/dev/all.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ctime>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/gpumat.hpp>
#include <iostream>
#include <math.h>
#include <algorithm>

#include "visionUtils.h"
#include "skinDetector.h"

using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::sig::draw;
using namespace yarp::sig::file;
using namespace cv;
using namespace yarp::dev;
using namespace std;
using namespace cv::gpu;


#define faceLimit 20


class visionDriver: public RFModule
{
    private:
	    string imageInPort;
	    string vectorOutPort;
	    string imageOutPort;
	    string hardware;
	    string format;
        string gazeOutPort;
        string syncPortConf;
        string cascadeFile;
	    int format_int;
	    int hardware_int;
	    int isGPUavailable;
	    int poll;
	    bool displayFaces;

	    BufferedPort< ImageOf<PixelRgb> > faceTrack;	
	    BufferedPort< yarp::sig::Vector > targetPort;	//init output port
	    BufferedPort< ImageOf<PixelRgb> > imageOut;

	    Port gazePort;	//x and y position for gaze controller
        Port syncPort;

	    bool inOpen;
	    bool outOpen;
	    bool imageOutOpen;

	    bool gazeOut;
	    bool syncPortIn;

        Bottle syncBottleIn;
        Bottle syncBottleOut;

    	int inCount;
    	int outCount;

		Mat vectArr;
        Mat captureFrame_cpu;
        Mat captureFrame_cpuRect;		
		cv::gpu::GpuMat captureFrame;
        cv::gpu::GpuMat grayscaleFrame;
        cv::gpu::GpuMat objBuf;

		int step;
        int maxSize;
        int biggestFace;
        int count;
        int noFaces;
        int faceSize;
		int centrex;
        int centrey;
        int centrex_old;
        int centrey_old;
        int d;
		bool inStatus;
        int boxScaleFactor; //Additional pixels for box sizing
    	int pollTime;
		std::vector< cv::Rect > facesOld;

    	CascadeClassifier_GPU face_cascade;

        visionUtils *utilsObj;
        skinDetector *detectorObj;

    public:
        visionDriver();
        ~visionDriver();
        bool updateModule();
        bool configure(ResourceFinder &);
        bool interruptModule();
        double getPeriod();
        void CVtoYarp(Mat, ImageOf<PixelRgb> &);
};

