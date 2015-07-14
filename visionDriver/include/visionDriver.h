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
//#include "skinDetector.h"

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
	    //string skinMaskOutPort;
	    string hardware;
	    string format;
        string gazeOutPort;
        //string syncPortConf;
        string faceCascadeFile;
        string bodyCascadeFile;
        // LB test
        //string handCascadeFile;
        
	    int format_int;
	    int hardware_int;
	    int isGPUavailable;
	    int poll;
	    bool displayFaces;
	    bool displayBodies;

        
        // 12 value Vector of bodyPartLocations
        // Face (x,y,z), Body (x,y,z), Left Arm (x,y,z), Right arm (x,y,z)...
        std::vector<double> bodyPartLocations;
        bool bodyPosFound; // flag if any body parts are found -> wont transmit if nothing
        
        
        Rect currentFaceRect;
        Mat faceSegMaskInv;
        
        // FLags
        bool faceSegFlag;
        bool bodySegFlag;

	    BufferedPort< ImageOf<PixelRgb> > faceTrack;	
	    BufferedPort< yarp::sig::Vector > targetPort;	//init output port
	    BufferedPort< ImageOf<PixelRgb> > imageOut;
	    //BufferedPort< ImageOf<PixelRgb> > leftArmSkinPort;
	    //BufferedPort< ImageOf<PixelRgb> > rightArmSkinPort;
	    

	    Port gazePort;	//x and y position for gaze controller
        //Port syncPort;

        //Port leftHandPort;
        //Port rightHandPort;
        //string leftHandPortName;
        //string rightHandPortName;
        
        String bodyPartPosName;
        Port bodyPartPosPort;
        
        
        //string leftArmSkinPortName;
       // string rightArmSkinPortName;
        
        std::vector<std::vector<cv::Point> > returnContours;
        std::vector<RotatedRect> armRotatedRects;
	    
	    bool inOpen;
	    bool outOpen;
	    bool imageOutOpen;
	    //bool skinMaskOutOpen;

	    bool gazeOut;
	    //bool syncPortIn;

        //Bottle syncBottleIn;
        //Bottle syncBottleOut;

    	int inCount;
    	int outCount;


    	Mat vectFaceArr;
    	Mat vectBodyArr;
    	// LB testing
    	//Mat vectLeftHandArr;
    	//Mat vectRightHandArr;
    	
        Mat captureFrameBGR;
        Mat captureFrameFace;		
        Mat captureFrameBody;		
	    cv::gpu::GpuMat captureFrameGPU;
        cv::gpu::GpuMat grayscaleFrameGPU;
        cv::gpu::GpuMat objBufFaceGPU;
        cv::gpu::GpuMat objBufBodyGPU;
        // LB testing
        //cv::gpu::GpuMat objBufLeftHandGPU;
        //cv::gpu::GpuMat objBufRightHandGPU;
        
		int step;
        int maxSize;
        int biggestFace;
        int count;
        int noFaces;
        int noBodies;
        int faceSize;
        int bodySize;
		int centrex;
        int centrey;
        int centrex_old;
        int centrey_old;
        int d;
		bool inStatus;
        int boxScaleFactor; //Additional pixels for box sizing
		int neckScaleFactor;// additional neck scale factor for masking the neck region..... basically add pixels south
    	int pollTime;
        int sagittalSplit;  // split person in left and right
        Point bodyCentre; // calc centre of body
    	
    	int imgBlurPixels; //blur pixels for gauss smoothing
		std::vector< cv::Rect > facesOld;
		std::vector< cv::Rect > bodiesOld;

    	CascadeClassifier_GPU face_cascade;
    	CascadeClassifier_GPU body_cascade;
    	
        // LB testing
        //CascadeClassifier_GPU hand_cascade;
        
        visionUtils *utilsObj;
        //skinDetector *detectorObj;
        
        // Detect skin using default values for first go then update.......
		std::vector<int> hsvAdaptiveValues;

        // Arm / hand tracking
        bool firstLeftHandMovement;
        bool firstRightHandMovement;
        Point right_hand_position;
        Point left_hand_position;
        Point previous_right_hand_position;        
        Point previous_left_hand_position;        
        
        // Tracking of Arm points 
        Point2f previousLeftArmPoints;
        Point2f previousRightArmPoints;
        // Index to points used to fix tracking....
        vector<int> leftPointIndex[4];
        vector<int> rightPointIndex[4];
        bool calibratedLeftPoints;
        bool calibratedRightPoints;
        

    public:
        visionDriver();
        ~visionDriver();
        bool updateModule();
        bool configure(ResourceFinder &);
        bool interruptModule();
        double getPeriod();
        void CVtoYarp(Mat, ImageOf<PixelRgb> &);
        Mat addText(string, Mat, Point, Scalar);
};

