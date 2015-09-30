#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>
#include <yarp/sig/all.h>
#include <yarp/os/all.h>
#include <yarp/dev/all.h>
#include <ctime>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/gpumat.hpp>
#include <iostream>
#include <math.h>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <ctype.h>

using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::sig::draw;
using namespace yarp::sig::file;
using namespace cv;
using namespace yarp::dev;
using namespace std;
using namespace cv::gpu;


#define maxObjects  10

struct MouseParams
{
    Rect selection;
    Point origin;
    bool selectObject;
    int trackObject;
    bool objectReady;
};

Mat frame;
Mat image;

float hranges[] = {0,180};
const float *phranges = hranges;
Mat histimg = Mat::zeros(200, 320, CV_8UC3);
int currentObject = 0;

vector<MouseParams> *mp;

static void mouseHandler(int event, int x, int y, int, void *)
{
    if( mp->at(currentObject).selectObject )
    {
        mp->at(currentObject).selection.x = MIN(x, mp->at(currentObject).origin.x);
        mp->at(currentObject).selection.y = MIN(y, mp->at(currentObject).origin.y);
        mp->at(currentObject).selection.width = std::abs(x - mp->at(currentObject).origin.x);
        mp->at(currentObject).selection.height = std::abs(y - mp->at(currentObject).origin.y);

        mp->at(currentObject).selection &= Rect(0, 0, image.cols, image.rows);
    }

    switch( event )
    {
        case EVENT_LBUTTONDOWN:
            mp->at(currentObject).origin = Point(x,y);
            mp->at(currentObject).selection = Rect(x,y,0,0);
            mp->at(currentObject).selectObject = true;
            break;
        case EVENT_LBUTTONUP:
            mp->at(currentObject).selectObject = false;
            if( mp->at(currentObject).selection.width > 0 && mp->at(currentObject).selection.height > 0 )
                mp->at(currentObject).trackObject = -1;
            break;
    }
}

class trackCamShift: public RFModule
{
    private:
        BufferedPort< ImageOf<PixelRgb> > inputCamera;
        Port outputObjects;
        Mat captureFrameBGR;

        int count;
        int step;
       	int inCount;

        VideoCapture cap;
        int camNum;

        int threadNumber;
        Mat hsv[maxObjects];
        Mat hue[maxObjects];
        Mat mask[maxObjects];
        Mat hist[maxObjects];
        Mat backproj[maxObjects];

        int vmin;
        int vmax;
        int smin;

        Rect trackWindow[maxObjects];
        RotatedRect trackBox[maxObjects];
        int hsize;

        bool backprojMode;

        vector<string> inputObjects;
        int nObjects;

    public:
        trackCamShift()
        {
            threadNumber = 0;
            vmin = 10;
            vmax = 256;
            smin = 30;
            hsize = 16;
            camNum = 0;

            backprojMode = false;
            mp = new vector<MouseParams>(maxObjects);

        }
        ~trackCamShift()
        {
        }
        void trackObject(int object)
        {
            cvtColor(image, hsv[object], COLOR_BGR2HSV);

            if( mp->at(object).trackObject )
            {
                int _vmin = vmin, _vmax = vmax;

                inRange(hsv[object], Scalar(0, smin, MIN(_vmin,_vmax)), Scalar(180, 256, MAX(_vmin, _vmax)), mask[object]);
                int ch[] = {0, 0};
                hue[object].create(hsv[object].size(), hsv[object].depth());
                mixChannels(&hsv[object], 1, &hue[object], 1, ch, 1);

                if( mp->at(object).trackObject < 0 )
                {
                    Mat roi(hue[object], mp->at(object).selection), maskroi(mask[object], mp->at(object).selection);
                    calcHist(&roi, 1, 0, maskroi, hist[object], 1, &hsize, &phranges);
                    normalize(hist[object], hist[object], 0, 255, NORM_MINMAX);

                    trackWindow[object] = mp->at(object).selection;
                    mp->at(object).trackObject = 1;

                    histimg = Scalar::all(0);
                    int binW = histimg.cols / hsize;
                    Mat buf(1, hsize, CV_8UC3);
                    for( int i = 0; i < hsize; i++ )
                        buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
                    cvtColor(buf, buf, COLOR_HSV2BGR);

                    for( int i = 0; i < hsize; i++ )
                    {
                        int val = saturate_cast<int>(hist[object].at<float>(i)*histimg.rows/255);
                        rectangle( histimg, Point(i*binW,histimg.rows), Point((i+1)*binW,histimg.rows - val), Scalar(buf.at<Vec3b>(i)), -1, 8 );
                    }
                }

                calcBackProject(&hue[object], 1, 0, hist[object], backproj[object], &phranges);
                backproj[object] &= mask[object];
                trackBox[object] = CamShift(backproj[object], trackWindow[object], TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
                if( trackWindow[object].area() <= 1 )
                {
                    int cols = backproj[object].cols, rows = backproj[object].rows, r = (MIN(cols, rows) + 5)/6;
                    trackWindow[object] = Rect(trackWindow[object].x - r, trackWindow[object].y - r, trackWindow[object].x + r, trackWindow[object].y + r) & Rect(0, 0, cols, rows);
                }

                if( backprojMode )
                    cvtColor( backproj[object], image, COLOR_GRAY2BGR );
                ellipse( image, trackBox[object], Scalar(0,0,255), 3, 8 );

                mp->at(object).objectReady = true;
           }

            if( mp->at(object).selectObject && mp->at(object).selection.width > 0 && mp->at(object).selection.height > 0 )
            {
                Mat roi(image, mp->at(object).selection);
                bitwise_not(roi, roi);
            }

        }

        bool updateModule()
        {
	        inCount = inputCamera.getInputCount();
            if(inCount == 0)
            {
	            cout << "Awaiting input and output connections" << endl;
            }
            else
            {
                ImageOf<PixelRgb> *yarpImage = inputCamera.read();

	            if (yarpImage!=NULL) 
	            {

		            count = 0;
		            step = yarpImage->getRowSize() + yarpImage->getPadding();
		            Mat captureFrameRaw(yarpImage->height(),yarpImage->width(),CV_8UC3,yarpImage->getRawImage(),step);
		            cvtColor(captureFrameRaw,captureFrameBGR,CV_RGB2BGR);

                    captureFrameBGR.copyTo(image);

    			    Bottle outputObjectPositions;
    			    outputObjectPositions.clear();

                    for( int i = 0; i <= currentObject; i++ )
                    {
                        trackObject(i);
            
                        imshow( "CamShift 1", image);
                        char c = (char)waitKey(5);

                        if( mp->at(i).objectReady == true )
                        {
    		                outputObjectPositions.addDouble(trackBox[i].center.x);
    		                outputObjectPositions.addDouble(trackBox[i].center.y);
    		                outputObjectPositions.addDouble(1.0);
    		                outputObjectPositions.addString(inputObjects.at(i));    		                
                        }
                    }

                    if( mp->at(0).objectReady == true )
           			    outputObjects.write(outputObjectPositions);

                    if( mp->at(currentObject).objectReady == true )
                    {             
                        if( currentObject < (maxObjects-1) )
                        {
                            currentObject++;
                            mp->at(currentObject).selectObject = false;
                            mp->at(currentObject).trackObject = 0;
                            mp->at(currentObject).objectReady = false;
                        }
                    }
                }
            }

            return true;
        }

        bool configure(ResourceFinder &rf)
        {
            Property config;
            config.fromConfigFile(rf.findFile("from").c_str());


            Bottle &nGeneral = config.findGroup("number_of_objects");
            nObjects = nGeneral.find("nobjects").asInt();
            
//            maxObjects = nObjects
            
            Bottle &inputGeneral = config.findGroup("object_names");
            
            string findObject = "object_";
            ostringstream convert;    

            cout << "INPUT OBJECTS" << endl;
            for( int i = 0; i < nObjects; i++ )
            {
                convert << (i+1);
                findObject = findObject + convert.str();
                inputObjects.push_back(inputGeneral.find(findObject).asString().c_str());
                cout << findObject << ": " << inputGeneral.find(findObject).asString().c_str() << endl;
                findObject = "object_";
                convert.str("");
            }

            cout << "Object labels" << endl;                       
            for( int i = 0; i < nObjects; i++ )
                cout << inputObjects.at(i) << endl;

	        bool inOpen = inputCamera.open("/objectTracker/image:i");
            bool outOpen = outputObjects.open("/objectTracker/positions:o");
	        if(!inOpen || !outOpen)
	        {
		        cout << "Could not open ports. Exiting" << endl;
		        return false;
	        }

            namedWindow("Histogram", 0);
            namedWindow("CamShift 1", 0);

            mp->at(currentObject).selectObject = false;
            mp->at(currentObject).trackObject = 0;
            mp->at(currentObject).objectReady = false;

            setMouseCallback("CamShift 1", mouseHandler, (void *)mp);

            return true;
        }

        bool interruptModule()
        {
            return true;
        }

        double getPeriod()
        {
            return 0.01;
        }
};



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
    
    trackCamShift mod;

    return mod.runModule(rf);
}

