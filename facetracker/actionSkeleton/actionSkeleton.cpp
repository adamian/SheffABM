#include <stdio.h>
#include <yarp/sig/all.h>
#include <yarp/os/all.h>
#include <yarp/dev/all.h>
#include <opencv2/opencv.hpp>

using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::sig::draw;
using namespace yarp::sig::file;
using namespace cv;
using namespace yarp::dev;
using namespace std;
using std::cout;

void CVtoYarp(Mat, ImageOf<PixelRgb> & yarpImage);
Mat cannySegmentation(Mat, int);
Mat cannySegmentationColour(Mat, int);
Mat skinSegment(Mat);
Mat skinDetect(Mat);
Mat maskImage(Mat, Mat);
Mat skeletonDetect(Mat);
void thin(Mat& img,
          bool need_boundary_smoothing=false,
          bool need_acute_angle_emphasis=false,
          bool destair=false);


bool singleRegionChoice = false; // On = Find single largest region of skin
bool verboseOutput = true; // Turn on to show image processing steps

bool useYarp = true; // Option to use yarp images or load from files



int minPixelSize=2500; // Minimum pixel size for keeping skin regions!
int imgBlurPixels=7;//7, 15; // Number of pixels to smooth over for final thresholding
int imgMorphPixels=3; //7, 9; // Number pixels to do morphing over Erode dilate etc...
RNG rng(12345);

/*--------------- action Skeleton SEGMENTATION ---------------

Run as: "actionSkeleton singleRegionChoice /imageIn /info /imageOut" 

singleRegionChoice: single=1 (default), all regions returned=0 (depends on minPixelSize, =0 is all returned)

Ports:
1. /ImageIn (RGB Yarp image sent) default=/skinImage:i
2. /info (Skin centre x,y and skin found =0/1) default=/skinVector:o
3. /imageOut (Returned image segmnted with skin) default=/skinImage:o

*/
int main(int argc, char** argv)
{

	if (useYarp)
	{
		std::string imageInPort;
		std::string vectorOutPort;
		std::string imageOutPort;
		// 1st Argument, Choose method Adaptive threshold =0 (default) or Binary = 1 
		if(argc >= 3)
		{
			imageInPort = argv[1];
			vectorOutPort = argv[2];
			imageOutPort = argv[3];
		}
		else
		{
			imageInPort = "/skeleImage:i";
			vectorOutPort = "/skeleVector:o";
			imageOutPort = "/skeleImage:o";
		}
		
		if (argc>=4)
		{
    		if (argv[4]=="1") singleRegionChoice=true;
    		cout << "Using single region finder" << endl;
        }


		Network yarp;
		BufferedPort< ImageOf<PixelRgb> > faceTrack;	
		BufferedPort< yarp::sig::Vector > targetPort;	//init output port
		BufferedPort< ImageOf<PixelRgb> > imageOut;

		bool inOpen = faceTrack.open(imageInPort.c_str());
		bool vectorOutOpen = targetPort.open(vectorOutPort.c_str());
		bool imageOutOpen = imageOut.open(imageOutPort.c_str());

		if(!inOpen | !vectorOutOpen | !imageOutOpen)
		{
			cout << "Could not open ports. Exiting" << endl;
			return -3;
		}
	
		int inCount = faceTrack.getInputCount();
		int outCount = faceTrack.getOutputCount();
		bool inStatus = true;
		int step = 0;
	
		while(true)
		{
			inCount = faceTrack.getInputCount();
			outCount = targetPort.getOutputCount();
			if(inCount == 0 || outCount == 0) //convert to periodic checking whilst processing as well
			{
				if(inStatus == false)
				{
					faceTrack.open(imageInPort.c_str());
				}
				cout << "Awaiting input and output connections" << endl;
				waitKey(500); // Wait here to reduce CPU usage 0.5s
			}
			else
			{
				ImageOf<PixelRgb> *yarpImage = faceTrack.read();  // read an image			
				if (yarpImage!=NULL) 
				{
					step = yarpImage->getRowSize() + yarpImage->getPadding();
					Mat captureFrame(yarpImage->height(),yarpImage->width(),CV_8UC3,yarpImage->getRawImage(),step);
					cout << yarpImage->height() << " " << yarpImage->width() << endl;
					// CHANGED HERE TO BGR
					cvtColor(captureFrame, captureFrame, CV_RGB2BGR);
					// skin seg and skeleton
					Mat frame = captureFrame.clone(); // done to allow easier commenting
					// ### 1. skin detect (HSV & filter)		
					frame =	skinDetect(frame);
					// ### 2. skin segment (grayscale 1D image returned)
					Mat frameSkinGray=skinSegment(frame);
					// ### 3. Overlay mask onto original image
					Mat frameSkin = maskImage(captureFrame, frameSkinGray);
					// ### 4. Skeleton images
					// Use original segmented image
					Mat frameGray;
					cvtColor(frameSkin, frameGray, CV_BGR2GRAY);
					Mat skeletonFrame = skeletonDetect(frameGray);
					// ### 5. Recognise arms & hands

					// ### Display output...
					if (verboseOutput) imshow("Seg Skin",frameSkin);
					if (verboseOutput) imshow("Skeleton",skeletonFrame);
					// ### Return images
					// return frameSkin;
					//waitKey(1);

					ImageOf<PixelRgb>& frameOut = imageOut.prepare();
					CVtoYarp(frameSkin,frameOut);
					imageOut.write();
					waitKey(1);
				}
			}
		}
	}
	else // If no yarp use picture from file...
	{
		//Mat captureFrame = imread("D:/robotology/SheffABM/SkinData/Uriel_hands.png");
		Mat captureFrame = imread("/home/icub/SheffABM/facetracker/testActionImages/Uriel_hands.png");
		Mat frame = captureFrame.clone(); // done to allow easier commenting
		// ### 1. skin detect (HSV & filter)		
		// frame =	skinDetect(frame);
		// ### 2. skin segment (grayscale 1D image returned)
		Mat frameSkinGray=skinSegment(frame);
		// ### 3. Overlay mask onto original image
		Mat frameSkin = maskImage(captureFrame, frameSkinGray);
		// ### 4. Skeleton images
		// Use original segmented image
		Mat frameGray;
		cvtColor(frameSkin, frameGray, CV_BGR2GRAY);
		Mat skeletonFrame = skeletonDetect(frameGray);
		// ### 5. Recognise arms & hands

		// ### Display output...
		if (verboseOutput) imshow("Seg Skin",frameSkin);
		if (verboseOutput) imshow("Skeleton",skeletonFrame);
		// ### Return images
		// return frameSkin;
		waitKey(0);
	}
}

// HSV based skin detection
Mat skinDetect(Mat captureFrame)
{
	Mat3b frame;
	// Forcing resize to 640x480 -> all thresholds / pixel filters configured for this size..... 
	resize(captureFrame,captureFrame,Size(640,480));
	cout << "WARNING: resizing images to 640x480 for processing config" << endl;
	if (verboseOutput)	imshow("Raw Yarp Video (A)",captureFrame);
	/* THRESHOLD ON HSV*/
	// HSV data -> used to find skin
	cvtColor(captureFrame, frame, CV_BGR2HSV);
	//cvtColor(captureFrame, frame, CV_BGR2HLS);
	GaussianBlur(frame, frame, Size(imgBlurPixels,imgBlurPixels), 1, 1);
	//medianBlur(frame, frame, 15);
	for(int r=0; r<frame.rows; ++r){
		for(int c=0; c<frame.cols; ++c) 
			// 0<H<0.25  -   0.15<S<0.9    -    0.2<V<0.95   
			if( (frame(r,c)[0]>5) && (frame(r,c)[0] < 17) && (frame(r,c)[1]>38) && (frame(r,c)[1]<250) && (frame(r,c)[2]>51) && (frame(r,c)[2]<242) ); // do nothing
			else for(int i=0; i<3; ++i)	frame(r,c)[i] = 0;
	}
	if (verboseOutput)	imshow("Skin HSV (B)",frame);
	cvtColor(frame, frame, CV_HSV2BGR);
	return frame;
}

// Segment region -> e.g. skin found using hsv
Mat skinSegment(Mat frame)
{	
	/* BGR CONVERSION AND THRESHOLD */
	Mat1b frameGray;
	cvtColor(frame, frameGray, CV_BGR2GRAY);

	// Adaptive thresholding techni
	// 1. Threshold data to find main areas of skin
	adaptiveThreshold(frameGray,frameGray,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY_INV,9,1);
	if (verboseOutput)	imshow("Adaptive_threshold (D1)",frameGray);
	// 2. Fill in thresholded areas
	morphologyEx(frameGray, frameGray, CV_MOP_CLOSE, Mat1b(imgMorphPixels,imgMorphPixels,1), Point(-1, -1), 2);
	// Select single largest region from image, if singleRegionChoice is selected (1)
	if (singleRegionChoice)
	{
	frameGray = cannySegmentation(frameGray, -1);
	}
	else // Detect each separate block and remove blobs smaller than a few pixels
	{
	frameGray = cannySegmentation(frameGray, minPixelSize);
	}

	return frameGray;
}

Mat skeletonDetect(Mat captureFrame)
{

	// Alternative
	//Mat& skelThinned=captureFrame.clone();
	//thin(skelThinned,false,false,false);
	//if (verboseOutput) imshow("Skel erode",skelThinned);

	threshold(captureFrame, captureFrame, 127, 255, THRESH_BINARY);
	if (verboseOutput) imshow("Skel in",captureFrame);
	Mat skel(captureFrame.size(), CV_8UC1, Scalar(0));
	Mat temp;
	Mat eroded;
 
	Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
 
	bool done;		
	do
	{
		erode(captureFrame, eroded, element);
		dilate(eroded, temp, element); // temp = open(captureFrame)
		subtract(captureFrame, temp, temp);
		bitwise_or(skel, temp, skel);
		eroded.copyTo(captureFrame);
		//cout << "Zero count: " << countNonZero(captureFrame) << endl;
		done = (countNonZero(captureFrame) == 0);
	} while (!done);
	if (verboseOutput) imshow("Skel raw",skel);
	// Blur to reduce noise
	GaussianBlur(skel, skel, Size(imgBlurPixels,imgBlurPixels), 1, 1);
	// Find contours use canny seg...
	Mat skelCanny = cannySegmentationColour(skel, 50);

	return skel;
}

Mat maskImage(Mat captureFrame, Mat maskFrame)
{
	// Just return skin
	Mat frameReturn;
	captureFrame.copyTo(frameReturn,maskFrame);  // Copy captureFrame data to frame_skin, using mask from frame_ttt
	if (verboseOutput)	imshow("Skin segmented",frameReturn);
	return frameReturn;
}

Mat cannySegmentation(Mat img0, int minPixelSize)
{
	// Segments items in gray image (img0)
	// minPixelSize=
	// -1, returns largest region only
	// pixels, threshold for removing smaller regions, with less than minPixelSize pixels
	// 0, returns all detected segments

    Mat img1;
    
	// apply your filter
    Canny(img0, img1, 100, 200, 3); //100, 200, 3);

    // find the contours
    vector< vector<Point> > contours;
    findContours(img1, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    // Mask for segmented region
    Mat mask = Mat::zeros(img1.rows, img1.cols, CV_8UC1);

    vector<double> areas(contours.size());

	if (minPixelSize==-1)
	{ // Case of taking largest region
		for(int i = 0; i < contours.size(); i++)
			areas[i] = contourArea(Mat(contours[i]));
		double max;
		Point maxPosition;
		minMaxLoc(Mat(areas),0,&max,0,&maxPosition);
		drawContours(mask, contours, maxPosition.y, Scalar(1), CV_FILLED);
	}
	else
	{ // Case for using minimum pixel size
		for (int i = 0; i < contours.size(); i++)
		{
			if (contourArea(Mat(contours[i]))>minPixelSize)
				//Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
				drawContours(mask, contours, i, Scalar(1), CV_FILLED);

		}
	}
    // normalize so imwrite(...)/imshow(...) shows the mask correctly!
    normalize(mask.clone(), mask, 0.0, 255.0, CV_MINMAX, CV_8UC1);

    // show the images
    if (verboseOutput)	imshow("Canny: Img in", img0);
    if (verboseOutput)	imshow("Canny: Mask", mask);
    if (verboseOutput)	imshow("Canny Output", img1);
    return mask;
}

Mat cannySegmentationColour(Mat img0, int minPixelSize)
{
	// Segments items in gray image (img0)
	// minPixelSize=
	// -1, returns largest region only
	// pixels, threshold for removing smaller regions, with less than minPixelSize pixels
	// 0, returns all detected segments

    Mat img1;
    
	// apply your filter
    //Canny(img0, img1, 100, 200, 3); //100, 200, 3);
	img1=img0.clone();

    // find the contours
    vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;

    findContours(img1, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    // Mask for segmented region
    Mat mask = Mat::zeros(img1.rows, img1.cols, CV_8UC3);

    vector<double> areas(contours.size());

	if (minPixelSize==-1)
	{ // Case of taking largest region
		for(int i = 0; i < contours.size(); i++)
			areas[i] = contourArea(Mat(contours[i]));
		double max;
		Point maxPosition;
		minMaxLoc(Mat(areas),0,&max,0,&maxPosition);
		drawContours(mask, contours, maxPosition.y, Scalar(1), CV_FILLED);
	}
	else
	{ // Case for using minimum pixel size
		Vec4f lines;
		Scalar color;
		for (int i = 0; i < contours.size(); i++)
		{
			if (contourArea(Mat(contours[i]))>minPixelSize)
			{
				color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
				//drawContours(mask, contours, i, color, CV_FILLED);
				drawContours( mask, contours, i, color, 2, 8, hierarchy, 0, Point() );
				fitLine(Mat(contours[i]),lines,2,0,0.01,0.01);
				//lefty = int((-x*vy/vx) + y)
				//righty = int(((gray.shape[1]-x)*vy/vx)+y)
				int lefty = (-lines[2]*lines[1]/lines[0])+lines[3];
				int righty = ((mask.cols-lines[2])*lines[1]/lines[0])+lines[3];
				//line(mask,Point(mask.cols-1,righty),Point(0,lefty),color,2);


				// Find line limits....
				//x,y,w,h = cv2.boundingRect(cnt)
				//cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
				Rect boundingBox;
				boundingBox = boundingRect(Mat(contours[i]));
				// Start 
				//boundingBox.x;
				//boundingBox.x+boundingBox.width;

				int leftBoxy = ((boundingBox.x-lines[2])*lines[1]/lines[0])+lines[3];
				int rightBoxy = (((boundingBox.x+boundingBox.width)-lines[2])*lines[1]/lines[0])+lines[3];

				line(mask,Point((boundingBox.x+boundingBox.width)-1,rightBoxy),Point(boundingBox.x,leftBoxy),color,2);


			}
		}
	}
    // normalize so imwrite(...)/imshow(...) shows the mask correctly!
    normalize(mask.clone(), mask, 0.0, 255.0, CV_MINMAX, CV_8UC1);

    // show the images
    if (verboseOutput)	imshow("Canny color: Img in", img0);
    if (verboseOutput)	imshow("Canny color: Mask", mask);
    if (verboseOutput)	imshow("Canny color Output", img1);
    return mask;

	//vector<Vec2f> lines;
	//HoughLines(dst, lines, 1, CV_PI/180, 100, 0, 0 );

}
