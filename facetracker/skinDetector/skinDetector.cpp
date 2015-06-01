#include <stdio.h>
//#include <opencv/cv.h>
//#include <opencv/cvaux.h>
//#include <opencv/highgui.h>
#include <yarp/sig/all.h>
#include <yarp/os/all.h>
#include <yarp/dev/all.h>
//#include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include "OpenCVGrabber.h"
#include <opencv2\opencv.hpp>

using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::sig::draw;
using namespace yarp::sig::file;
using namespace cv;
using namespace yarp::dev;
using namespace std;
using std::cout;

void CVtoYarp(cv::Mat MatImage, ImageOf<PixelRgb> & yarpImage);



/*--------------- SKIN SEGMENTATION ---------------*/
int main(int argc, char** argv)
{
	std::string imageInPort;
	std::string vectorOutPort;
	std::string imageOutPort;

	int imgBlurPixels=3;//15; // Number of pixels to smooth over for final thresholding
	int imgMorphPixels=7; //9; // Number pixels to do morphing over Erode dilate etc....
	if(argc >= 3)
	{
		imageInPort = argv[1];
		vectorOutPort = argv[2];
		imageOutPort = argv[3];
	}
	else
	{
		imageInPort = "/skinImage:i";
		vectorOutPort = "/skinVector:o";
		imageOutPort = "/skinImage:o";
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
	RNG rng(12345);
	//VideoCapture cap(0);

	//if(!cap.isOpened()){
	//	cout << "Error:"; return -1;
	//}
	Mat3b frame;
	while(true)//cap.read(frame))
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
			//waitKey(poll);
		}
		else
		{
			ImageOf<PixelRgb> *yarpImage = faceTrack.read();  // read an image
			if (yarpImage!=NULL) 
			{
				step = yarpImage->getRowSize() + yarpImage->getPadding();
				Mat captureframe(yarpImage->height(),yarpImage->width(),CV_8UC3,yarpImage->getRawImage(),step);
				cout << yarpImage->height() << " " << yarpImage->width() << endl;
				// CHANGED HERE TO BGR
				cvtColor(captureframe, captureframe, CV_RGB2BGR);
				imshow("Raw_Yarp_Video",captureframe);
				//frame = captureframe;
				/* THRESHOLD ON HSV*/
				cvtColor(captureframe, frame, CV_BGR2HSV);
				GaussianBlur(frame, frame, Size(7,7), 1, 1);
				//medianBlur(frame, frame, 15);
				for(int r=0; r<frame.rows; ++r){
					for(int c=0; c<frame.cols; ++c) 
						// 0<H<0.25  -   0.15<S<0.9    -    0.2<V<0.95   
						if( (frame(r,c)[0]>5) && (frame(r,c)[0] < 17) && (frame(r,c)[1]>38) && (frame(r,c)[1]<250) && (frame(r,c)[2]>51) && (frame(r,c)[2]<242) ); // do nothing
						else for(int i=0; i<3; ++i)	frame(r,c)[i] = 0;
				}

				/* BGR CONVERSION AND THRESHOLD */
				Mat1b frame_gray;
				cvtColor(frame, frame, CV_HSV2BGR);
				cvtColor(frame, frame_gray, CV_BGR2GRAY);

				//threshold(frame_gray, frame_gray, 20, 255, CV_THRESH_BINARY);
				//imshow("Threshold_Binary", frame_gray);
				//morphologyEx(frame_gray, frame_gray, CV_MOP_ERODE, Mat1b(2,2,1), Point(-1, -1), 3);
				//imshow("Morph_Erode_2_2", frame_gray);
				//morphologyEx(frame_gray, frame_gray, CV_MOP_OPEN, Mat1b(5,5,1), Point(-1, -1), 1);
				//imshow("Morph_Open_5_5", frame_gray);
				//morphologyEx(frame_gray, frame_gray, CV_MOP_CLOSE, Mat1b(15,15,1), Point(-1, -1), 1);
				//imshow("Morph_Close_15_15", frame_gray);
				//medianBlur(frame_gray, frame_gray, imgBlurPixels);

				threshold(frame_gray, frame_gray, 20, 255, CV_THRESH_BINARY);
				imshow("Threshold_Binary", frame_gray);
				morphologyEx(frame_gray, frame_gray, CV_MOP_ERODE, Mat1b(2,2,1), Point(-1, -1), 3);
				imshow("Morph_Erode_2_2", frame_gray);
				morphologyEx(frame_gray, frame_gray, CV_MOP_DILATE, Mat1b(5,5,1), Point(-1, -1), 3);
				imshow("Morph_Dilate_5_5", frame_gray);
				morphologyEx(frame_gray, frame_gray, CV_MOP_OPEN, Mat1b(3,3,1), Point(-1, -1), 1);
				imshow("Morph_Open_3_3", frame_gray);

				//distanceTransform(frame_gray,frame_ttt3,CV_DIST_L2,5);
				////threshold(frame_ttt, frame_ttt, 20, 255, CV_THRESH_BINARY);
				//imshow("Distance_Transform", frame_ttt3);


				/// Detect edges using canny
				int thresh = 100;
				Mat canny_output;
				vector<vector<Point> > contours;
				vector<Vec4i> hierarchy;
				
				Canny( frame_gray, canny_output, thresh, thresh*2, 3 );
				/// Find contours
				findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

				/// Draw contours
				Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
				for( int i = 0; i< contours.size(); i++ )
				{
				   Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
				   drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
				}

				/// Show in a window
				namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
				imshow( "Contours", drawing );



				
				// Do Morphology...............
				morphologyEx(frame_gray, frame_gray, CV_MOP_CLOSE, Mat1b(9,9,1), Point(-1, -1), 1);
				imshow("Morph_Close_9_9", frame_gray);
				medianBlur(frame_gray, frame_gray, imgBlurPixels);
				imshow("Threshold_full", frame_gray);

				// Compare thresholding techniques

				Mat1b frame_ttt;
				Mat frame_ttt2;
				cvtColor(frame, frame_ttt, CV_BGR2GRAY);
				GaussianBlur(frame_ttt,frame_ttt, Size(imgBlurPixels,imgBlurPixels),0,0);
				adaptiveThreshold(frame_ttt,frame_ttt,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY_INV,7,1);
				imshow("Adaptive_threshold",frame_ttt);
				morphologyEx(frame_ttt, frame_ttt, CV_MOP_CLOSE, Mat1b(imgMorphPixels,imgMorphPixels,1), Point(-1, -1), 4);
				imshow("Adaptive_threshold_morph_close",frame_ttt);

				cvtColor(frame, frame, CV_BGR2HSV);
				//resize(frame, frame, Size(), 0.5, 0.5);
				// HSV data -> used to find skin
				imshow("Video",frame);

				// Apply threshold using binary mask to original data
				//captureframe.copyTo(frame_ttt2,frame_ttt);
				//imshow("Skin_only",frame_ttt2);

				captureframe.copyTo(frame_ttt2,frame_gray);
				imshow("Skin_only",frame_ttt2);


				//############ Test watershed
				//Mat markers(frame.rows,frame.cols,CV_32FC1);
				//watershed(frame,markers);

				//imshow("Watershed",markers);

				//#################################################################
				//cvtColor(frame,captureframe,CV_BGR2RGB);
				// Send image to yarp out port
				ImageOf<PixelRgb>& frameOut = imageOut.prepare();
				CVtoYarp(frame_ttt2,frameOut);
				imageOut.write();
				
				waitKey(1);
			}
		}
	}
}
