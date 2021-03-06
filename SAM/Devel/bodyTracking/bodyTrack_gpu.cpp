#include "bodyTrack_gpu.h"
//#include <opencv/cv.h>
#include <opencv2/opencv.hpp>

using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::sig::draw;
using namespace yarp::sig::file;
using namespace cv;
using namespace yarp::dev;
using namespace std;
using namespace cv::gpu;

void CVtoYarp(cv::Mat MatImage, ImageOf<PixelRgb> & yarpImage);

int boxScaleFactor = 0; //Additional pixels for box sizing

int faceSize = 200; //pixel resize for face output
int sagittalSplit = 0;  // split person in left and right


int main(int argc, char** argv)
{
	std::string imageInPort;
	std::string vectorOutPort;
	std::string imageOutPort;
	std::string hardware;
	std::string format;
	int format_int;

	int isGPUavailable;
	int poll = 500;

	bool displayFaces = true;

	if(argc >= 3)
	{
		imageInPort = argv[1];
		vectorOutPort = argv[2];
		imageOutPort = argv[3];
	}
	else
	{
		imageInPort = "/bodyTrackerImg:i";
		vectorOutPort = "/bodyTracker:coordinatePort:o";
		imageOutPort = "/bodyTrackerImg:o";
		cout << "Running default ports on GPU " << endl;
		//cout << "Not enough arguments. Must provide port name to the input" << endl;
		//cout << "and output ports and specify hardware used (\"CPU\" or \"GPU\")" << endl;
		//return -2;
	}
	
	isGPUavailable = getCudaEnabledDeviceCount();

	if (isGPUavailable == 0)
	{
		cout << "No GPU found or the library is compiled without GPU support" << endl;
		cout << "Proceeding on CPU" << endl;
		cout << "Detecting largest face in view only for performance" << endl;
		return -2;
	}
	else
	{
		cv::gpu::getDevice();
		cout << "Proceeding on GPU" << endl;
	}
	
	
	Network yarp;
	BufferedPort< ImageOf<PixelRgb> > faceTrack;	
	BufferedPort< yarp::sig::Vector > targetPort;	//init output port
	BufferedPort< ImageOf<PixelRgb> > imageOut;

	//Port gazePort;	//x and y position for gaze controller
    //Port syncPort;

	
	bool inOpen = faceTrack.open(imageInPort.c_str());
	bool outOpen = targetPort.open(vectorOutPort.c_str());
	bool imageOutOpen = imageOut.open(imageOutPort.c_str());

	//bool gazeOut = gazePort.open("/gazePositionControl:o");
	//bool syncPortIn = syncPort.open("/faceTracker/syncPort:i");

    //Bottle syncBottleIn, syncBottleOut;

	//syncBottleOut.clear();
	//syncBottleOut.addString("stat");

	if(!inOpen | !outOpen | !imageOutOpen)// | !gazeOut )
	{
		cout << "Could not open ports. Exiting" << endl;
		return -3;
	}

	int inCount = faceTrack.getInputCount();
	int outCount = faceTrack.getOutputCount();
	//poll every 100ms to check if there is an output and an input connection
	//yarp.connect("/out","/read");
	//yarp.connect("/hello","/in");
	//yarp.connect("/imout","/show");

	Mat vectArr, captureFrame_cpu,captureFrame_cpuRect;		
	cv::gpu::GpuMat captureFrame, grayscaleFrame, objBuf;		
	int step = 0, maxSize = 0, biggestFace = 0, count = 0, noFaces;//, faceSize = 200;
	int centrex, centrey, centrex_old, centrey_old, d;
	bool inStatus = true;
	std::vector< cv::Rect > facesOld;

	//initialise image viewers

	if( displayFaces )
	{
		cout << "Displaying additional images" << endl;
		namedWindow("Body",1);
		namedWindow("Whole Image Body",1);
		waitKey(1);
	}		
	
	CascadeClassifier_GPU face_cascade;
//		face_cascade.load("../haarcascade_frontalface_alt.xml");
//		face_cascade.load("/home/icub/Downloads/facetracker/faceTracking/haarcascade_frontalface_alt.xml");
//		face_cascade.load("D:/robotology/SheffABM/facetracker/bodyTracking/haarcascade_frontalface_alt.xml");
//		face_cascade.load("D:/robotology/SheffABM/facetracker/bodyTracking/haarcascade_mcs_upperbody.xml"); // LB best for upper body
	face_cascade.load("/home/icub/SheffABM/facetracker/bodyTracking/haarcascade_mcs_upperbody.xml");

	while(true)
	{
		inCount = faceTrack.getInputCount();
		outCount = targetPort.getOutputCount();
		if(inCount == 0 || outCount == 0)
		{
			cout << "Awaiting input and output connections" << endl;
			waitKey(500); // LB -> added wait 0.5s here to reduce cpu usage when unconnected
		}
		else
		{
			ImageOf<PixelRgb> *yarpImage = faceTrack.read();
			if (yarpImage!=NULL) 
			{
				//Alternative way of creating an openCV compatible image
				//Takes approx twice as much time as uncomented implementation
				//Also generates IplImage instead of the more useable format Mat
				//IplImage *cvImage = cvCreateImage(cvSize(yarpImage->width(), yarpImage->height()), IPL_DEPTH_8U, 3);
				//cvCvtColor((IplImage*)yarpImage->getIplImage(), cvImage, CV_RGB2BGR);
				count = 0;
				step = yarpImage->getRowSize() + yarpImage->getPadding();
				Mat captureFrame_cpuBayer(yarpImage->height(),yarpImage->width(),CV_8UC3,yarpImage->getRawImage(),step);
                
                // Get height and width of original image
                //cout << "Height " << yarpImage->height() << "Width " << yarpImage->width() << endl;
                int height = yarpImage->height();
                int width = yarpImage->width();
                
                
				if(format_int == 0)
				{
					cv::cvtColor(captureFrame_cpuBayer,captureFrame_cpu,CV_RGB2BGR);
				}
				else
				{
					cv::cvtColor(captureFrame_cpuBayer,captureFrame_cpuBayer,CV_RGB2GRAY); //1D bayer image
					cv::cvtColor(captureFrame_cpuBayer,captureFrame_cpu,CV_BayerGB2BGR);	//rgb image out
				}

				if (displayFaces) imshow("Input image body",captureFrame_cpu);

				captureFrame.upload(captureFrame_cpu);
				gpu::cvtColor(captureFrame,grayscaleFrame,CV_BGR2GRAY);
				gpu::equalizeHist(grayscaleFrame,grayscaleFrame);
				
				noFaces = face_cascade.detectMultiScale(grayscaleFrame,objBuf,1.2,5,Size(100,100)); // LB -> Upped to 100x100 to reduce small regions found 
				
				captureFrame_cpu.copyTo(captureFrame_cpuRect);
				
				if(noFaces != 0)
				{
					cout << noFaces << endl;
					std::vector<cv::Mat> faceVec;
					
					noFaces = 1;

					Mat vecSizes = cv::Mat::zeros(noFaces,1,CV_16UC1);
					Mat allFaces(faceSize,1,CV_8UC3,count);

					objBuf.colRange(0,noFaces).download(vectArr);

					Rect* facesNew = vectArr.ptr<Rect>();
					yarp::sig::Vector& posOutput = targetPort.prepare();
					posOutput.resize(noFaces*3); //each face in the list has a number id, x centre and y centre

					ImageOf<PixelRgb>& faceImages = imageOut.prepare();

					for(int i = 0; i<noFaces; i++)
					{
						int numel = int(facesOld.size());
						if(i < numel)
						{
							centrex = facesNew[i].x;
							centrey = facesNew[i].y;
							
							centrex_old = facesOld[i].x;
							centrey_old = facesOld[i].y;

							d = (centrex_old - centrex) + (centrey_old- centrey);
							d = abs(d);

							if(d > 10)
							{
								centrex_old = facesOld[i].x;
								centrey_old = facesOld[i].y;
								facesOld[i] = facesNew[i];
							}
						}		
						else
						{
							centrex_old = facesNew[i].x;
							centrey_old = facesNew[i].y;
							centrex = centrex_old;
							centrey = centrey_old;
							facesOld.push_back(facesNew[i]);
						}
                        // LB - expand rectangle using additional pixels in boxScaleFactor
                        facesOld[i].x=facesOld[i].x-boxScaleFactor;
                        facesOld[i].y=facesOld[i].y-boxScaleFactor;
                        facesOld[i].width=facesOld[i].width+(boxScaleFactor*2);
                        facesOld[i].height=facesOld[i].height+(boxScaleFactor*2);
                        // LB - Check the extra sizes are not outside the original image size
                        // WARNING -> MIGHT produce distortions -> could reject image instead...
                        if (facesOld[i].x<0) facesOld[i].x=0;
                        if (facesOld[i].y<0) facesOld[i].y=0;
                        if ((facesOld[i].width+facesOld[i].x)>width) facesOld[i].width=width-facesOld[i].x;
                        if ((facesOld[i].height+facesOld[i].y)>height) facesOld[i].height=height-facesOld[i].y;
                        
                        //cout << "Img Height " << yarpImage->height() << "Img Width " << yarpImage->width() << endl;
                        //cout << "x " << facesOld[i].x << "y " << facesOld[i].y << endl;
                        //cout << "Roi Height " << facesOld[i].height << "Roi Width " << facesOld[i].width << endl;
                        
                                                        
						vecSizes.at<unsigned short>(i) = facesOld[i].width;

						if(facesOld[i].width > maxSize)
						{
							maxSize = facesOld[i].width;
							biggestFace = i;
						}
						
						//required for rectangle faces in full image view
						Point pt1(facesOld[i].x + facesOld[i].width, facesOld[i].y + facesOld[i].height);
						Point pt2(facesOld[i].x, facesOld[i].y);
							
						//Point pt1(facesOld[i].x + facesOld[i].width - 100, facesOld[i].y + facesOld[i].height - 100);
						//Point pt2(facesOld[i].x + 100, facesOld[i].y + 100);
							
						rectangle(captureFrame_cpuRect,pt1,pt2,Scalar(0,255,0),1,8,0);
						sagittalSplit = int(facesOld[i].x+(facesOld[i].width/2));
						
						line(captureFrame_cpuRect,Point(sagittalSplit,0),Point(sagittalSplit,height),Scalar(0,0,255),1,8,0);

						int base = (i*3);
						posOutput[base] = i;
						posOutput[base+1] = centrex;
						posOutput[base+2] = centrey;


						/*if( i == 0 )
						{
							Bottle posGazeOutput;
							posGazeOutput.clear();
							posGazeOutput.addString("left");
							posGazeOutput.addDouble(centrex);
							posGazeOutput.addDouble(centrey);
							posGazeOutput.addDouble(1.0);

							gazePort.write(posGazeOutput);
						}*/

					}
					Mat indices;
					sortIdx(vecSizes, indices, cv::SORT_EVERY_COLUMN | cv::SORT_DESCENDING);
					
					for(int i = 0; i<noFaces; i++)
					{
						if(facesOld[i].area() != 0)
						{
							Mat temp = captureFrame_cpu.operator()(facesOld[i]).clone();
							cv::resize(temp,temp,Size(faceSize,faceSize));
							faceVec.push_back(temp);
						}
					}
					hconcat(faceVec,allFaces);
					//faceVec.~vector();

					if( displayFaces )
					{
						//cout << "Number faces: " << noFaces << endl;
						if (noFaces>0)
						{
						cv::imshow("Body",allFaces);
						}
						cv::imshow("Whole Image Body",captureFrame_cpuRect);
					}

//							CVtoYarp(allFaces,faceImages);


///////////////////////////////////////////////////////////////////////////////////////////////////////			
/*
					int segX1 = 1;
					int segY1 = 1;
					int segX2 = allFaces.cols;
					int segY2 = allFaces.rows;
					CvRect rectROI = cvRect(segX1, segY1, segX2, segY2);
					Mat bgModel, fgModel;
					Mat result;

					grabCut(allFaces, result, rectROI, bgModel, fgModel, 1, cv::GC_INIT_WITH_RECT);

					cv::rectangle(allFaces, rectROI, cv::Scalar(255,255,255), 1);
					cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
					Mat foreground(allFaces.size(), CV_8UC3, cv::Scalar(255,255,255));
					allFaces.copyTo(foreground, result);

					cv::namedWindow("Segmented image");
					cv::imshow("Segmented image", foreground);
*/
///////////////////////////////////////////////////////////////////////////////////////////////////////			

					CVtoYarp(allFaces,faceImages);
					imageOut.write();
				}

				targetPort.write();
				waitKey(1);
			}
		}
	}
	return 0;
}
