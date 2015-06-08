#include "faceTrack_gpu.h"
#include <opencv/cv.h>
#include <fstream>
#include <iostream>
#include <windows.h>

using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::sig::draw;
using namespace yarp::sig::file;
using namespace cv;
using namespace yarp::dev;
using namespace std;
//using namespace cv::gpu;

void CVtoYarp(Mat MatImage, ImageOf<PixelRgb> & yarpImage);
Mat skinDetect(Mat captureframeSkin, bool verboseSelect);

int main(int argc, char** argv)
{
	std::string imageInPort;
	std::string vectorOutPort;
	std::string imageOutPort;
	std::string hardware;
	std::string format;
	//int format_int=0;
	//int hardware_int;
	int isGPUavailable;
	int pollTime = 500; //wait delay in ms 500 = 0.5s

	bool displayFaces = TRUE;


	if(argc >= 4)
	{
		imageInPort = argv[2];
		vectorOutPort = argv[3];
		imageOutPort = argv[4];
	}
	else
	{
		imageInPort = "/faceTrackerImg:i";
		vectorOutPort = "/faceTracker:coordinatePort:o";
		imageOutPort = "/faceTrackerImg:o";
	}  


	isGPUavailable = cv::gpu::getCudaEnabledDeviceCount();

	if (isGPUavailable == 0)
	{
		cout << "No GPU found or the library is compiled without GPU support" << endl;
		cout << "Exiting" << endl;

		return 0;
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

	Port gazePort;	//x and y position for gaze controller
    Port syncPort;

	bool inOpen = faceTrack.open(imageInPort.c_str());
	bool outOpen = targetPort.open(vectorOutPort.c_str());
	bool imageOutOpen = imageOut.open(imageOutPort.c_str());

	bool gazeOut = gazePort.open("/gazePositionControl:o");
	bool syncPortIn = syncPort.open("/faceTracker/syncPort:i");

    Bottle syncBottleIn, syncBottleOut;

	syncBottleOut.clear();
	syncBottleOut.addString("stat");

	if(!inOpen | !outOpen | !imageOutOpen | !gazeOut )
	{
		cout << "Could not open ports. Exiting" << endl;
		return -3;
	}

	int inCount = faceTrack.getInputCount();
	int outCount = faceTrack.getOutputCount();

	Mat vectArr, captureFrameBGR,captureFrameRect;		
	cv::gpu::GpuMat captureFrameGPU, grayscaleFrameGPU, objBufGPU;		
	int step = 0, maxSize = 0, biggestFace = 0, count = 0, noFaces, faceSize = 200;
	int centrex, centrey, centrex_old, centrey_old, d;
	bool inStatus = true;
	std::vector< Rect > facesOld;

	//initialise image viewers

	if( displayFaces )
	{
		namedWindow("faces",1);
		namedWindow("wholeImage",1);
		waitKey(1);
	}	
	

	cv::gpu::CascadeClassifier_GPU face_cascade;
	// Check if file exists
	char filename[]="D:/robotology/install/haarcascade_frontalface_alt.xml";


	//// LB NEEDS FIXING -> CHECK FOR FILE!!!!!
	//ifstream infile(filename);

	//if (!infile.bad())
	//{

	//	face_cascade.load(filename);
	//}
	//else
	//{
	//	cout << "Cannot load face cascade xml file" << endl;
	//	cout << "Exiting" << endl;
	//	return 0;
	//}

	face_cascade.load(filename);

	//face_cascade.load("/home/icub/Downloads/facetracker/faceTracking/haarcascade_frontalface_alt.xml");
	while(true)
	{
		inCount = faceTrack.getInputCount();
		outCount = targetPort.getOutputCount();
		if(inCount == 0 || outCount == 0)
		{
			cout << "Awaiting input and output connections" << endl;
			Sleep(pollTime);
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
					Mat captureFrameRaw(yarpImage->height(),yarpImage->width(),CV_8UC3,yarpImage->getRawImage(),step);
					cout << "Got here 0" << endl;
					cvtColor(captureFrameRaw,captureFrameBGR,CV_RGB2BGR);
					captureFrameGPU.upload(captureFrameBGR);
					cv::gpu::cvtColor(captureFrameGPU,grayscaleFrameGPU,CV_BGR2GRAY);
					cv::gpu::equalizeHist(grayscaleFrameGPU,grayscaleFrameGPU);
						
					noFaces = face_cascade.detectMultiScale(grayscaleFrameGPU,objBufGPU,1.2,5,Size(30,30));
					cout << "Got here 1" << endl;

					if(noFaces != 0)
					{

						captureFrameBGR.copyTo(captureFrameRect);
						cout << noFaces << endl;
						cout << "Got here 2" << endl;
						// LB addition here....
						// As face has been detected.... Detect skin and overlay face extracted region
						Mat skinImage;
						skinImage = skinDetect(captureFrameRect, displayFaces);
						// Display skin image if on
						if(displayFaces) imshow("Skin only",skinImage);

						captureFrameRect=skinImage;
						cout << "Got here 3" << endl;
						std::vector<cv::Mat> faceVec;
							
						noFaces = 1;

						Mat vecSizes = Mat::zeros(noFaces,1,CV_16UC1);
						Mat allFaces(faceSize,1,CV_8UC3,count);

						objBufGPU.colRange(0,noFaces).download(vectArr);

						Rect* facesNew = vectArr.ptr<Rect>();
						yarp::sig::Vector& posOutput = targetPort.prepare();
						posOutput.resize(noFaces*3); //each face in the list has a number id, x centre and y centre

						ImageOf<PixelRgb>& faceImages = imageOut.prepare();

						for(int i = 0; i<noFaces; i++)
						{
							int numel = facesOld.size();
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
									
							rectangle(captureFrameRect,pt1,pt2,cvScalar(0,255,0,0),1,8,0); 
								
							int base = (i*3);
							posOutput[base] = i;
							posOutput[base+1] = centrex;
							posOutput[base+2] = centrey;


							if( i == 0 )
							{
								Bottle posGazeOutput;
								posGazeOutput.clear();
								posGazeOutput.addString("left");
								posGazeOutput.addDouble(centrex);
								posGazeOutput.addDouble(centrey);
								posGazeOutput.addDouble(1.0);

								gazePort.write(posGazeOutput);
							}

						}
						Mat indices;
						sortIdx(vecSizes, indices, SORT_EVERY_COLUMN | SORT_DESCENDING);
							
						for(int i = 0; i<noFaces; i++)
						{
							if(facesOld[i].area() != 0)
							{
								Mat temp = captureFrameBGR.operator()(facesOld[i]).clone();
								resize(temp,temp,Size(faceSize,faceSize));
								faceVec.push_back(temp);
							}
						}
						hconcat(faceVec,allFaces);
						//faceVec.~vector();

						if( displayFaces )
						{
							imshow("faces",allFaces);
//							imshow("wholeImage",captureFrameRect);
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

						grabCut(allFaces, result, rectROI, bgModel, fgModel, 1, GC_INIT_WITH_RECT);

						rectangle(allFaces, rectROI, Scalar(255,255,255), 1);
						compare(result, GC_PR_FGD, result, CMP_EQ);
						Mat foreground(allFaces.size(), CV_8UC3, Scalar(255,255,255));
						allFaces.copyTo(foreground, result);

						namedWindow("Segmented image");
						imshow("Segmented image", foreground);
*/
///////////////////////////////////////////////////////////////////////////////////////////////////////			

						CVtoYarp(allFaces,faceImages);


//							syncPort.write(syncBottleOut, syncBottleIn);
//                            syncBottleIn = syncPort.read(false);
//							cout << "SYNC BOTTLE: " << syncBottleIn.get(0).asString() << endl;
//                            if( syncBottleIn.toString().c_str() == "sam_ready" )
//							{
//								cout << "SENDING IMAGE TO SAM_PYTHON" << endl;
						imageOut.write();
//							}

//                            syncBottleIn->clear();

							
						//if(facesOld[biggestFace].area() != 0)
						//{
						//	Mat temp = captureFrameBGR.operator()(facesOld[biggestFace]).clone();
						//	resize(temp,face[0],Size(faceSize,faceSize));
						//	//Rect myROI(1, 0, allFaces.cols-1, allFaces.rows);
						//	//allFaces = allFaces.operator()(myROI);
						//	imshow("faces",face[0]);
						//	imshow("wholeImage",captureFrameBGR);
						//	CVtoYarp(face[0],faceImages);
						//	imageOut.write();
						//}
					}

					targetPort.write();
					waitKey(1);
				}
		}

		//if( displayFaces ) imshow("wholeImage",captureFrameRect);

	}

	return 0;
}
