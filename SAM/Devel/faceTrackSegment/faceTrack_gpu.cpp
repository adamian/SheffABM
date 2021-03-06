#include "faceTrack_gpu.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
//#include <windows.h>

using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::sig::draw;
using namespace yarp::sig::file;
using namespace cv;
using namespace yarp::dev;
using namespace std;
//using namespace cv::gpu;


int boxScaleFactor = 20; //Additional pixels for box sizing

int faceSize = 400; //pixel resize for face output

Rect checkRoiInImage(Mat src, Rect roi);
void CVtoYarp(Mat MatImage, ImageOf<PixelRgb> & yarpImage);
Mat skinDetect(Mat captureframeSkin, bool verboseSelect);
Mat segmentEllipse(Mat srcImage, Mat maskImage, bool displayFaces);

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

	bool displayFaces = false; // Select to show all output images.... There will be many!!!!


	if(argc >= 3)
	{
		imageInPort = argv[1];
		vectorOutPort = argv[2];
		imageOutPort = argv[3];
	}
	else
	{
		imageInPort = "/faceImage:i";
		vectorOutPort = "/faceVector:o";
		imageOutPort = "/faceImage:o";
	}  

	isGPUavailable = gpu::getCudaEnabledDeviceCount();

	if (isGPUavailable == 0)
	{
		cout << "No GPU found or the library is compiled without GPU support" << endl;
		cout << "Exiting" << endl;

		return 0;
	}
	else
	{
		
		gpu::getDevice();
		cout << "Proceeding on GPU" << endl;
	}

	
	Network yarp;
	BufferedPort< ImageOf<PixelRgb> > faceTrack;	
	BufferedPort< yarp::sig::Vector > targetPort;	//init output port
	BufferedPort< ImageOf<PixelRgb> > imageOut;

	Port gazePort;	//x and y position for gaze controller
    //Port syncPort;

	bool inOpen = faceTrack.open(imageInPort.c_str());
	bool outOpen = targetPort.open(vectorOutPort.c_str());
	bool imageOutOpen = imageOut.open(imageOutPort.c_str());

	bool gazeOut = gazePort.open("/gazePositionControl:o");
	//bool syncPortIn = syncPort.open("/faceTracker/syncPort:i");

    //Bottle syncBottleIn, syncBottleOut;

	//syncBottleOut.clear();
	//syncBottleOut.addString("stat");

	if(!inOpen | !outOpen | !imageOutOpen | !gazeOut )
	{
		cout << "Could not open ports. Exiting" << endl;
		return -3;
	}

	int inCount = faceTrack.getInputCount();
	int outCount = faceTrack.getOutputCount();

	Mat vectArr, captureFrameBGR,captureFrameRect;		
	gpu::GpuMat captureFrameGPU, grayscaleFrameGPU, objBufGPU;		
	int step = 0, maxSize = 0, biggestFace = 0, count = 0, noFaces;
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
	

	gpu::CascadeClassifier_GPU face_cascade;
	// Check if file exists
	//char filename[]="D:/robotology/install/haarcascade_frontalface_alt.xml";


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

	//face_cascade.load(filename);
	//face_cascade.load("D:/robotology/install/haarcascade_frontalface_alt.xml");

	face_cascade.load("/home/icub/Downloads/facetracker/faceTracking/haarcascade_frontalface_alt.xml");


	while(true)
	{
		inCount = faceTrack.getInputCount();
		outCount = targetPort.getOutputCount();
		if(inCount == 0 || outCount == 0)
		{
			cout << "Awaiting input and output connections" << endl;
			//Sleep(pollTime);
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
					cvtColor(captureFrameRaw,captureFrameBGR,CV_RGB2BGR);
					
                    // Get height and width of original image
                    //cout << "Height " << yarpImage->height() << "Width " << yarpImage->width() << endl;
                    int height = yarpImage->height();
                    int width = yarpImage->width();
					
					captureFrameGPU.upload(captureFrameBGR);
					cv::gpu::cvtColor(captureFrameGPU,grayscaleFrameGPU,CV_BGR2GRAY);
					cv::gpu::equalizeHist(grayscaleFrameGPU,grayscaleFrameGPU);
						
					noFaces = face_cascade.detectMultiScale(grayscaleFrameGPU,objBufGPU,1.2,5,Size(30,30));

					// LB addition here....
					// Detect skin and overlay face extracted region once detected
					Mat skinImage;
					skinImage = skinDetect(captureFrameBGR, displayFaces);
					// Display skin image if on
					//if(displayFaces) imshow("Skin only",skinImage);

					if(noFaces != 0)
					{

						
						cout << noFaces << endl;
                        // copy in last skin image
						captureFrameRect=skinImage.clone();
						std::vector<cv::Mat> faceVec;
						std::vector<cv::Mat> faceVecSkin;
						
						noFaces = 1;

						Mat vecSizes = Mat::zeros(noFaces,1,CV_16UC1);
						Mat allFaces(faceSize,1,CV_8UC3,count);
                        Mat allFacesSkin(faceSize,1,CV_8UC3,count);
                        
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
                            
                            // LB - expand rectangle using additional pixels in boxScaleFactor
                            if (boxScaleFactor != 0)
                            {
                            facesOld[i].x=facesOld[i].x-boxScaleFactor;
                            facesOld[i].y=facesOld[i].y-boxScaleFactor;
                            facesOld[i].width=facesOld[i].width+(boxScaleFactor*2);
                            facesOld[i].height=facesOld[i].height+(boxScaleFactor*2);
                            // LB - Check the extra sizes are not outside the original image size
                            // WARNING -> MIGHT produce distortions -> could reject image instead...
                            facesOld[i]=checkRoiInImage(captureFrameRaw, facesOld[i]); // LB: seg fault (need to pass rect inside of vector...)
                            
                            //if (facesOld[i].x<0) facesOld[i].x=0;
                            //if (facesOld[i].y<0) facesOld[i].y=0;
                            //if ((facesOld[i].width+facesOld[i].x)>width) facesOld[i].width=width-facesOld[i].x;
                            //if ((facesOld[i].height+facesOld[i].y)>height) facesOld[i].height=height-facesOld[i].y;
                            
                            //cout << "Img Height " << yarpImage->height() << "Img Width " << yarpImage->width() << endl;
                            //cout << "x " << facesOld[i].x << "y " << facesOld[i].y << endl;
                            //cout << "Roi Height " << facesOld[i].height << "Roi Width " << facesOld[i].width << endl;
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
							    // Standard image facedetector, take original image
								Mat temp = captureFrameBGR.operator()(facesOld[i]).clone();
								resize(temp,temp,Size(faceSize,faceSize));
								faceVec.push_back(temp);
								// LB processed skin segmented data
								Mat temp2 = skinImage.operator()(facesOld[i]).clone();
								resize(temp2,temp2,Size(faceSize,faceSize));
								faceVecSkin.push_back(temp2);
								
							}
						}
						//hconcat(faceVec,allFaces); // LB original code -> segmented face from original data
						hconcat(faceVec,allFaces);					
                        hconcat(faceVecSkin,allFacesSkin);
                        
						if( displayFaces )
						{
							imshow("faces",allFaces);
							imshow("faces Skin",allFacesSkin);
//							imshow("wholeImage",captureFrameRect);
						}

                        // LB: Test Ellipse extraction of face....
                        //int ttt=segmentEllipse(skinImage);
                        Mat faceSegmented=segmentEllipse(allFaces,allFacesSkin,displayFaces); 
                        //cout << "Is face seg empty: " <<  faceSegmented.empty() << endl;
                        //LB Check face was found!
                        if (!faceSegmented.empty())
                        {
                        // Resize to standard
                        resize(faceSegmented,faceSegmented,Size(faceSize,faceSize));
                        CVtoYarp(faceSegmented,faceImages);
                        imageOut.write();
                        cout << "Sending face to output port" << endl;
                        }
                        else
                        {
                        cout << " Face segmentation unsuccessful" << endl;
                        }
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
//						CVtoYarp(faceSegmented,faceImages);
						//CVtoYarp(allFaces,faceImages); //LB original


//							syncPort.write(syncBottleOut, syncBottleIn);
//                            syncBottleIn = syncPort.read(false);
//							cout << "SYNC BOTTLE: " << syncBottleIn.get(0).asString() << endl;
//                            if( syncBottleIn.toString().c_str() == "sam_ready" )
//							{
//								cout << "SENDING IMAGE TO SAM_PYTHON" << endl;
//						imageOut.write();
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
