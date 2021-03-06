#include "faceTrack_gpu.h"
#include <opencv/cv.h>

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

int main(int argc, char** argv)
{
	std::string imageInPort;
	std::string vectorOutPort;
	std::string imageOutPort;
	std::string hardware;
	std::string format;
	int format_int;
	int hardware_int;
	int isGPUavailable;
	int poll = 500;

	bool displayFaces = true;

	if(argc >= 5)
	{
		imageInPort = argv[1];
		vectorOutPort = argv[2];
		imageOutPort = argv[3];
		hardware = argv[4];
		if(hardware == "CPU")
		{
			hardware_int = 0;
			cout << "Detecting largest face in view only for performance" << endl;
		}
		else if(hardware == "GPU")
		{
			isGPUavailable = getCudaEnabledDeviceCount();

			if (isGPUavailable == 0)
			{
				cout << "No GPU found or the library is compiled without GPU support" << endl;
				cout << "Proceeding on CPU" << endl;
				cout << "Detecting largest face in view only for performance" << endl;
				hardware_int = 0;
			}
			else
			{
				hardware_int = 1;
				cv::gpu::getDevice();
				cout << "Proceeding on GPU" << endl;
			}
		}
		else
		{
			cout << "Invalid hardware parameter. Supported hardware parameters: CPU or GPU" << endl;
		}

		if(argc == 5)
		{
			format_int = 0;
		}
		else if(argc > 5)
		{
			format = argv[5];
			if(format == "Bayer")
			{
				format_int = 1;
				cout << "Expecting Bayer Image in" << endl;
			}
			else
			{
				format_int = 0;
				cout << "Expecting RGB Image" << endl;
			}
		}
	}
	else if(argc<4)
	{
		hardware_int = 1; //GPU default
		imageInPort = "/faceTrackerImg:i";
		vectorOutPort = "/faceTracker:coordinatePort:o";
		imageOutPort = "/faceTrackerImg:o";
		cout << "Running default ports on GPU " << endl;
		///faceTrackerImg:i /faceTracker:coordinatePort:o /faceTrackerImg:o GPU
		//cout << "Not enough arguments. Must provide port name to the input" << endl;
		//cout << "and output ports and specify hardware used (\"CPU\" or \"GPU\")" << endl;
		//return -2;
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
	//poll every 100ms to check if there is an output and an input connection
	//yarp.connect("/out","/read");
	//yarp.connect("/hello","/in");
	//yarp.connect("/imout","/show");

	if(hardware_int == 1)//"GPU"
	{
		Mat vectArr, captureFrame_cpu,captureFrame_cpuRect;		
		cv::gpu::GpuMat captureFrame, grayscaleFrame, objBuf;		
		int step = 0, maxSize = 0, biggestFace = 0, count = 0, noFaces;//, faceSize = 200;
		int centrex, centrey, centrex_old, centrey_old, d;
		bool inStatus = true;
		std::vector< cv::Rect > facesOld;

		//initialise image viewers

		if( displayFaces )
		{
			namedWindow("faces",1);
			namedWindow("wholeImage",1);
			waitKey(1);
		}		
		
		CascadeClassifier_GPU face_cascade;
//		face_cascade.load("../haarcascade_frontalface_alt.xml");
//		face_cascade.load("/home/icub/Downloads/facetracker/faceTracking/haarcascade_frontalface_alt.xml");
		face_cascade.load("/home/icub/SheffABM/facetracker/faceTracking/haarcascade_frontalface_alt.xml");
//		face_cascade.load("/home/icub/SheffABM/facetracker/faceTracking/haarcascade_upperbody.xml");//LB testing upper body
		
		cout << "loaded haar cascade xml" << endl;

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

						captureFrame.upload(captureFrame_cpu);
						cv::gpu::cvtColor(captureFrame,grayscaleFrame,CV_BGR2GRAY);
						cv::gpu::equalizeHist(grayscaleFrame,grayscaleFrame);
						
						noFaces = face_cascade.detectMultiScale(grayscaleFrame,objBuf,1.2,5,Size(30,30));
						
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
									
								cv::rectangle(captureFrame_cpuRect,pt1,pt2,cvScalar(0,255,0,0),1,8,0); 
								
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
							cv::sortIdx(vecSizes, indices, cv::SORT_EVERY_COLUMN | cv::SORT_DESCENDING);
							
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
								if (noFaces>0) cv::imshow("faces",allFaces);
								cv::imshow("wholeImage",captureFrame_cpuRect);
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
							//	Mat temp = captureFrame_cpu.operator()(facesOld[biggestFace]).clone();
							//	cv::resize(temp,face[0],Size(faceSize,faceSize));
							//	//cv::Rect myROI(1, 0, allFaces.cols-1, allFaces.rows);
							//	//allFaces = allFaces.operator()(myROI);
							//	cv::imshow("faces",face[0]);
							//	cv::imshow("wholeImage",captureFrame_cpu);
							//	CVtoYarp(face[0],faceImages);
							//	imageOut.write();
							//}
						}

						targetPort.write();
						waitKey(1);
					}
			}

		}
	}
	else if(hardware_int == 0) //"CPU" LB: NOT WORKING NEEDS FIXING OR REMOVING
	{
		Mat grayscaleFrame,face,corners;
		int step = 0;
		int noFaces;
		int count = 0;
		bool inStatus = true;

		CascadeClassifier face_cascade;
		face_cascade.load("haarcascade_frontalface_alt.xml");
		std::vector<Rect> faces;

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
				//waitKey(poll);
				waitKey(500); // LB -> added wait 0.5s here to reduce cpu usage when unconnected
			}
			else
			{
				for(int i=0;i<=0;i++)
				{ 
					ImageOf<PixelRgb> *yarpImage = faceTrack.read();  // read an image
					if (yarpImage!=NULL) 
					{
						//Alternative way of creating an openCV compatible image
						//Takes approx twice as much time as uncomented implementation
						//Also generates IplImage instead of the more useable format Mat
						//IplImage *cvImage = cvCreateImage(cvSize(yarpImage->width(), yarpImage->height()), IPL_DEPTH_8U, 3);
						//cvCvtColor((IplImage*)yarpImage->getIplImage(), cvImage, CV_RGB2BGR);
						count = 0;
						step = yarpImage->getRowSize() + yarpImage->getPadding();
						Mat captureFrame(yarpImage->height(),yarpImage->width(),CV_8UC3,yarpImage->getRawImage(),step);

						if(format_int == 0)
						{
							//required for display purposes only. To implement code with explicit requirements 
							//for an input parameter that specifies output or not
							cvtColor(captureFrame,captureFrame,CV_RGB2BGR);
						}
						else
						{
							cvtColor(captureFrame,captureFrame,CV_RGB2GRAY); //1D bayer image
							cv::cvtColor(captureFrame,captureFrame,CV_BayerGB2BGR); //debayer image
						}

						cvtColor(captureFrame,grayscaleFrame,CV_BGR2GRAY);
						cv::equalizeHist(grayscaleFrame,grayscaleFrame);

						face_cascade.detectMultiScale(grayscaleFrame,faces,1.1,3,CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_SCALE_IMAGE,Size(30,30));
						noFaces = int(faces.size());

						yarp::sig::Vector& output = targetPort.prepare();
						output.resize(noFaces*3); //each face in the list has a number id, x centre and y centre
						
						for(int i = 0; i<noFaces; i++)
						{
							//Required for display purposes only
							Point pt1(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
							Point pt2(faces[i].x, faces[i].y);
							cv::rectangle(captureFrame,pt1,pt2,cvScalar(0,255,0,0),1,8,0);

							face = captureFrame.operator()(faces[i]);
							goodFeaturesToTrack(face,corners,25,3,1);

							int centrex = faces[i].x + (faces[i].width/2);
							int centrey = faces[i].y + (faces[i].height/2);
							int base = (i*3);
							output[base] = i;
							output[base+1] = centrex;
							output[base+2] = centrey;
						}
						cv::imshow("outputCapture",face);
						targetPort.write();
						//waitKey(1);
					}
					else
					{
						count++;
						if(count > 20)
						{
							faceTrack.close();
							inStatus = false;
						}
					}
				}
			}
		}
	}
	return 0;
}
