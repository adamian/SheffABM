
#include "visionDriver.h"


visionDriver::visionDriver()
{
	displayFaces = false;
    utilsObj = new visionUtils();
    detectorObj = new skinDetector();
}

visionDriver::~visionDriver()
{
}

bool visionDriver::updateModule()
{
    inCount = faceTrack.getInputCount();
    outCount = targetPort.getOutputCount();
    if(inCount == 0 || outCount == 0)
    {
	    cout << "Awaiting input and output connections" << endl;
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

                    int height = yarpImage->height();
                    int width = yarpImage->width();
			
		    captureFrameGPU.upload(captureFrameBGR);
		    cv::gpu::cvtColor(captureFrameGPU,grayscaleFrameGPU,CV_BGR2GRAY);
		    cv::gpu::equalizeHist(grayscaleFrameGPU,grayscaleFrameGPU);

/*
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
*/
		
		    noFaces = face_cascade.detectMultiScale(grayscaleFrameGPU,objBufGPU,1.2,5,Size(30,30));

			Mat skinImage;
			skinImage = detectorObj->detect(captureFrameBGR, displayFaces);
		
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
                        facesOld[i]=utilsObj->checkRoiInImage(captureFrameRaw, facesOld[i]); // LB: seg fault (need to pass rect inside of vector...)
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
				}

                // LB: Test Ellipse extraction of face....
                //int ttt=segmentEllipse(skinImage);
                Mat faceSegmented=utilsObj->segmentEllipse(allFaces,allFacesSkin,displayFaces); 
                //cout << "Is face seg empty: " <<  faceSegmented.empty() << endl;
                //LB Check face was found!
                if (!faceSegmented.empty())
                {
                    // Resize to standard
                    resize(faceSegmented,faceSegmented,Size(faceSize,faceSize));
                    utilsObj->convertCvToYarp(faceSegmented,faceImages);
                    imageOut.write();
                    cout << "Sending face to output port" << endl;
                }
                else
                {
                    cout << " Face segmentation unsuccessful" << endl;
                }
			}


		    targetPort.write();
		    waitKey(1);
	    }
    }

    return true;
}

bool visionDriver::configure(ResourceFinder &rf)
{
    Property config;
    config.fromConfigFile(rf.findFile("from").c_str());

    Bottle &bGeneral = config.findGroup("general");

    imageInPort = bGeneral.find("imageInPort").asString().c_str();
    vectorOutPort = bGeneral.find("vectorOutPort").asString().c_str();
    imageOutPort = bGeneral.find("imageOutPort").asString().c_str();
    gazeOutPort = bGeneral.find("gazeOutPort").asString().c_str();
    syncPortConf = bGeneral.find("syncInPort").asString().c_str();
    cascadeFile = bGeneral.find("cascadeFile").asString().c_str();

    cout << "------------------------" << endl;
    cout << imageInPort.c_str() << endl;
    cout << vectorOutPort << endl;
    cout << imageOutPort << endl;
    cout << gazeOutPort << endl;
    cout << syncPortConf << endl;
    cout << cascadeFile << endl;
    cout << "------------------------" << endl;

    isGPUavailable = getCudaEnabledDeviceCount();

	if (isGPUavailable == 0)
	{
		cout << "No GPU found or the library is compiled without GPU support" << endl;
		cout << "Proceeding on CPU" << endl;
		cout << "Detecting largest face in view only for performance" << endl;
		hardware_int = 0;

        return false;
	}
	else
	{
		hardware_int = 1;
		cv::gpu::getDevice();
		cout << "Proceeding on GPU" << endl;
	}

	inOpen = faceTrack.open(imageInPort.c_str());
	outOpen = targetPort.open(vectorOutPort.c_str());
	imageOutOpen = imageOut.open(imageOutPort.c_str());

	gazeOut = gazePort.open(gazeOutPort.c_str());
	syncPortIn = syncPort.open(syncPortConf.c_str());

	syncBottleOut.clear();
	syncBottleOut.addString("stat");

	if(!inOpen | !outOpen | !imageOutOpen | !gazeOut )
	{
		cout << "Could not open ports. Exiting" << endl;
		return false;
	}

	inCount = faceTrack.getInputCount();
	outCount = faceTrack.getOutputCount();

	step = 0;
    maxSize = 0;
    biggestFace = 0;
    count = 0;
    faceSize = 400;
    boxScaleFactor = 20;
    
	inStatus = true;

	if( displayFaces )
	{
		namedWindow("faces",1);
		namedWindow("wholeImage",1);
		waitKey(1);
	}		
	
	face_cascade.load(cascadeFile.c_str());
}

bool visionDriver::interruptModule()
{
    return true;
}

double visionDriver::getPeriod()
{
    return 0.1;
}

