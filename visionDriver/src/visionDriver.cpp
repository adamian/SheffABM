
#include "visionDriver.h"


visionDriver::visionDriver()
{
    imgBlurPixels=7; //gauss smoothing pixels
    sagittalSplit = 0;  // split person in left and right
	displayFaces = true;
	displayBodies = true;
    utilsObj = new visionUtils();
    detectorObj = new skinDetector();
    //Mat faceSegMaskInv;
    faceSegFlag=false;
    bodySegFlag=false;    
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
		
		    noFaces = face_cascade.detectMultiScale(grayscaleFrameGPU,objBufFaceGPU,1.2,5,Size(30,30));
		    noBodies = body_cascade.detectMultiScale(grayscaleFrameGPU,objBufBodyGPU,1.2,5,Size(100,100));

			Mat skinImage;
			Mat skinMask;
			skinImage = detectorObj->detect(captureFrameBGR, displayFaces, &skinMask);

			captureFrameFace=captureFrameBGR.clone();
			
		    if(noFaces != 0)
		    {
			    cout << "Number of faces " << noFaces << endl;
			    std::vector<cv::Mat> faceVec;
			    std::vector<cv::Mat> faceVecSkin;
			
			    noFaces = 1;

			    Mat vecSizes = Mat::zeros(noFaces,1,CV_16UC1);
			    Mat allFaces(faceSize,1,CV_8UC3,count);
                Mat allFacesSkin(faceSize,1,CV_8UC3,count);
                
			    objBufFaceGPU.colRange(0,noFaces).download(vectFaceArr);

				Rect* facesNew = vectFaceArr.ptr<Rect>();
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

/*					if(facesOld[i].width > maxSize)
					{
						maxSize = facesOld[i].width;
						biggestFace = i;
					}
*/								
					//required for rectangle faces in full image view
					Point pt1(facesOld[i].x + facesOld[i].width, facesOld[i].y + facesOld[i].height);
					Point pt2(facesOld[i].x, facesOld[i].y);
					rectangle(captureFrameFace,pt1,pt2,Scalar(0,255,0),1,8,0); 	
					
					// Text face onto picture
					captureFrameFace=addText("Face", captureFrameFace, pt1, Scalar(0,255,0));
					
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
					//imshow("Face seg", captureFrameFace);
				}

                // LB: Segment out just face....
                
                Mat faceSegTemp;
                
                Mat faceSegmented=utilsObj->segmentEllipse(allFaces,allFacesSkin,displayFaces,&faceSegTemp); 
               
                //cout << "Is face seg empty: " <<  faceSegmented.empty() << endl;
                //LB Check face was found!
                if (!faceSegmented.empty())
                {
                    currentFaceRect=facesOld[0];
                    // Resize to standard
                    resize(faceSegmented,faceSegmented,Size(faceSize,faceSize));
                    utilsObj->convertCvToYarp(faceSegmented,faceImages);
                    imageOut.write();
                    cout << "Sending face to output port" << endl;
                    
//                    imshow("facemaskinv",faceSegMaskInv);
                    faceSegMaskInv = faceSegTemp.clone();
                    faceSegFlag=true;
                }
                else
                {//LB warning disabled flag here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    //faceSegFlag=false;
                    cout << " Face segmentation unsuccessful" << endl;
                }                           
			}


		    targetPort.write();
		    waitKey(1);
		    
		    
		    
        // BODY TRACK
		    if(noBodies != 0)
		    {
			    cout << "Number of bodies: " << noBodies << endl;
                // copy in last skin image
			    captureFrameBody=captureFrameBGR.clone();
			    std::vector<cv::Mat> bodyVec;
			    std::vector<cv::Mat> bodyVecSkin;
			
			    noBodies = 1;

			    Mat vecSizes = Mat::zeros(noBodies,1,CV_16UC1);
			    Mat allBodies(bodySize,1,CV_8UC3,count);
                Mat allBodiesSkin(bodySize,1,CV_8UC3,count);
                
			    objBufBodyGPU.colRange(0,noBodies).download(vectBodyArr);

//				Rect* bodiesNew = vectBodyArr.ptr<Rect>();
				Rect* bodiesOld = vectBodyArr.ptr<Rect>();

//				ImageOf<PixelRgb>& bodyImages = imageOut.prepare();

/*
				for(int i = 0; i<noBodies; i++)
				{
					int numel = bodiesOld.size();
					if(i < numel)
					{
						centrex = bodiesNew[i].x;
						centrey = bodiesNew[i].y;
							
						centrex_old = bodiesOld[i].x;
						centrey_old = bodiesOld[i].y;

						d = (centrex_old - centrex) + (centrey_old- centrey);
						d = abs(d);

						if(d > 10)
						{
							centrex_old = bodiesOld[i].x;
							centrey_old = bodiesOld[i].y;
							bodiesOld[i] = bodiesNew[i];
						}
					}		
					else
					{
						centrex_old = bodiesNew[i].x;
						centrey_old = bodiesNew[i].y;
						centrex = centrex_old;
						centrey = centrey_old;
						bodiesOld.push_back(bodiesNew[i]);
					}
*/                            
        
                int i = 0;
//                    bodiesOld.empty();
//                    bodiesOld.push_back(bodiesNew[i]);
//                    bodiesOld[i] = bodiesNew[i];
                
                // LB - expand rectangle using additional pixels in boxScaleFactor
                if (boxScaleFactor != 0)
                {
                    bodiesOld[i].x=bodiesOld[i].x-boxScaleFactor;
                    bodiesOld[i].y=bodiesOld[i].y-boxScaleFactor;
                    bodiesOld[i].width=bodiesOld[i].width+(boxScaleFactor*2);
                    bodiesOld[i].height=bodiesOld[i].height+(boxScaleFactor*2);
                    // LB - Check the extra sizes are not outside the original image size
                    // WARNING -> MIGHT produce distortions -> could reject image instead...
                    bodiesOld[i]=utilsObj->checkRoiInImage(captureFrameRaw, bodiesOld[i]); // LB: seg fault (need to pass rect inside of vector...)
                }

				vecSizes.at<unsigned short>(i) = bodiesOld[i].width;
							
				//required for rectangle faces in full image view
				Point pt1(bodiesOld[i].x + bodiesOld[i].width, bodiesOld[i].y + bodiesOld[i].height);
				Point pt2(bodiesOld[i].x, bodiesOld[i].y);

				rectangle(captureFrameBody,pt1,pt2,Scalar(0,255,0),1,8,0); 
				sagittalSplit = int(bodiesOld[i].x+(bodiesOld[i].width/2));				
				line(captureFrameBody,Point(sagittalSplit,0),Point(sagittalSplit,height),Scalar(0,0,255),1,8,0);
				
				// LB: CHECK sagittal split is sensible -> around the middle of the image (15%of either side).....
				// if not reject segmentation....
				
				if (sagittalSplit > skinMask.cols*0.85 || sagittalSplit < skinMask.cols*0.15)
				{
				cout << " Sagittal split line is too near edge -> rejecting body detection" << endl;
				bodySegFlag=false;
				}else{
				bodySegFlag=true;				
				}
				
				Mat indices;
				sortIdx(vecSizes, indices, SORT_EVERY_COLUMN | SORT_DESCENDING);
					
				for(int i = 0; i<noBodies; i++)
				{
					if(bodiesOld[i].area() != 0)
					{
					    // Standard image facedetector, take original image
						Mat temp = captureFrameBGR.operator()(bodiesOld[i]).clone();
						resize(temp,temp,Size(bodySize,bodySize));
						bodyVec.push_back(temp);
						// LB processed skin segmented data
						Mat temp2 = skinImage.operator()(bodiesOld[i]).clone();
						resize(temp2,temp2,Size(bodySize,bodySize));
						bodyVecSkin.push_back(temp2);
						
					}
				}

				//hconcat(faceVec,allFaces); // LB original code -> segmented face from original data
				hconcat(bodyVec,allBodies);					
                hconcat(bodyVecSkin,allBodiesSkin);
                        
				if( displayBodies )
				{
					//imshow("bodies",allBodies);
					//imshow("bodies Skin",allBodiesSkin);
					imshow("Body seg",captureFrameBody);
				}
			}
			
		// #####################################################################
        // LB: Skeleton segmentation to find arms for action detection
        // ########################################################
        
        //cout << "Faceseg empty?: " << faceSegMaskInv.empty() << "face  seg flag" << faceSegFlag << " body seg flag:" << bodySegFlag << endl;
        
            if (!faceSegMaskInv.empty() && faceSegFlag && bodySegFlag)
            {
			    cout << skinMask.size() << " inv mask " << faceSegMaskInv.size() << endl;
			    cout << currentFaceRect.width << " h=" << currentFaceRect.height << endl;
			    Mat rectMaskFaceOnly = Mat::zeros( skinMask.size(), CV_8UC1 );
			    Mat skinMaskNoFace;
			    Mat faceSegTemp;
			    resize(faceSegMaskInv,faceSegTemp,Size(currentFaceRect.height,currentFaceRect.width));
			    faceSegTemp.copyTo(rectMaskFaceOnly(currentFaceRect) );
		        //threshold(rectMaskFaceOnly, rectMaskFaceOnly, 127, 255, THRESH_BINARY);
			    bitwise_not(rectMaskFaceOnly,rectMaskFaceOnly);
		        //threshold(skinMask, skinMask, 127, 255, THRESH_BINARY);
		        bitwise_and(rectMaskFaceOnly,skinMask,skinMaskNoFace);
		        
		        if( displayBodies )
		        {
			        imshow("Rectangle mask face",rectMaskFaceOnly);
			        //imshow("skinmask face",skinMask);
                    imshow("skinmask no face :)",skinMaskNoFace);					    
                }
                
                // FOR SKELETON TRACKING draw over face in skin mask... facemask
			    //rectangle(skinMask,pt1,pt2,cvScalar(0,0,0,0),-1,8,0); 	
			    //if (displayBodies) imshow("Skin_mask_noface",skinMask);
			    // Send to skeleton fn here
			    Mat skelMat;
			    //skelMat=utilsObj->skeletonDetect(skinMaskNoFace, imgBlurPixels, displayBodies);
			    //vector<Rect> boundingBox = utilsObj->getArmRects(skinMaskNoFace, imgBlurPixels, &skelMat, displayFaces);
		        vector<Rect> boundingBox = utilsObj->segmentLineBoxFit(skinMaskNoFace, 100, 2, &skelMat, &returnContours, displayFaces);

			    //check atleast two bounding boxes found for left and right arms...
			    if (boundingBox.size()>1)
			    {
			        
                    int leftArmInd=0;
                    int rightArmInd=0;
                    int testInXMost=0;
                    int testInXLeast=captureFrameFace.cols;
			        
			        for (int i = 0; i<boundingBox.size(); i++)
			        {
			            if (boundingBox[i].x<testInXLeast){
			            testInXLeast=boundingBox[i].x;
			            rightArmInd=i;
			            }
			            
			            if (boundingBox[i].x>testInXMost){
			            testInXMost=boundingBox[i].x;
			            leftArmInd=i;
			            }			        
                     }

                    // Check both boudingBoxes arent the same for left and right 
                    if (rightArmInd!=leftArmInd)
                    {
		            	//Draw left arm rectangles
				        Point pt1(boundingBox[leftArmInd].x + boundingBox[leftArmInd].width, boundingBox[leftArmInd].y + boundingBox[leftArmInd].height);
				        Point pt2(boundingBox[leftArmInd].x, boundingBox[leftArmInd].y);
				        rectangle(captureFrameFace,pt1,pt2,Scalar(0,0,255),1,8,0);
				        captureFrameFace=addText("Left arm", captureFrameFace, pt1, Scalar(0,0,255));
			            
		            	//Draw right arm rectangles
				        Point pt3(boundingBox[rightArmInd].x + boundingBox[rightArmInd].width, boundingBox[rightArmInd].y + boundingBox[rightArmInd].height);
				        Point pt4(boundingBox[rightArmInd].x, boundingBox[rightArmInd].y);
				        rectangle(captureFrameFace,pt3,pt4,Scalar(0,0,255),1,8,0);
				        captureFrameFace=addText("Right arm", captureFrameFace, pt3, Scalar(0,0,255));			    
			            
			            // ###############################################
			            // Extract arms -> for CamShift and Skeleton processing
	                    // Original color versions
			            Mat leftArmBGR=captureFrameBGR(boundingBox[leftArmInd]);
			            Mat rightArmBGR=captureFrameBGR(boundingBox[rightArmInd]);
			            
//			            if (displayFaces) imshow("Left arm ",leftArmBGR);
			            if (displayFaces) imshow("Right arm",rightArmBGR);
			            
			            // skinMask
			            Mat leftArmSkin=skinMask(boundingBox[leftArmInd]);
			            Mat rightArmSkin=skinMask(boundingBox[rightArmInd]);
                        // Apply skeleton masking to arms....
                        Mat leftskel = utilsObj->skeletonDetect(leftArmSkin, imgBlurPixels, displayFaces);
                        Mat rightskel = utilsObj->skeletonDetect(rightArmSkin, imgBlurPixels, displayFaces);
                        // Get contours of regions....
                        Mat leftArmSkelContours;
                        
                        vector<Rect> leftboundingBox = utilsObj->segmentLineBoxFit(leftskel, 50, 3, &leftArmSkelContours, &returnContours, false);
                        if (displayFaces) imshow("Left arm skeleton contours",leftArmSkelContours);
                        // Find hand in image..... where there are most contours...

                        if (leftboundingBox.size()>0)
                        {
                            /// Get the moments
                            vector<Moments> mu(returnContours.size() );
                            for( int i = 0; i < returnContours.size(); i++ )
                            { mu[i] = moments( returnContours[i], false ); }
                            ///  Get the mass centers:
                            vector<Point2f> mc( returnContours.size() );
                            for( int i = 0; i < returnContours.size(); i++ )
                            { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }
                            
                            Point average_mc;
                            
                            for( int i = 0; i < returnContours.size(); i++ )
                            {
                                average_mc.x = average_mc.x + mc[i].x;
                                average_mc.y = average_mc.y + mc[i].y;
                            }
                            
                            average_mc.x = average_mc.x/returnContours.size();
                            average_mc.y = average_mc.y/returnContours.size();
                            
                            average_mc.x = average_mc.x + boundingBox[leftArmInd].x;
                            average_mc.y = average_mc.y + boundingBox[leftArmInd].y;
                            
                            circle(captureFrameFace,average_mc,10,Scalar(0,255,0),3);
                            
    						Bottle leftHandPositionOutput;
    						leftHandPositionOutput.clear();
    						leftHandPositionOutput.addDouble(average_mc.x);
    						leftHandPositionOutput.addDouble(average_mc.y);
    						leftHandPort.write(leftHandPositionOutput);
                        }

                        Mat rightArmSkelContours;
                        
                        vector<Rect> rightboundingBox = utilsObj->segmentLineBoxFit(rightskel, 50, 3, &rightArmSkelContours, &returnContours, false);
                        if (displayFaces) imshow("Right arm skeleton contours",rightArmSkelContours);
                        // Find hand in image..... where there are most contours...

                        if (rightboundingBox.size()>0)
                        {
                            /// Get the moments
                            vector<Moments> mu(returnContours.size() );
                            for( int i = 0; i < returnContours.size(); i++ )
                            { mu[i] = moments( returnContours[i], false ); }
                            ///  Get the mass centers:
                            vector<Point2f> mc( returnContours.size() );
                            for( int i = 0; i < returnContours.size(); i++ )
                            { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }
                            
                            Point average_mc;
                            
                            for( int i = 0; i < returnContours.size(); i++ )
                            {
                                average_mc.x = average_mc.x + mc[i].x;
                                average_mc.y = average_mc.y + mc[i].y;
                            }
                            
                            average_mc.x = average_mc.x/returnContours.size();
                            average_mc.y = average_mc.y/returnContours.size();
                            
                            average_mc.x = average_mc.x + boundingBox[rightArmInd].x;
                            average_mc.y = average_mc.y + boundingBox[rightArmInd].y;
                            
                            circle(captureFrameFace,average_mc,10,Scalar(0,255,0),3);                            

    						Bottle rightHandPositionOutput;
    					    rightHandPositionOutput.clear();
    						rightHandPositionOutput.addDouble(average_mc.x);
    						rightHandPositionOutput.addDouble(average_mc.y);
    						rightHandPort.write(rightHandPositionOutput);
                        }
  
                    }
                    else
		            {
		                // LB can ADD in here to find which is visibile using the saggital split from the body tracker....
		                cout << "Only one arm found....." << endl;	    
			        }
			    }
			    else
		        {
		            cout << "No arms found....." << endl;	    
			    }
			    
			    
			    if( displayFaces )
				{
					imshow("Face & Arms", captureFrameFace);
				} 
            }
					    
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
    // LB added
    //skinMaskOutPort= bGeneral.find("skinMaskOutPort").asString().c_str();
    gazeOutPort = bGeneral.find("gazeOutPort").asString().c_str();
    //syncPortConf = bGeneral.find("syncInPort").asString().c_str();
    faceCascadeFile = bGeneral.find("faceCascadeFile").asString().c_str();
    bodyCascadeFile = bGeneral.find("bodyCascadeFile").asString().c_str();

    cout << "------------------------" << endl;
    cout << imageInPort.c_str() << endl;
    cout << vectorOutPort << endl;
    cout << imageOutPort << endl;
    cout << gazeOutPort << endl;
    //cout << syncPortConf << endl;
    //cout << skinMaskOutPort << endl;
    cout << faceCascadeFile << endl;
    cout << bodyCascadeFile << endl;
    
    leftHandPortName = "/visionDriver/leftHandPosition:o";
    rightHandPortName = "/visionDriver/rightHandPosition:o";
    
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
	
	leftHandPort.open(leftHandPortName.c_str());
	rightHandPort.open(rightHandPortName.c_str());
	
	//skinMaskOutOpen = skinMaskOut.open(skinMaskOutPort.c_str());
	//syncPortIn = syncPort.open(syncPortConf.c_str());

	//syncBottleOut.clear();
	//syncBottleOut.addString("stat");

	if(!inOpen | !outOpen | !imageOutOpen | !gazeOut )//| !skinMaskOutOpen )
	{
		cout << "Could not open ports. Exiting" << endl;
		return false;
	}

	inCount = faceTrack.getInputCount();
	outCount = faceTrack.getOutputCount();
    
	step = 0;
//    maxSize = 0;
//    biggestFace = 0;
    count = 0;
    faceSize = 400;
    bodySize = faceSize;
    boxScaleFactor = 20;
    
	inStatus = true;

	if( displayFaces )
	{
		namedWindow("faces",1);
		namedWindow("wholeImage",1);
		waitKey(1);
	}		
	
	face_cascade.load(faceCascadeFile.c_str());
	body_cascade.load(bodyCascadeFile.c_str());
}

bool visionDriver::interruptModule()
{
    return true;
}

double visionDriver::getPeriod()
{
    return 0.1;
}


Mat visionDriver::addText(string textIn, Mat img, Point textLocation, Scalar colour)
{

//string text = "Funny text inside the box";
int fontFace = FONT_HERSHEY_SIMPLEX;
double fontScale = 1;
int thickness = 2;

//Mat img(600, 800, CV_8UC3, Scalar::all(0));

int baseline=0;
Size textSize = getTextSize(textIn, fontFace,
                            fontScale, thickness, &baseline);
baseline += thickness;

// center the textIn
Point textOrg(textLocation.x - (textSize.width/2),textLocation.y + (textSize.height/2));

// draw the box
//rectangle(img, textOrg + Point(0, baseline),
//          textOrg + Point(textSize.width, -textSize.height),
//          Scalar(0,0,255));
// ... and the baseline first
//line(img, textOrg + Point(0, thickness),
//     textOrg + Point(textSize.width, thickness),
//     Scalar(0, 0, 255));
//
// then put the textIn itself
putText(img, textIn, textOrg, fontFace, fontScale,
        colour, thickness, 8);

return img;
}



