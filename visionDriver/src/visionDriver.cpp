
#include "visionDriver.h"


visionDriver::visionDriver()
{
    // ##### Options ###### 
    imgBlurPixels=7; //gauss smoothing pixels
	faceSize = 400;
    bodySize = faceSize;
    boxScaleFactor = 20; // Expand face and body detected regions by this amount in pixels
	// LB additional neck scale factor for masking the neck region..... basically add pixels south
	neckScaleFactor = 40; // pixels south...

	displayFaces = false;//true;
	displayBodies = false; //true;
    utilsObj = new visionUtils();
    //detectorObj = new skinDetector();
    //Mat faceSegMaskInv;
    faceSegFlag=false;
    bodySegFlag=false;
    firstLeftHandMovement = false;
    firstRightHandMovement = false;
    
    left_hand_position.x = 0;
    left_hand_position.y = 0;
    right_hand_position.x = 0;
    right_hand_position.y = 0;
    
    previous_left_hand_position.x = 0;
    previous_left_hand_position.y = 0;
    previous_right_hand_position.x = 0;
    previous_right_hand_position.y = 0;
    
    // Detect skin using default values for first go.......
	hsvAdaptiveValues.clear();
	
	// Compare point distances for tracking the same point....
	calibratedLeftPoints = false;
	calibratedRightPoints = false;
	//leftArmPointIndex
//	previousLeftArmPoints = new Point2f(4);
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
	    
	        // Init bodyPartLocations vector if anything found...
	        for (int i; i<12;i++)
	            bodyPartLocations[i]=-1.0; // -1 is no position found....
	        bodyPosFound=false; // set flag off -> set to true when body part position found

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
		    
		    // Haar cascades on GPU	
		    captureFrameGPU.upload(captureFrameBGR);
		    cv::gpu::cvtColor(captureFrameGPU,grayscaleFrameGPU,CV_BGR2GRAY);
		    cv::gpu::equalizeHist(grayscaleFrameGPU,grayscaleFrameGPU);
		    // Face and Body
		    noFaces = face_cascade.detectMultiScale(grayscaleFrameGPU,objBufFaceGPU,1.2,5,Size(30,30));
		    noBodies = body_cascade.detectMultiScale(grayscaleFrameGPU,objBufBodyGPU,1.2,5,Size(100,100));
		    
			Mat skinImage;
			Mat skinMaskDefault;
			Mat skinHSV;
			// Detect skin using default values.......
			std::vector<int> hsvDefault;
			// LB: this will always use the default values, to prevent runaway adaption!
            skinImage = utilsObj->skinDetect(captureFrameBGR, &skinHSV, &skinMaskDefault, hsvDefault, 400,7, 3, 0, displayFaces);

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

					// Add extra pixels to bottom to remove neck skin region...
					if (neckScaleFactor !=0)
					{
						facesOld[i].height=facesOld[i].height+neckScaleFactor;
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
					
				    // Add values to body part pos vector (Face x(0),y(1),z(2))
                    bodyPartLocations[0]=int(facesOld[i].x+(facesOld[i].width/2));// Face  x
                    bodyPartLocations[1]=int(facesOld[i].y+(facesOld[i].height/2));// Face y
                    bodyPartLocations[2]=1.0;// Face z -> ++++++++++++++++++ SET AT DEFAULT 1 for NOW NEED TO UPDATE LATER...... STEREOVISION
                    bodyPosFound=true; // position found -> set flag to on	
					
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
                
                Mat1b faceSegMask;
                
                Mat faceSegmented=utilsObj->segmentFace(allFaces,allFacesSkin,displayFaces,&faceSegMask); 
               
                //cout << "Is face seg empty: " <<  faceSegmented.empty() << endl;
                //LB Check face was found!
                if (!faceSegmented.empty())
                {
                    currentFaceRect=facesOld[0];
                    // Resize to standard
                    resize(faceSegmented,faceSegmented,Size(faceSize,faceSize));
                    utilsObj->convertCvToYarp(faceSegmented,faceImages);
                    imageOut.write();
                    //cout << "Sending face to output port" << endl;
                    
                    // LB: ADAPTIVE SKIN MASKING SECTION -> uses skin on face to refine the skin detector......
                    // LB testing -> get all values from skin mask 
                    // Run through pixels in mask and 
                    // Needs to be more efficient... see image scan opencv documentation...
                    
                    // HSV pixels that pass skin detection -> uses face track and segmented face data
                    std::vector<Mat> hsvPixels(3);
                    // Take skinHSV (returned from skin detector) and uses face extracted rect..... 
                    Mat3b faceHSV = skinHSV(currentFaceRect);
                    //cvtColor(faceSegmented, faceHSV, CV_BGR2HSV);

                    // Loop through rows
                    for (int i =0; i < faceHSV.rows; i++)
                    {
                    // Loop through cols
                        for (int j =0; j < faceHSV.cols; j++)
                        {
                            //cout << "faceSegMask pix val:" << faceSegMask.at<int>(i,j) << endl;
                            // check pixels in face mask
                            //if  (faceSegMask)
                            //if  (faceSegMask.at<int>(i,j)>0)
                            //{   // Check HSV is not zero....
                            if (faceHSV(i,j)[0]!=0 || faceHSV(i,j)[1]!=0 || faceHSV(i,j)[2]!=0)
                            {
                                // save HSV values from pixel in mask    
                                hsvPixels[0].push_back(faceHSV(i,j)[0]); //H
                                hsvPixels[1].push_back(faceHSV(i,j)[1]); //S
                                hsvPixels[2].push_back(faceHSV(i,j)[2]); //V
                            }
                        }
                    }
                    
                    
                    //imshow("HSV seg face",faceHSV);
                    
                    // Update adaptive skin detection vector.... for person specific detection....
                    hsvAdaptiveValues = utilsObj-> updateHSVAdaptiveSkin(hsvPixels, false);
                    
                    //cout << "h pixel values:" << hsvPixels[0] << endl;
                    //cout << "found " << hsvPixels[0].size() << " hsv pixels in mask" << endl;
                    
                    faceSegMaskInv = faceSegMask.clone();
                    //imshow("facemaskinv",faceSegMaskInv);                 
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
				
				if (sagittalSplit > skinMaskDefault.cols*0.85 || sagittalSplit < skinMaskDefault.cols*0.15)
				{
				    cout << " Sagittal split line is too near edge -> rejecting body detection" << endl;
				    bodySegFlag=false;
				}
				else
				{
				    bodySegFlag=true;
				    // Add values to body part pos vector (right arm x(3),y(4),z(5))
                    bodyPartLocations[3]=int(bodiesOld[i].x+(bodiesOld[i].width/2));// Body  x
                    bodyPartLocations[4]=int(bodiesOld[i].y+(bodiesOld[i].height/2));// Body y
                    bodyPartLocations[5]=1.0;// Body z -> ++++++++++++++++++ SET AT DEFAULT 1 for NOW NEED TO UPDATE LATER...... STEREOVISION
                    bodyPosFound=true; // position found -> set flag to on		
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
            
                Mat skinImageTemp;
                Mat skinHSVtemp;
                Mat skinMask;
                // LB redo skin masking but with adaptive filter 
                skinImageTemp = utilsObj->skinDetect(captureFrameBGR, &skinHSVtemp, &skinMask, hsvAdaptiveValues, 400,7, 3, 0, displayFaces);   
//			    cout << skinMask.size() << " inv mask " << faceSegMskInv.size() << endl;
	//		    cout << currentFaceRect.width << " h=" << currentFaceRect.height << endl;
			    Mat rectMaskFaceOnly = Mat::zeros( skinMask.size(), CV_8UC1 );
			    Mat skinMaskNoFace;
			    Mat faceSegTemp;
			    //resize(faceSegMaskInv,faceSegTemp,Size(currentFaceRect.height,currentFaceRect.width));
				resize(faceSegMaskInv,faceSegTemp,Size(currentFaceRect.width,currentFaceRect.height));//Size(currentFaceRect.height,currentFaceRect.width));
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
                    imshow("skin seg WITH ADAPTIVE SKIN SEG...... ",skinImageTemp);
                    				    
                }
                
                // FOR ARM TRACKING draw over face in skin mask... facemask
			    //rectangle(skinMask,pt1,pt2,cvScalar(0,0,0,0),-1,8,0); 	
			    //if (displayBodies) imshow("Skin_mask_noface",skinMask);
			    // Send to skeleton fn here
			    Mat skelMat;
			    //skelMat=utilsObj->skeletonDetect(skinMaskNoFace, imgBlurPixels, displayBodies);
			    //vector<Rect> boundingBox = utilsObj->getArmRects(skinMaskNoFace, imgBlurPixels, &skelMat, displayFaces);
			    
			    // FIND LEFT AND RIGHT ARM REGIONS....
		        vector<Rect> boundingBox = utilsObj->segmentLineBoxFit(skinMaskNoFace, 1000, 2, &skelMat, &returnContours, &armRotatedRects, displayFaces);

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
                    
                        // LB: Testing -> find approx centre of person....
                        if (sagittalSplit!=0)
                        {
                            bodyCentre.x=sagittalSplit;
                            bodyCentre.y=currentFaceRect.y+currentFaceRect.height;
                            circle(captureFrameFace,bodyCentre,10,Scalar(147,20,255),3);
                        
                        }
                        
                        // ######### LEFT ARM
                        // Set positon left_hand_position -> uses centre of bouding rect
                        //left_hand_position=armRotatedRects[leftArmInd].center;
                        
                        //Size leftBoxSize = armRotatedRects[leftArmInd].size();
                        //left_hand_position=armRotatedRects[leftArmInd].center+int(leftBoxSize.height*0.4);
                        
                        Point2f leftArmPoints[4];                        
                        armRotatedRects[leftArmInd].points(leftArmPoints);
                        
                        if( calibratedLeftPoints == false )
                        {                            
                            if( (abs(bodyCentre.x-armRotatedRects[leftArmInd].center.x) > 20.0) && (abs(bodyCentre.y-armRotatedRects[leftArmInd].center.y) > 20.0) )
                            {
                                cout << "=============================================================================" << endl;
                                cout << "++++++++++++++++++++++++++++++CALIBRATING LEFT+++++++++++++++++++++++++++++++" << endl;
                                cout << "=============================================================================" << endl;

                                int longestLeftIndex = utilsObj->updateArmPoints(previousLeftArmPoints, leftArmPoints,1);   //finds initial longest point
                                previousLeftArmPoints = leftArmPoints[longestLeftIndex];
                            
    //                            previousLeftArmPoints = leftArmPoints[0];
                                calibratedLeftPoints = true;                             
                            }                           
                            // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ COULD BE USEFUL -> furthest distance from centre
    						// LB Testing -> use distance from centre of person to predict hand location?!?!?!?
    						// Furthest will be least 
/*    						if (bodyCentre.x!=0)
    						{
    						    int greatestDistance=0;
        						int temp;
        						int xContour;
        						int yContour;
        						Point2f leftArmLoc;
        						// Largest contour = 0 (hopefully), go through points...
        						for (int i=0; i<4; i++)
        						{
            						// Classic pythagoras (no need to sqrt)
            						
            						temp=pow((bodyCentre.x-leftArmPoints[i].x),2) + pow((bodyCentre.y-leftArmPoints[i].y),2);
            						if (temp>greatestDistance)
            						{
                						greatestDistance=temp;
                						leftArmLoc.x=leftArmPoints[i].x;
                						leftArmLoc.y=leftArmPoints[i].y;
            						}	
        						}
        						//circle(captureFrameFace,leftArmLoc,10,Scalar(255,255,255),3);
        						previousLeftArmPoints = leftArmLoc;        						
    						}
*/
                        }
                        
                        
                        // Find current point which is closest to previous point
                        int closestLeftIndex = utilsObj->updateArmPoints(previousLeftArmPoints, leftArmPoints, 0);  //finds closest point
                        cout << "Closest index: " << closestLeftIndex << endl;
                        // Set left arm location
                        left_hand_position=leftArmPoints[closestLeftIndex];
                        // Update previous point
                        previousLeftArmPoints=left_hand_position;
                                            
                        for (int i=0;i<4;i++)
                        {
                            char buffer[100];
                            sprintf(buffer,"%d",i);
                            putText(captureFrameFace, buffer, leftArmPoints[i], FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255), 1, 8);
                        }
                        

//                        previousLeftArmPoints = leftArmPoints[0;
                        
//                        if( armRotatedRects[leftArmInd].size.width < armRotatedRects[leftArmInd].size.height )
//                            tAngle = 90 + armRotatedRects[leftArmInd].angle;
                        
//                        char buffer[100];
//                        sprintf(buffer,"%f", tAngle);                     
//                        putText(captureFrameFace, buffer, armRotatedRects[leftArmInd].center, FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,255), 2, 8);
                        
                        
                        circle(captureFrameFace, left_hand_position,10,Scalar(0,255,0),3);
                        
		            	//Draw left arm rectangles
				        Point pt1(boundingBox[leftArmInd].x + boundingBox[leftArmInd].width, boundingBox[leftArmInd].y + boundingBox[leftArmInd].height);
				        Point pt2(boundingBox[leftArmInd].x, boundingBox[leftArmInd].y);
				        rectangle(captureFrameFace,pt1,pt2,Scalar(0,0,255),1,8,0);
				        utilsObj->drawRotatedRect(captureFrameFace, armRotatedRects[leftArmInd], Scalar(255,0,0));
				        captureFrameFace=addText("Left arm", captureFrameFace, pt1, Scalar(0,0,255));
			            
			            // ######### Right ARM
                        // Set positon left_hand_position -> uses centre of bouding rect
//                        right_hand_position=armRotatedRects[rightArmInd].center;
//                        circle(captureFrameFace, right_hand_position,10,Scalar(0,255,0),3);
                                                    
		            	//Draw right arm rectangles
				        Point pt3(boundingBox[rightArmInd].x + boundingBox[rightArmInd].width, boundingBox[rightArmInd].y + boundingBox[rightArmInd].height);
				        Point pt4(boundingBox[rightArmInd].x, boundingBox[rightArmInd].y);
				        rectangle(captureFrameFace,pt3,pt4,Scalar(0,0,255),1,8,0);
				        utilsObj->drawRotatedRect(captureFrameFace, armRotatedRects[rightArmInd], Scalar(255,0,0));
				        captureFrameFace=addText("Right arm", captureFrameFace, pt3, Scalar(0,0,255));			    
			            
			            // ###############################################
			            // Extract arms -> for CamShift and Skeleton processing
	                    // Original color versions
			            //Mat leftArmBGR=captureFrameBGR(boundingBox[leftArmInd]);
			            //Mat rightArmBGR=captureFrameBGR(boundingBox[rightArmInd]);
			            
                        //if (displayFaces) imshow("Left arm ",leftArmBGR);
                        //if (displayFaces) imshow("Right arm",rightArmBGR);
		                
		                //int noRightHands = hand_cascade.detectMultiScale(grayscaleFrameGPU(boundingBox[rightArmInd]),objBufRightHandGPU,1.2,5,Size(10,10));			               	
		                
			            // skinMask
			            // Mat leftArmSkin=skinMask(boundingBox[leftArmInd]);
			            // Mat rightArmSkin=skinMask(boundingBox[rightArmInd]);
                        // Apply skeleton masking to arms....
                        //Mat leftskel = utilsObj->skeletonDetect(leftArmSkin, imgBlurPixels, displayFaces);
                        //Mat rightskel = utilsObj->skeletonDetect(rightArmSkin, imgBlurPixels, displayFaces);
                        // Get contours of regions....
                        //Mat leftArmSkelContours;
                        //vector<Rect> leftboundingBox;

                        // Defined in header file                      
//                        Point average_mc;
//                        Point previous_average_mc;
//                        average_mc.x = 0;
//                        average_mc.y = 0;
                        /*
                        int windowSize = 20;
                        int limitWindow = 10;

                        left_hand_average_mc.x = 0;
                        left_hand_average_mc.y = 0;

                        vector<RotatedRect> leftArmRotatedRect; 

                        for( int j = 0; j < windowSize; j++ )
                        {
                        leftboundingBox = utilsObj->segmentLineBoxFit(leftskel, 50, 1, &leftArmSkelContours, &returnContours, &leftArmRotatedRect, false); // was 3 contours...
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
                            
//                            Point average_mc;
//                            average_mc.x = 0;
//                            average_mc.y = 0;
                            
                            for( int i = 0; i < returnContours.size(); i++ )
                            {
                                // Centre of mass method                                
//                                left_hand_average_mc.x = left_hand_average_mc.x + mc[i].x;
//                                left_hand_average_mc.y = left_hand_average_mc.y + mc[i].y;

                                left_hand_average_mc.x = left_hand_average_mc.x + boundingBox[leftArmInd].x;
                                left_hand_average_mc.y = left_hand_average_mc.y + boundingBox[leftArmInd].y;
                            }
                            
                            left_hand_average_mc.x = left_hand_average_mc.x/returnContours.size();
                            left_hand_average_mc.y = left_hand_average_mc.y/returnContours.size();
                            
//                            left_hand_average_mc.x = left_hand_average_mc.x + boundingBox[leftArmInd].x;
//                            left_hand_average_mc.y = left_hand_average_mc.y + boundingBox[leftArmInd].y;
                                                      
//                            circle(captureFrameFace,left_hand_average_mc,10,Scalar(0,255,0),3);
                            
//                            utilsObj->isHandMoving(left_hand_average_mc);
                               						
    						
    						// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ COULD BE USEFUL -> furthest distance from centre
    						// LB Testing -> use distance from centre of person to predict hand location?!?!?!?
    						// Furthest will be least 
    						if (bodyCentre.x!=0)
    						{
    						    int greatestDistance=0;
        						int temp;
        						int xContour;
        						int yContour;
        						Point leftArmLoc;
        						// Largest contour = 0 (hopefully), go through points...
        						for (int i=0; i<returnContours[0].size(); i++)
        						{
            						// Classic pythagoras (no need to sqrt)
            						xContour=returnContours[0][i].x+boundingBox[leftArmInd].x;
            						yContour=returnContours[0][i].y+boundingBox[leftArmInd].y;
            						
            						temp=(bodyCentre.x-xContour)*(bodyCentre.x-xContour) + (bodyCentre.y-yContour)*(bodyCentre.y-yContour);
            						if (temp>greatestDistance)
            						{
                						greatestDistance=temp;
                						leftArmLoc.x=xContour;
                						leftArmLoc.y=yContour;
            						}	
        						}
        						circle(captureFrameFace,leftArmLoc,10,Scalar(255,255,255),3);
        						
    						}
    						
                        }
                        }
  
                        if (leftboundingBox.size()>0 )
                        {
                        left_hand_average_mc.x = left_hand_average_mc.x/windowSize;
                        left_hand_average_mc.y = left_hand_average_mc.y/windowSize;
                        left_hand_average_mc.x = left_hand_average_mc.x + boundingBox[leftArmInd].x;
                        left_hand_average_mc.y = left_hand_average_mc.y + boundingBox[leftArmInd].y;
                        }

                        circle(captureFrameFace,left_hand_average_mc,10,Scalar(0,255,0),3);
                        */
                      
                        //if (leftboundingBox.size()>0 )
                        //{
                        
                        // @@@@@@@@@@@@@@@@@@@@@@ Are the hands moving @@@@@@@@@@@@@@@@@@@@@
                        //Set number of pixels to detect hand movement....
                        int limitWindow = 5;
                        // IS the left hand moving
                        if( !firstLeftHandMovement )
                        {
                            previous_left_hand_position = left_hand_position;
                            firstLeftHandMovement = true;
                        }

                        //cout << "PREVIOUS POINT: " << left_hand_position.x << ", " << left_hand_position.y << endl;
                        //cout << "CURRENT POINT: " << previous_left_hand_position.x << ", " << previous_left_hand_position.y << endl;
                        
                        // Relative positions (take difference from start.....)
                        int relLeftXPosition = 0;
                        int relLeftYPosition = 0;

                        if( utilsObj->isHandMoving(left_hand_position,previous_left_hand_position, limitWindow) )
                        {
                            cout << "==================== LEFT HAND IS MOVING =======================" << endl;
                            relLeftXPosition = left_hand_position.x - previous_left_hand_position.x;
                            relLeftYPosition = left_hand_position.y - previous_left_hand_position.y;
                        }

                        // Send out hand positions over yarp
						/*Bottle leftHandPositionOutput;
						leftHandPositionOutput.clear();
						leftHandPositionOutput.addDouble(relLeftXPosition);
						leftHandPositionOutput.addDouble(relLeftYPosition);
						leftHandPort.write(leftHandPositionOutput);                       
                        */
                        // Add values to body part pos vector (left arm x(6),y(7),z(8))
                        bodyPartLocations[6]=relLeftXPosition;// left hand  x
                        bodyPartLocations[7]=relLeftYPosition;// left hand  y
                        bodyPartLocations[8]=1.0;// left hand  z -> ++++++++++++++++++ SET AT DEFAULT 1 for NOW NEED TO UPDATE LATER...... STEREOVISION
                        bodyPosFound=true; // position found -> set flag to on
                        previous_left_hand_position = left_hand_position;                          
                        //}                                       

 /*                       
                       Mat rightArmSkelContours;

                        vector<Rect> rightboundingBox;
                        vector<RotatedRect> rightArmRotatedRect; 
//                        Point right_hand_average_mc;
                        right_hand_average_mc.x = 0;
                        right_hand_average_mc.y = 0;

//                        int windowSize = 20;
                        for( int j = 0; j < windowSize; j++ )
                        {                       
                        rightboundingBox = utilsObj->segmentLineBoxFit(rightskel, 50, 1, &rightArmSkelContours, &returnContours, &rightArmRotatedRect, false);
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
                            
//                            Point right_hand_average_mc;
//                            right_hand_average_mc.x = 0;
//                            right_hand_average_mc.y = 0;
                            
                            for( int i = 0; i < returnContours.size(); i++ )
                            {
                                // Centre of mass method
//                                right_hand_average_mc.x = right_hand_average_mc.x + mc[i].x;
//                                right_hand_average_mc.y = right_hand_average_mc.y + mc[i].y;

                                right_hand_average_mc.x = right_hand_average_mc.x + boundingBox[rightArmInd].x;
                                right_hand_average_mc.y = right_hand_average_mc.y + boundingBox[rightArmInd].y;
                            }
                            
                            right_hand_average_mc.x = right_hand_average_mc.x/returnContours.size();
                            right_hand_average_mc.y = right_hand_average_mc.y/returnContours.size();
                                                       
//                            right_hand_average_mc.x = right_hand_average_mc.x + boundingBox[rightArmInd].x;
//                            right_hand_average_mc.y = right_hand_average_mc.y + boundingBox[rightArmInd].y;
                            
//                            circle(captureFrameFace,right_hand_average_mc,10,Scalar(0,255,0),3);                            


                        }
                        }
                        
                        right_hand_average_mc.x = right_hand_average_mc.x/windowSize;
                        right_hand_average_mc.y = right_hand_average_mc.y/windowSize;
                        right_hand_average_mc.x = right_hand_average_mc.x + boundingBox[rightArmInd].x;
                        right_hand_average_mc.y = right_hand_average_mc.y + boundingBox[rightArmInd].y;

                        circle(captureFrameFace,right_hand_average_mc,10,Scalar(0,255,0),3);
*/


                        Point2f rightArmPoints[4];
                        armRotatedRects[rightArmInd].points(rightArmPoints);

                        if( calibratedRightPoints == false )
                        {
                            if( (abs(bodyCentre.x-armRotatedRects[rightArmInd].center.x) > 20.0) && (abs(bodyCentre.y-armRotatedRects[rightArmInd].center.y) > 20.0) )
                            {
                                cout << "=============================================================================" << endl;
                                cout << "++++++++++++++++++++++++++++++CALIBRATING RIGHT+++++++++++++++++++++++++++++++" << endl;
                                cout << "=============================================================================" << endl;
                                int longestRightIndex = utilsObj->updateArmPoints(previousRightArmPoints, rightArmPoints, 1);   //finds initial longest point
                                previousRightArmPoints = rightArmPoints[longestRightIndex];

    //                            previousRightArmPoints = rightArmPoints[0];
                                calibratedRightPoints = true;
                            }                           
                            // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ COULD BE USEFUL -> furthest distance from centre
    						// LB Testing -> use distance from centre of person to predict hand location?!?!?!?
    						// Furthest will be least 
/*    						if (bodyCentre.x!=0)
    						{
    						    int greatestDistance=0;
        						int temp;
        						int xContour;
        						int yContour;
        						Point2f rightArmLoc;
        						// Largest contour = 0 (hopefully), go through points...
        						for (int i=0; i<4; i++)
        						{
            						// Classic pythagoras (no need to sqrt)
            						
            						temp=pow((bodyCentre.x-rightArmPoints[i].x),2) + pow((bodyCentre.y-rightArmPoints[i].y),2);
            						if (temp>greatestDistance)
            						{
                						greatestDistance=temp;
                						rightArmLoc.x=rightArmPoints[i].x;
                						rightArmLoc.y=rightArmPoints[i].y;
            						}	
        						}
        						//circle(captureFrameFace,rightArmLoc,10,Scalar(255,255,255),3);
        						previousRightArmPoints = rightArmLoc;        						
    						}
*/
                        }
                        // Find current point which is closest to previous point
                        int closestRightIndex = utilsObj->updateArmPoints(previousRightArmPoints, rightArmPoints, 0);   //finds closest point
                        cout << "Closest index: " << closestRightIndex << endl;
                        // Set right arm location
                        right_hand_position=rightArmPoints[closestRightIndex];
                        // Update previous point
                        previousRightArmPoints=right_hand_position;
                                            
                        for (int i=0;i<4;i++)
                        {
                            char buffer[100];
                            sprintf(buffer,"%d",i);
                            putText(captureFrameFace, buffer, rightArmPoints[i], FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255), 1, 8);
                        }

                        circle(captureFrameFace, right_hand_position,10,Scalar(0,255,0),3);


//                        if (rightboundingBox.size()>0 )
//                        {
                   
                        if( !firstRightHandMovement )
                        {
                            previous_right_hand_position = right_hand_position;
                            firstRightHandMovement = true;
                        }

//                            cout << "PREVIOUS POINT: " << right_hand_position.x << ", " << right_hand_position.y << endl;
//                            cout << "CURRENT POINT: " << previous_right_hand_position.x << ", " << previous_right_hand_position.y << endl;
                            
                        int relRightXPosition = 0;
                        int relRightYPosition = 0;

                        if( utilsObj->isHandMoving(right_hand_position,previous_right_hand_position, limitWindow) )
                        {
                            cout << "**************************** RIGHT HAND IS MOVING ***********************************" << endl;
                            relRightXPosition = right_hand_position.x - previous_right_hand_position.x;
                            relRightYPosition = right_hand_position.y - previous_right_hand_position.y;
                        }

						/*Bottle rightHandPositionOutput;
						rightHandPositionOutput.clear();
						rightHandPositionOutput.addDouble(relRightXPosition);
						rightHandPositionOutput.addDouble(relRightYPosition);
						rightHandPort.write(rightHandPositionOutput);*/
						
                        // Add values to body part pos vector (right arm x(9),y(10),z(11))
                        bodyPartLocations[9]=relRightXPosition;// Right hand  x
                        bodyPartLocations[10]=relRightYPosition;// Right hand  y
                        bodyPartLocations[11]=1.0;// Right hand  z -> ++++++++++++++++++ SET AT DEFAULT 1 for NOW NEED TO UPDATE LATER...... STEREOVISION
                        bodyPosFound=true; // position found -> set flag to on						
						
                                                 
                        previous_right_hand_position = right_hand_position;                          
//                        }
  
                    }
                    else
		            {
		                // LB can ADD in here to find which is visibile using the sagittal split from the body tracker....
		                cout << "Only one arm found....." << endl;	    

    		            calibratedLeftPoints = false;
		                calibratedRightPoints = false;
			        }
			    }
			    else
		        {
		            calibratedLeftPoints = false;
		            calibratedRightPoints = false;
		            cout << "No arms found....." << endl;	    
			    }
			    
//			    if(displayFaces) 
                    imshow("Face / Body / Arms", captureFrameFace);

				// @@@@@@@@@' Send found body pos values out over YARP
				// If any body part position has been found -> face, body, left hand, right hand
				if (bodyPosFound)
				{
				    Bottle bodyPartPosOutput;
				    bodyPartPosOutput.clear();
			        for (int i;i<12;i++)
				        bodyPartPosOutput.addDouble(bodyPartLocations[i]); 
				    bodyPartPosPort.write(bodyPartPosOutput);
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
    // LB testing
    //handCascadeFile = bGeneral.find("handCascadeFile").asString().c_str();


    cout << "------------------------" << endl;
    cout << imageInPort.c_str() << endl;
    cout << vectorOutPort << endl;
    cout << imageOutPort << endl;
    cout << gazeOutPort << endl;
    //cout << syncPortConf << endl;
    //cout << skinMaskOutPort << endl;
    cout << faceCascadeFile << endl;
    // LB testing
    //cout << handCascadeFile << endl;
    
    //leftHandPortName = "/visionDriver/leftHandPosition:o";
    //rightHandPortName = "/visionDriver/rightHandPosition:o";
    bodyPartPosName="/visionDriver/bodyPartPosition:o";

    //leftArmSkinPortName = "/visionDriver/leftArmSkin:o";
    //rightArmSkinPortName = "/visionDriver/rightArmSkin:o";
    
    cout << "------------------------" << endl;


    // Init bodyPartLocations vector if anything found...
    bodyPartLocations.clear();
    for (int i; i<12;i++)
        bodyPartLocations.push_back(-1.0); // -1 is no position found....

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
	
	bodyPartPosPort.open(bodyPartPosName.c_str());
	//leftHandPort.open(leftHandPortName.c_str());
	//rightHandPort.open(rightHandPortName.c_str());


	//leftArmSkinPort.open(leftArmSkinPortName.c_str());
	//rightArmSkinPort.open(rightArmSkinPortName.c_str());

	
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
    
    // Init dyn variables
    sagittalSplit = 0;  // split person in left and right 
    bodyCentre.x=0;
    bodyCentre.y=0; 
    
	step = 0;
//    maxSize = 0;
//    biggestFace = 0;
    count = 0;
    
    
	inStatus = true;

	if( displayFaces )
	{
		namedWindow("faces",1);
		namedWindow("wholeImage",1);
		waitKey(1);
	}		
	
	face_cascade.load(faceCascadeFile.c_str());
	body_cascade.load(bodyCascadeFile.c_str());
	// LB Test Hand cascade
	//hand_cascade.load(handCascadeFile.c_str());
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



