
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
    
    
    addFrameRate = true; // optional frameRate control
    
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
	
	namedWindow("Face / Body / Arms",1);
	
	//leftArmPointIndex
//	previousLeftArmPoints = new Point2f(4);
}

visionDriver::~visionDriver()
{
}

bool visionDriver::updateModule()
{

    if (addFrameRate)
    {
        startTime = clock(); 
    }
    
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
            // PROCESS NEW IMAGE
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
            
	        // Init bodyPartLocations vector if anything found... later sent out via yarp
	        for (int i; i<12;i++) bodyPartLocations[i]=0.0; // 0 is no position found....
	        bodyPosFound=false; // set flag off -> set to true when body part position found

	        captureFrameFace=captureFrameBGR.clone();

	        Mat skinImage;
			Mat skinMaskDefault;
			Mat3b skinHSV;
			// Detect skin using default values.......
			std::vector<int> hsvDefault;
			// LB: this will always use the default values, to prevent runaway adaption!
            skinImage = utilsObj->skinDetect(captureFrameBGR, &skinHSV, &skinMaskDefault, hsvDefault, 400,7, 3, 0, displayFaces);

	        
		    // SET FACE AND BODY DETECTION cascades
		    // Haar cascades on GPU	
		    // LB TEMP TESTING
		    //if (!faceSegFlag || !bodySegFlag) // LB just check for these once....
		    //{
		    
		    // Haar cascades on GPU	
		    captureFrameGPU.upload(captureFrameBGR);
		    cv::gpu::cvtColor(captureFrameGPU,grayscaleFrameGPU,CV_BGR2GRAY);
		    cv::gpu::equalizeHist(grayscaleFrameGPU,grayscaleFrameGPU);
		    // Face and Body
		    noFaces = face_cascade.detectMultiScale(grayscaleFrameGPU,objBufFaceGPU,1.2,5,Size(30,30));
		    noBodies = body_cascade.detectMultiScale(grayscaleFrameGPU,objBufBodyGPU,1.2,5,Size(100,100));

			//cout << "Got to face seg 0..." << endl;        
			// Check if face found
		    if(noFaces != 0)
		    {
//			    cout << "Number of faces " << noFaces << endl;
			    //std::vector<cv::Mat> faceVec;
			    //std::vector<cv::Mat> faceVecSkin;
			
			    noFaces = 1;

			    // Mat vecSizes = Mat::zeros(noFaces,1,CV_16UC1);
			    Mat3b face_HSV;
			    Mat allFaces;//(faceSize,1,CV_8UC3,count);
                Mat allFacesSkin;//(faceSize,1,CV_8UC3,count);
                //cout << "Got to face seg 0.1 ..." << endl;        
                // Get face cascadde info back from GPU
			    objBufFaceGPU.colRange(0,noFaces).download(vectFaceArr);

                //cout << "Got to face seg 0.2 ..." << endl;        				
				yarp::sig::Vector& posOutput = targetPort.prepare();
				posOutput.resize(noFaces*3); //each face in the list has a number id, x centre and y centre

				ImageOf<PixelRgb>& faceImages = imageOut.prepare();
                //cout << "Got to face seg 0.3 ..." << endl;                        
                int i = 0;
                //Rect* facesNew = vectFaceArr.ptr<Rect>();
                // single face disabled
				/*for(int i = 0; i<noFaces; i++)
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
                    */

                    Rect* facesOld = vectFaceArr.ptr<Rect>();

                    //cout << "Got to face seg 0.4 ..." << endl;                                    
                    // LB - expand rectangle using additional pixels in boxScaleFactor
                    if (boxScaleFactor != 0)
                    {
                        //facesOld[i].x=facesOld[i].x-boxScaleFactor;
                        //facesOld[i].y=facesOld[i].y-boxScaleFactor;
                        //facesOld[i].width=facesOld[i].width+(boxScaleFactor*2);
                        //facesOld[i].height=facesOld[i].height+(boxScaleFactor*2);
                        
                        facesOld[i]= Rect(facesOld[i].x-boxScaleFactor, facesOld[i].y-boxScaleFactor, facesOld[i].width+(boxScaleFactor*2), facesOld[i].height+(boxScaleFactor*2));

                        // LB - Check the extra sizes are not outside the original image size
                        // WARNING -> MIGHT produce distortions -> could reject image instead...
                        facesOld[i]=utilsObj->checkRoiInImage(captureFrameRaw, facesOld[i]); // LB: seg fault (need to pass rect inside of vector...)
                    }
			        //cout << "Got to face seg 1..." << endl;        
					// Add extra pixels to bottom to remove neck skin region...
					if (neckScaleFactor !=0)
					{
						facesOld[i].height=facesOld[i].height+neckScaleFactor;
						facesOld[i]=utilsObj->checkRoiInImage(captureFrameRaw, facesOld[i]); // LB: seg fault (need to pass rect inside of vector...)
					}

					//vecSizes.at<unsigned short>(i) = facesOld[i].width;

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

					//if( i == 0 )
					//{
					// Send position of face in image to iKinGazeCtrl
					Bottle posGazeOutput;
					posGazeOutput.clear();
					posGazeOutput.addString("left");
					posGazeOutput.addDouble(centrex);
					posGazeOutput.addDouble(centrey);
					posGazeOutput.addDouble(1.0);
					gazePort.write(posGazeOutput);	
					//}
			    //cout << "Got to face seg 2..." << endl;        
				//}

				//Mat indices;
				//sortIdx(vecSizes, indices, SORT_EVERY_COLUMN | SORT_DESCENDING);
					
				//for(int i = 0; i<noFaces; i++)
				//{
				if(facesOld[i].area() != 0)
				{
				        // Standard image facedetector, take original image
				        // Take face from original data
					    //Mat temp = captureFrameBGR.operator()(facesOld[i]).clone();
					    allFaces = captureFrameBGR.operator()(facesOld[i]).clone();
					    // Resixe image 
					    resize(allFaces,allFaces,Size(faceSize,faceSize));
					    //faceVec.push_back(allFaces);
					    // LB processed skin segmented data
					    //Mat temp2 = skinImage.operator()(facesOld[i]).clone();
					    allFacesSkin = skinImage.operator()(facesOld[i]).clone();
					    resize(allFacesSkin,allFacesSkin,Size(faceSize,faceSize));
					    //faceVecSkin.push_back(temp2);
					    
					    // Take skinHSV (returned from skin detector) and uses face extracted rect.....
                        face_HSV = skinHSV.operator()(facesOld[i]).clone();
                        // LB lots of testing
					    //Mat HSVTemp, HSVTemp2;
					    //cvtColor(skinHSV, HSVTemp, CV_HSV2BGR);
                        //HSVTemp2 = HSVTemp.operator()(facesOld[i]).clone();
                        //cvtColor(HSVTemp2, face_HSV, CV_BGR2HSV);
                        //resize(face_HSV,face_HSV,Size(facesOld[i].width+1,facesOld[i].height+1));
                        // LB testing			
                        //Mat faceOut = captureFrameFace(facesOld[i]).clone();	
					    currentFaceRect = facesOld[0]; //Rect(facesOld[0].x,facesOld[0].y,facesOld[0].width*2,facesOld[0].height*2);
				
				    //}

				    //hconcat(faceVec,allFaces); // LB original code -> segmented face from original data
				    //hconcat(faceVec,allFaces);					
                    //hconcat(faceVecSkin,allFacesSkin);
                    //cout << "Got to face seg 3..." << endl;        
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
                        // Resize to standard
                        resize(faceSegmented,faceSegmented,Size(faceSize,faceSize));
                        // Send segmented face to yarp output... e.g. SAM face recog
                        utilsObj->convertCvToYarp(faceSegmented,faceImages);
                        imageOut.write();
                        //cout << "Sending face to output port" << endl;
                        // LB: ADAPTIVE SKIN MASKING SECTION -> uses skin on face to refine the skin detector......
                        // LB testing -> get all values from skin mask 
                        // Run through pixels in mask and 
                        // Needs to be more efficient... see image scan opencv documentation...
                        
                        // HSV pixels that pass skin detection -> uses face track and segmented face data
                        std::vector<Mat> hsvPixels(3);

                        //cvtColor(faceSegmented, face_HSV, CV_BGR2HSV);


                        // LB testing -> WEIRD DISPLAY EFFECT... shows quarter of image
                        
                        /*Point ptf1(currentFaceRect.x + currentFaceRect.width, currentFaceRect.y + currentFaceRect.height);
				        Point ptf2(currentFaceRect.x, currentFaceRect.y);
				        rectangle(skinHSV,ptf1,ptf2,Scalar(0,100,0),1,8,0);	
                        
                        imshow("HSV all skin",skinHSV);
                        imshow("HSV face",face_HSV);
                        imshow("Face cut",faceOut);
                        imshow("allFaces",allFaces);
                        imshow("allFacesSkin",allFacesSkin);
                        waitKey(1);
                        cout << "Face HSV rows:" << face_HSV.rows << " Faces HSV cols: " << face_HSV.cols << endl;
                        cout << "Face cut:" << faceOut.rows << " Face cut: " << faceOut.cols << endl;
                        */
                        
                        // Little effect on frame rate here....
                        // Loop through rows
                        for (int i =0; i < face_HSV.rows; i++)
                        {
                        // Loop through cols
                            for (int j =0; j < face_HSV.cols; j++)
                            {
                                //cout << "faceSegMask pix val:" << faceSegMask.at<int>(i,j) << endl;
                                // check pixels in face mask
                                //if  (faceSegMask)
                                //if  (faceSegMask.at<int>(i,j)>0)
                                //{   // Check HSV is not zero....
                                if (face_HSV(i,j)[0]!=0 || face_HSV(i,j)[1]!=0 || face_HSV(i,j)[2]!=0)
                                {
                                    // save HSV values from pixel in mask    
                                    hsvPixels[0].push_back(face_HSV(i,j)[0]); //H
                                    hsvPixels[1].push_back(face_HSV(i,j)[1]); //S
                                    hsvPixels[2].push_back(face_HSV(i,j)[2]); //V
                                }
                            }
                        }
                        
                        //imshow("HSV seg face",face_HSV);
                        
                        // Update adaptive skin detection vector.... for person specific detection....
                        hsvAdaptiveValues = utilsObj-> updateHSVAdaptiveSkin(hsvPixels, false);
                        
                        faceSegMaskInv = faceSegMask.clone();
                        //imshow("facemaskinv",faceSegMaskInv);                 
                        faceSegFlag=true;
                        targetPort.write(); // LB CHECK THIS WORKS!!!!!!!!!!!!!!!!!!!!!!!!
                    }
                    else
                    {//LB warning disabled flag here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        //faceSegFlag=false;
                        cout << " Face segmentation unsuccessful" << endl;
                    } 
                }                          
			}
		    
		    //cout << "Got to body seg 0..." << endl;
		    
		    // ##################################################
            // ############### BODY TRACK
            // #####################################################
		    if(noBodies != 0)
		    {
//			    cout << "Number of bodies: " << noBodies << endl;
                // copy in last skin image
			    captureFrameBody=captureFrameBGR.clone();
			    //std::vector<cv::Mat> bodyVec;
			    //std::vector<cv::Mat> bodyVecSkin;
			
			    noBodies = 1;

			    Mat allBodies;//(bodySize,1,CV_8UC3,count);
                Mat allBodiesSkin;//(bodySize,1,CV_8UC3,count);
                
			    objBufBodyGPU.colRange(0,noBodies).download(vectBodyArr);

				Rect* bodiesOld = vectBodyArr.ptr<Rect>();

                int i = 0;
                
                // LB - expand rectangle using additional pixels in boxScaleFactor
                if (boxScaleFactor != 0)
                {
                    //bodiesOld[i].x=bodiesOld[i].x-boxScaleFactor;
                    //bodiesOld[i].y=bodiesOld[i].y-boxScaleFactor;
                    //bodiesOld[i].width=bodiesOld[i].width+(boxScaleFactor*2);
                    //bodiesOld[i].height=bodiesOld[i].height+(boxScaleFactor*2);
                    
                    bodiesOld[i] = Rect(bodiesOld[i].x-boxScaleFactor,bodiesOld[i].y-boxScaleFactor,bodiesOld[i].width+(boxScaleFactor*2),bodiesOld[i].height+(boxScaleFactor*2));

                    // LB - Check the extra sizes are not outside the original image size
                    // WARNING -> MIGHT produce distortions -> could reject image instead...
                    bodiesOld[i]=utilsObj->checkRoiInImage(captureFrameRaw, bodiesOld[i]); // LB: seg fault (need to pass rect inside of vector...)
                }
                
                //Mat vecSizes = Mat::zeros(noBodies,1,CV_16UC1);
				//vecSizes.at<unsigned short>(i) = bodiesOld[i].width;
							
							
				//cout << "Got to body seg 1..." << endl;	
				if( displayBodies )
				{		
				    //Draw rectangle for body and line down centre
				    Point pt1(bodiesOld[i].x + bodiesOld[i].width, bodiesOld[i].y + bodiesOld[i].height);
				    Point pt2(bodiesOld[i].x, bodiesOld[i].y);
				    rectangle(captureFrameBody,pt1,pt2,Scalar(0,255,0),1,8,0);			
				    line(captureFrameBody,Point(sagittalSplit,0),Point(sagittalSplit,height),Scalar(0,0,255),1,8,0);						
                    imshow("Body seg",captureFrameBody);
                }
                
				// LB: CHECK sagittal split is sensible -> around the middle of the image (15%of either side).....
				// Body split using centre of body region found...
				sagittalSplit = int(bodiesOld[i].x+(bodiesOld[i].width/2));
				
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
				
				//Mat indices;
				//sortIdx(vecSizes, indices, SORT_EVERY_COLUMN | SORT_DESCENDING);
					
				//for(int i = 0; i<noBodies; i++)
				//{
				//cout << "Got to body seg 2..." << endl;
					if(bodiesOld[i].area() != 0)
					{
					    // Standard image facedetector, take original image
						//Mat temp = captureFrameBGR.operator()(bodiesOld[i]).clone();
						allBodies = captureFrameBGR.operator()(bodiesOld[i]).clone();
						resize(allBodies,allBodies,Size(bodySize,bodySize));
						//bodyVec.push_back(allBodies);
						// LB processed skin segmented data
						allBodiesSkin = skinImage.operator()(bodiesOld[i]).clone();
						resize(allBodiesSkin,allBodiesSkin,Size(bodySize,bodySize));
						//bodyVecSkin.push_back(temp2);
						
					}
				//}

				//hconcat(faceVec,allFaces); // LB original code -> segmented face from original data
				//hconcat(bodyVec,allBodies);					
                //hconcat(bodyVecSkin,allBodiesSkin);
                        
			}
			
	    //} // temp test
			
			
		// #####################################################################
        // LB: Body segmentation to find arms for action detection
        // ########################################################
        //cout << "Got here 1 Face seg:" << faceSegFlag << " Body seg:" << bodySegFlag << " face seg empty=" << faceSegMaskInv.empty() << endl;
        
            if (!faceSegMaskInv.empty() && faceSegFlag && bodySegFlag)
            {
                //cout << "Got to arm seg 1..." << endl;
                Mat skinImageTemp;
                Mat3b skinHSVtemp;
                Mat skinMask;
                // LB redo skin masking but with adaptive filter
                // DISABLED 
                //skinImageTemp = utilsObj->skinDetect(captureFrameBGR, &skinHSVtemp, &skinMask, hsvAdaptiveValues, 400,7, 3, 0, displayFaces);   

                skinMask = skinMaskDefault.clone();
                
                // LB: Remove face skin region.... this could be improved!!!!!
			    Mat rectMaskFaceOnly = Mat::zeros( skinMask.size(), CV_8UC1 );
			    Mat skinMaskNoFace;
			    Mat faceSegTemp;
				resize(faceSegMaskInv,faceSegTemp,Size(currentFaceRect.width,currentFaceRect.height));
			    faceSegTemp.copyTo(rectMaskFaceOnly(currentFaceRect) );
			    bitwise_not(rectMaskFaceOnly,rectMaskFaceOnly);
		        bitwise_and(rectMaskFaceOnly,skinMask,skinMaskNoFace);
                //cout << "Got to arm seg 2..." << endl;		        
		        if( displayBodies )
		        {
			        imshow("Rectangle mask face",rectMaskFaceOnly);
                    imshow("skinmask no face :)",skinMaskNoFace);	
                    //imshow("skin seg WITH ADAPTIVE SKIN SEG...... ",skinImageTemp); 				    
                }
                
                // FOR ARM TRACKING draw over face in skin mask... facemask
			    // Send to skeleton fn here
			    Mat skelMat;
			    //skelMat=utilsObj->skeletonDetect(skinMaskNoFace, imgBlurPixels, displayBodies);
			    
			    // FIND LEFT AND RIGHT ARM REGIONS....
			    // LB: currently to set to a minimum of two regions > has to find both arms!
		        vector<Rect> boundingBox = utilsObj->segmentLineBoxFit(skinMaskNoFace, 1000, 2, &skelMat, &returnContours, &armRotatedRects, displayFaces);
		        
		        // LB -> SKELMAT not used could remove to speed up!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!^^^^^!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
		        
		        
			    //check atleast two bounding boxes found for left and right arms... > has to find both arms!
			    //cout << "Got to arm seg 3..." << endl;
			    if (boundingBox.size()>1)
			    {
                    int leftArmInd=0;
                    int rightArmInd=0;
                    int testInXMost=0;
                    int testInXLeast=captureFrameFace.cols;
			        // Find boxes for left and right arms..... could be improved to cope with x-over the centre line of person!
			        
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
                    //cout << "Got to arm seg 4..." << endl;
                    // Check both boundingBoxes aren't the same for left and right 
                    if (rightArmInd!=leftArmInd)
                    {
                    
                        // LB: Testing -> find approx centre of person....
                        if (sagittalSplit!=0)
                        {
                            bodyCentre.x=sagittalSplit;
                            bodyCentre.y=currentFaceRect.y+currentFaceRect.height;
                            // Draw circle at centre below face
                            circle(captureFrameFace,bodyCentre,10,Scalar(0,255,255),3);
                        
                        }
                        
                        // ######### LEFT ARM
                        Point2f leftArmPoints[4];                        
                        armRotatedRects[leftArmInd].points(leftArmPoints);
                        vector<Point2f> leftArmMiddlePoint;
                        
                        // Find middle point at base of left arm and track...
                        if( calibratedLeftPoints == false )
                        {                            
                            if( (abs(bodyCentre.x-armRotatedRects[leftArmInd].center.x) > 40.0) && (abs(bodyCentre.y-armRotatedRects[leftArmInd].center.y) > 40.0) )
                            {
                                int longestLeftIndex = utilsObj->updateArmPoints(bodyCentre, leftArmPoints,1);   //finds initial longest point
                                previousLeftArmPoints = leftArmPoints[longestLeftIndex];
                                leftArmMiddlePoint= utilsObj->updateArmMiddlePoint(previousLeftArmPoints, leftArmPoints,1);   //finds initial longest point
                                calibratedLeftPoints = true;                             
                            }                           
                        }
                        //cout << "Got to arm seg 5..." << endl;                 
                        // Find current point which is closest to previous point
                        //int closestLeftIndex = utilsObj->updateArmPoints(previousLeftArmPoints, leftArmPoints, 0);  //finds closest point
                        leftArmMiddlePoint = utilsObj->updateArmMiddlePoint(previousLeftArmPoints, leftArmPoints,0);   //finds initial longest point
                        // Set left arm location
                        // Update previous point
                        left_hand_position = leftArmMiddlePoint.at(2);
                        previousLeftArmPoints=left_hand_position;
                        
             
/*                        for (int i=0;i<4;i++)
                        {
                            char buffer[100];
                            sprintf(buffer,"%d",i);
                            putText(captureFrameFace, buffer, leftArmPoints[i], FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255), 1, 8);
                        }
*/                                                
                        //circle(captureFrameFace, left_hand_position, 10, Scalar(0,255,0), 3);
                        //circle(captureFrameFace, leftArmMiddlePoint.at(0), 10, Scalar(255,255,0), 3);    //first closest point
                        //circle(captureFrameFace, leftArmMiddlePoint.at(1), 10, Scalar(255,255,0), 3);    //second closest point
                        circle(captureFrameFace, leftArmMiddlePoint.at(2), 10, Scalar(255,0,255), 3);    //middle point
                        
		            	//Draw left arm rectangles
				        Point pt1(boundingBox[leftArmInd].x + boundingBox[leftArmInd].width, boundingBox[leftArmInd].y + boundingBox[leftArmInd].height);
				        //Point pt2(boundingBox[leftArmInd].x, boundingBox[leftArmInd].y);
				        //rectangle(captureFrameFace,pt1,pt2,Scalar(0,0,255),1,8,0);
				        utilsObj->drawRotatedRect(captureFrameFace, armRotatedRects[leftArmInd], Scalar(255,0,0));
				        captureFrameFace=addText("Left arm", captureFrameFace, pt1, Scalar(0,0,255));
			            
			            // ######### Right ARM                           
		            	//Draw right arm rectangles
				        Point pt3(boundingBox[rightArmInd].x + boundingBox[rightArmInd].width, boundingBox[rightArmInd].y + boundingBox[rightArmInd].height);
				        //Point pt4(boundingBox[rightArmInd].x, boundingBox[rightArmInd].y);
				        //rectangle(captureFrameFace,pt3,pt4,Scalar(0,0,255),1,8,0);
				        utilsObj->drawRotatedRect(captureFrameFace, armRotatedRects[rightArmInd], Scalar(255,0,0));
				        captureFrameFace=addText("Right arm", captureFrameFace, pt3, Scalar(0,0,255));			    
			            
                      
                        // @@@@@@@@@@@@@@@@@@@@@@ Are the hands moving @@@@@@@@@@@@@@@@@@@@@
                        //Set number of pixels to detect hand movement....
                        int limitWindow = 3; // ################################### WARNING LB MOD HERE FROM 5 to 3.....
                        // IS the left hand moving
                        if( !firstLeftHandMovement )
                        {
                            previous_left_hand_position = left_hand_position;
                            firstLeftHandMovement = true;
                        }
                        
                        // Relative positions (take difference from start.....)
                        int relLeftXPosition = 0;
                        int relLeftYPosition = 0;

                        if( utilsObj->isHandMoving(left_hand_position,previous_left_hand_position, limitWindow) )
                        {
                            relLeftXPosition = left_hand_position.x - previous_left_hand_position.x;
                            relLeftYPosition = left_hand_position.y - previous_left_hand_position.y;
                        }

                        // Send out hand positions over yarp
                        // Add values to body part pos vector (left arm x(6),y(7),z(8))
                        bodyPartLocations[6]=relLeftXPosition;// left hand  x
                        bodyPartLocations[7]=relLeftYPosition;// left hand  y
                        bodyPartLocations[8]=1.0;// left hand  z -> ++++++++++++++++++ SET AT DEFAULT 1 for NOW NEED TO UPDATE LATER...... STEREOVISION
                        bodyPosFound=true; // position found -> set flag to on
                        previous_left_hand_position = left_hand_position;                          
                        //cout << "Got to arm seg 5..." << endl;
                        // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        // REPEAT IT ALL FOR the Right Arm
                        // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        Point2f rightArmPoints[4];
                        armRotatedRects[rightArmInd].points(rightArmPoints);
                        vector<Point2f> rightArmMiddlePoint;

                        if( calibratedRightPoints == false )
                        {
                            if( (abs(bodyCentre.x-armRotatedRects[rightArmInd].center.x) > 40.0) && (abs(bodyCentre.y-armRotatedRects[rightArmInd].center.y) > 40.0) )
                            {
                                int longestRightIndex = utilsObj->updateArmPoints(bodyCentre, rightArmPoints, 1);   //finds initial longest point
                                previousRightArmPoints = rightArmPoints[longestRightIndex];
                                rightArmMiddlePoint= utilsObj->updateArmMiddlePoint(previousRightArmPoints, rightArmPoints,1);   //finds initial longest point

                                calibratedRightPoints = true;
                            }                           
                        }
                        // Find current point which is closest to previous point
                        //int closestRightIndex = utilsObj->updateArmPoints(previousRightArmPoints, rightArmPoints, 0);   //finds closest point
                        rightArmMiddlePoint = utilsObj->updateArmMiddlePoint(previousRightArmPoints, rightArmPoints,0);   //finds initial longest point
                        // Set right arm location
                        //right_hand_position=rightArmPoints[closestRightIndex];
                        right_hand_position = rightArmMiddlePoint.at(2);

                        // Update previous point
                        previousRightArmPoints=right_hand_position;
                                            
/*                        for (int i=0;i<4;i++)
                        {
                            char buffer[100];
                            sprintf(buffer,"%d",i);
                            putText(captureFrameFace, buffer, rightArmPoints[i], FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255), 1, 8);
                        }
*/
                        //circle(captureFrameFace, right_hand_position,10,Scalar(0,255,0),3);
                        //circle(captureFrameFace, rightArmMiddlePoint.at(0), 10, Scalar(255,255,0), 3);    //first closest point
                        //circle(captureFrameFace, rightArmMiddlePoint.at(1), 10, Scalar(255,255,0), 3);    //second closest point
                        circle(captureFrameFace, rightArmMiddlePoint.at(2), 10, Scalar(255,0,255), 3);    //middle point
                        
                        if( !firstRightHandMovement )
                        {
                            previous_right_hand_position = right_hand_position;
                            firstRightHandMovement = true;
                        }

                        int relRightXPosition = 0;
                        int relRightYPosition = 0;

                        if( utilsObj->isHandMoving(right_hand_position,previous_right_hand_position, limitWindow) )
                        {
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
//		                cout << "Only one arm found....." << endl;	    

    		            calibratedLeftPoints = false;
		                calibratedRightPoints = false;
			        }
			    }
			    else
		        {
		            calibratedLeftPoints = false;
		            calibratedRightPoints = false;
//		            cout << "No arms found....." << endl;	    
			    }
			                    //cout << "Got to end..." << endl;
//			    if(displayFaces) 
                // Main display here..... ALways on -> shows Face / Body and arms with hand locations detected
                if (addFrameRate)
                {
                    double frameRate = utilsObj->getFrameRate(startTime);
                    
                    std::stringstream ss(stringstream::in | stringstream::out);
                    ss << setprecision(2) << frameRate <<  " fps";
                    std::string str = ss.str();   
                    captureFrameFace=addText(str, captureFrameFace, Point2f(200,15), Scalar(0,0,255));		
                }
                
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
    waitKey(1);
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
        bodyPartLocations.push_back(0.0); // 0.0 is no position found....

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
    return 0.01;
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



