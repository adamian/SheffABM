
#include "visionUtils.h"

visionUtils::visionUtils()
{
}

visionUtils::~visionUtils()
{
}

void visionUtils::convertCvToYarp(cv::Mat MatImage, ImageOf<PixelRgb> & yarpImage)
{
	IplImage* IPLfromMat = new IplImage(MatImage);

	yarpImage.resize(IPLfromMat->width,IPLfromMat->height);

	IplImage * iplYarpImage = (IplImage*)yarpImage.getIplImage();

	if (IPL_ORIGIN_TL == IPLfromMat->origin){
			cvCopy(IPLfromMat, iplYarpImage, 0);
	}
	else{
			cvFlip(IPLfromMat, iplYarpImage, 0);
	}

	if (IPLfromMat->channelSeq[0]=='B') {
			cvCvtColor(iplYarpImage, iplYarpImage, CV_BGR2RGB);
	}
}

Rect visionUtils::checkRoiInImage(Mat src, Rect roi)
{
    // Get image sizes
    Size s = src.size();
    int height = s.height;
    int width = s.width;
    
    //cout << "Rect-> x:" << boundRect.x << " y:" << boundRect.y << " h:" << boundRect.height << " w:" << boundRect.width << endl;
    if (roi.x<0)
    {
        cout << "Trimming x from: " << roi.x << " to: 0" << endl;
        roi.x=0;
    }
    if (roi.y<0)
    {
        cout << "Trimming y from: " << roi.y << " to: 0" << endl;
        roi.y=0;
    }
    if ((roi.width+roi.x)>width) 
    {
        int temp=roi.width;
        roi.width=width-roi.x;
        cout << "Trimming width from: " << temp << " to: " << roi.width << endl; 
    }
    if ((roi.height+roi.y)>height)
    {
        int temp=roi.height;
        roi.height=height-roi.y;
        cout << "Trimming height from: " << temp << " to: " << roi.height << endl; 
    }
    return roi;
}


Mat visionUtils::segmentEllipse(Mat srcImage, Mat maskImage, bool displayFaces, Mat *skinSegMaskInv)
{
    RNG rng(12345); // for colour generation

    // Check mask and original image are the same size

    Size srcS = srcImage.size();
    int heightS = srcS.height;
    int widthS = srcS.width;

    Size maskS = maskImage.size();
    int heightM = maskS.height;
    int widthM = maskS.width;    

    if (heightS!=heightM || widthS!=widthM)
    {
        cout << "hS:" << heightS << " wS:" << widthS << " hM:" << heightM << " wM" << widthM << endl;  
        cout << "Source and mask images are not the same size... aborting" << endl;
        Mat ttt;
        return (ttt);
    }
  
    /// Convert image to gray and blur it
    cvtColor( maskImage, src_gray, CV_BGR2GRAY );
    blur( src_gray, src_gray, Size(3,3) );
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    /// Detect edges using Threshold
    //threshold( src_gray, src_gray, thresh, 255, THRESH_BINARY );
    /// Find contours
    findContours( src_gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    // ########## Remove contour indents (defects), by finding the convex 
    /// Find the convex hull object for each contour
    vector<vector<Point> >hull( contours.size() );

    for( int i = 0; i < contours.size(); i++ )
    {  
        convexHull( Mat(contours[i]), hull[i], false );
    }

    /// Draw contours + hull results
    Mat drawingHull = Mat::zeros( src_gray.size(), CV_8UC3 );
    
    //Check minimum contour size and find largest....
    int largest_area=-1;
    int largest_contour_index=0;
    
    for( int i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawingHull, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
        drawContours( drawingHull, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
        if( contours[i].size() > minContourSize )
        { 
            double a=contourArea( contours[i],false);  //  Find the area of contour
            if(a>largest_area)
            {
                largest_area=a;
                largest_contour_index=i;  

            }
        }
    }

    if (displayFaces)
    {
        namedWindow( "Contour Convex Hull", CV_WINDOW_AUTOSIZE );
        imshow( "Contour Convex Hull", drawingHull );
    }

  //// ############### Selected Hull contour to use -> ignoring ellipse etc
  // Check if hull found successfully... if not ABORT
    if (hull.empty() )
    {
        cout << "Hull region not found > returning...." << endl;
        Mat ttt;
        return (ttt);
    }

    // Check area of hull and abort if neded  
    double area0 = contourArea(hull[largest_contour_index]);
    vector<Point> approx;
    approxPolyDP(hull[largest_contour_index], approx, 5, true);
    double area1 = contourArea(approx);
    cout << "area0 =" << area0 << endl << "area1 =" << area1 << endl << "approx poly vertices: " << approx.size() << endl;
    if  (area1<5000)
    {
        cout << "Hull area too small > returning...." << endl;
        Mat ttt;
        return (ttt);
    }

    // Cut down rect around convex contour hull 
    Rect boundRect;
    boundRect=boundingRect(Mat(hull[largest_contour_index]));
    // Check bounding box fits inside image.... resize if needed
    boundRect=checkRoiInImage(srcImage, boundRect);
    //cout << "Bd h: " << boundRect.height << " Bd w: " << boundRect.width << " Bd x: " << boundRect.x << " Bd y: " << boundRect.y << endl;

    // Check bounding box has greater dimensions than 5x5pix
    if (boundRect.height<=5 || boundRect.width<=5)
    {
        cout << "Region selected too small... exiting" << endl;
        Mat ttt;
        return (ttt);
    }
    else
    {
        /// Take boxed region of face from original image data
        // Copy inital image and section with bounding box
        if (displayFaces)
        {
            Mat srcSegmented = srcImage.clone();
            srcSegmented=srcSegmented(boundRect);
            namedWindow( "Rect region orig", CV_WINDOW_NORMAL );
            imshow("Rect region orig",srcSegmented);
        }

        // Repeat for mask image
        if (displayFaces)
        {      
            Mat maskSegmented = maskImage.clone();
            maskSegmented=maskSegmented(boundRect);
            namedWindow( "Rect region, with SkinSeg", CV_WINDOW_NORMAL );
            imshow("Rect region, with SkinSeg",maskSegmented);      
        }

        /// Repeat boxing but for masked skin data (Hull)
        // Make binary mask using hull largest contour
        Mat srcSegSkin = Mat::zeros( srcImage.size(), CV_8UC3 );
        Mat skinSegMask = Mat::zeros( srcImage.size(), CV_8UC1 );
        drawContours( skinSegMask, hull, largest_contour_index, Scalar(255), -1, 8, vector<Vec4i>(), 0, Point() );
        srcImage.copyTo(srcSegSkin,skinSegMask);  // Copy using mask from skinSegMask
        srcSegSkin=srcSegSkin(boundRect);
        
        // Make face blocking mask (face pix = 0)
//        Mat skinSegMaskInvTemp = Scalar::all(255)-skinSegMask;
        Mat skinSegMaskInvTemp = Mat::zeros( srcImage.size(), CV_8UC1 );
        bitwise_not(skinSegMaskInvTemp,*skinSegMaskInv,skinSegMask);

        if (displayFaces)
        {
            namedWindow( "Rect region, with hull region SkinSeg", CV_WINDOW_NORMAL );
            imshow("Rect region, with hull region SkinSeg",srcSegSkin);      
        }      

        return(srcSegSkin);
    }
}
/*
vector<Rect> visionUtils::getArmRects(Mat threshImage, int imgBlurPixels, Mat *skelSeg, bool displayFaces)
{
    // Find skeleton in bw image (using skin seg)
    // Returns sekleton segmented image 
    //LB TEST
    //Mat skel = skeletonDetect(threshImage, imgBlurPixels, displayFaces);
    
    Mat skel = threshImage.clone();
    
    // Find contours use canny seg...and put bouding boxes around segmented regions
    // 2= no of segments = 2 arms
    vector<Rect> boundingBox = segmentLineBoxFit(skel, 100, 2, skelSeg , displayFaces);
    // Return bouding boxes around arms....
    return boundingBox;
}
*/

Mat visionUtils::skeletonDetect(Mat threshImage, int imgBlurPixels, bool displayFaces)
{
    // Finds skeleton of white objects in black image, using erode / dilate morph operations

    threshold(threshImage, threshImage, 127, 255, THRESH_BINARY);
    if (displayFaces) imshow("Skel in",threshImage);
    Mat skel(threshImage.size(), CV_8UC1, Scalar(0));
    Mat temp;
    Mat eroded;
 
    Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
 
    bool done;		
    do
    {
	    erode(threshImage, eroded, element);
	    dilate(eroded, temp, element); // temp = open(threshImage)
	    subtract(threshImage, temp, temp);
	    bitwise_or(skel, temp, skel);
	    eroded.copyTo(threshImage);
	    //cout << "Zero count: " << countNonZero(threshImage) << endl;
	    done = (countNonZero(threshImage) == 0);
    } while (!done);
    if (displayFaces) imshow("Skel raw",skel);
    // Blur to reduce noise
    GaussianBlur(skel, skel, Size(imgBlurPixels,imgBlurPixels), 1, 1);
    return skel;
}

//bool visionUtils::compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 )
bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 )
{
    double i = fabs( contourArea(cv::Mat(contour1)) );
    double j = fabs( contourArea(cv::Mat(contour2)) );
    return ( i < j );
}

vector<Rect> visionUtils::segmentLineBoxFit(Mat img0, int minPixelSize, int maxSegments, Mat *returnMask, std::vector<std::vector<cv::Point> > *returnContours,bool displayFaces)
{
    // Segments items in gray image (img0)
    // minPixelSize= pixels, threshold for removing smaller regions, with less than minPixelSize pixels
    // 0, returns all detected segments
    // maxSegments = max no segments to return, 0 = all
    RNG rng(12345);

    
    // apply your filter
    //Canny(img0, img1, 100, 200, 3); //100, 200, 3);
    
    // LB: Zero pad image to remove edge effects when getting regions....	
    int padPixels=20;
    // Rect border added at start...
    Rect tempRect;
    tempRect.x=padPixels;
    tempRect.y=padPixels;
    tempRect.width=img0.cols;
    tempRect.height=img0.rows;
    Mat img1 = Mat::zeros(img0.rows+(padPixels*2), img0.cols+(padPixels*2), CV_8UC1);
    img0.copyTo(img1(tempRect));

    // find the contours
    std::vector<std::vector<cv::Point> > contours;
    vector<Vec4i> hierarchy;

    findContours(img1, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    // Mask for segmented region
    Mat mask = Mat::zeros(img1.rows, img1.cols, CV_8UC3);

    //Mat returnMask = Mat::zeros(img1.rows, img1.cols, CV_8UC3);
    
    vector<double> areas(contours.size());

    // Case for using minimum pixel size
    Vec4f lines;
    Scalar color;
    
    // sort contours
    std::sort(contours.begin(), contours.end(), compareContourAreas);

    // grab contours
    //std::vector<cv::Point> firstContour = contours[contours.size()-1];
    //std::vector<cv::Point> secondContour = contours[contours.size()-2];
    
//    cout << "No of contours =" << contours.size() << endl;
    vector<Rect> boundingBox;
    std::vector<std::vector<cv::Point> > tempReturnContours;
    int maxIterations = 0;
    
    if( contours.size() > 0 )
    {
        if (maxSegments==0)// return all contours..
            maxIterations = contours.size();
        else if(contours.size() >= maxSegments)
            maxIterations = maxSegments;
        else
            maxIterations = 1;    // LB: need to check this is correct!
        int contourCount=0;
        
        for (int j = 1; j < maxIterations+1; j++)
        {
            int i = contours.size()-j;
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
		        //padPixels
		        boundingBox.push_back(boundingRect(Mat(contours[i])));

		        // LB OPTIONAL LINE FITTING HERE...
		        //int leftBoxy = ((boundingBox[contourCount].x-lines[2])*lines[1]/lines[0])+lines[3];
		        //int rightBoxy = (((boundingBox[contourCount].x+boundingBox[contourCount].width)-lines[2])*lines[1]/lines[0])+lines[3];
		        //line(mask,Point((boundingBox[contourCount].x+boundingBox[contourCount].width)-1,rightBoxy),Point(boundingBox[contourCount].x,leftBoxy),color,2);
		        
		        // Remove edge padding effects....
		        boundingBox[contourCount].x=boundingBox[contourCount].x-padPixels;
		        boundingBox[contourCount].y=boundingBox[contourCount].y-padPixels;
		        
		        boundingBox[contourCount]=checkRoiInImage(img0, boundingBox[contourCount]);
		        
		        contourCount++;
		        
		        tempReturnContours.push_back(contours[i]);
	        }
        }
    
        returnContours->resize(tempReturnContours.size());
        *returnContours = tempReturnContours;
        
        // normalize so imwrite(...)/imshow(...) shows the mask correctly!
        normalize(mask.clone(), mask, 0.0, 255.0, CV_MINMAX, CV_8UC1);
        // To Remove border added at start...    
        *returnMask=mask(tempRect);
        //mask(tempRect).copyTo(*returnMask);
        // show the images
        if (displayFaces)	imshow("Seg line utils: Img in", img0);
        if (displayFaces)	imshow("Seg line utils: Mask", *returnMask);
        if (displayFaces)	imshow("Seg line utils: Output", img1);
    }
    return boundingBox;

    //vector<Vec2f> lines;
    //HoughLines(dst, lines, 1, CV_PI/180, 100, 0, 0 );

}

Mat visionUtils::cannySegmentation(Mat img0, int minPixelSize, bool displayFaces)
{
/*	// Segments items in gray image (img0)
	// minPixelSize=
	// -1, returns largest region only
	// pixels, threshold for removing smaller regions, with less than minPixelSize pixels
	// 0, returns all detected segments

    // LB: Zero pad image to remove edge effects when getting regions....	
    int padPixels=20;
    // Rect border added at start...
    Rect tempRect;
    tempRect.x=padPixels;
    tempRect.y=padPixels;
    tempRect.width=img0.cols;
    tempRect.height=img0.rows;
    Mat img1 = Mat::zeros(img0.rows+(padPixels*2), img0.cols+(padPixels*2), CV_8UC1);
    img0.copyTo(img1(tempRect));
    
	// apply your filter
    Canny(img1, img1, 100, 200, 3); //100, 200, 3);

    // find the contours
    vector< vector<Point> > contours;
    findContours(img1, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    // Mask for segmented regiond
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
			drawContours(mask, contours, i, Scalar(1), CV_FILLED);

		}
	}
    // normalize so imwrite(...)/imshow(...) shows the mask correctly!
    normalize(mask.clone(), mask, 0.0, 255.0, CV_MINMAX, CV_8UC1);
    
    // Remove border added at start...
    
    Mat returnMask;
    returnMask=mask(tempRect);
    // show the images
    if (displayFaces)	imshow("Canny: Img in", img0);
    if (displayFaces)	imshow("Canny: Mask", returnMask);
    if (displayFaces)	imshow("Canny Output", img1);
    */
    
    Mat returnMask=img0;
    
    return returnMask;
}

bool visionUtils::isHandMoving(Point handPoints, Point previousHandPoints, int limit)
{
/*    vector<Point> windowMovement(2);
    windowMovement[0].x = previousHandPoints.x - limit;
    windowMovement[0].y = previousHandPoints.y - limit;
    windowMovement[1].x = previousHandPoints.x + limit;
    windowMovement[1].y = previousHandPoints.y + limit;

    bool movement = false;
    if( ( handPoints.x < windowMovement[0].x ) || ( handPoints.x > windowMovement[1].x ) )
        movement = true;
    if( ( handPoints.y < windowMovement[0].y ) || ( handPoints.y > windowMovement[1].y ) )
        movement = true;
*/        

    bool movement = false;
    if( ( handPoints.x < previousHandPoints.x - limit ) || ( handPoints.x > previousHandPoints.x + limit ) )
        movement = true;
    if( ( handPoints.y < previousHandPoints.y - limit ) || ( handPoints.y > previousHandPoints.y + limit ) )
        movement = true;

    return movement;
}

/**
 * @Draw histogram
 * Takes in triple vector of values e.g. std::vector<Mat> bgrPixels(3);
 */
int visionUtils::drawHist(std::vector<Mat> pixelPlanes)//( int, char** argv )
{
  //Mat dst, src;

  /// Load image
  /*src = imread( argv[1], 1 );

  if( !src.data )
    { return -1; }
*/
  /// Separate the image in 3 places ( B, G and R )
  //vector<Mat> pixelPlanes;
  //split( src, pixelPlanes );

  /// Establish the number of bins
  int histSize = 256;

  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  Mat b_hist, g_hist, r_hist;

  /// Compute the histograms:
  calcHist( &pixelPlanes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &pixelPlanes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &pixelPlanes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

  // Draw the histograms for B, G and R or H S V
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

  /// Normalize the result to [ 0, histImage.rows ]
  normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

  /// Draw for each channel
  for( int i = 1; i < histSize; i++ )
  {
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                       Scalar( 0, 255, 0), 2, 8, 0  );
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                       Scalar( 0, 0, 255), 2, 8, 0  );
  }

  /// Display
  namedWindow("calcHist Demo", WINDOW_AUTOSIZE );
  imshow("calcHist Demo", histImage );

  waitKey(1);

  return 0;

}




