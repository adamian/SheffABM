
#include "skinDetector.h"

skinDetector::skinDetector()
{
    singleRegionChoice = 0; // On = Find single largest region of skin
    verboseOutput = 0; // Turn on to show image processing steps
    minPixelSize = 400; // Minimum pixel size for keeping skin regions!

	imgBlurPixels = 7;//7, 15; // Number of pixels to smooth over for final thresholding
	imgMorphPixels = 3; //7, 9; // Number pixels to do morphing over Erode dilate etc....
}

skinDetector::~skinDetector()
{
}

Mat skinDetector::detect(Mat captureframe, bool verboseSelect, Mat *skinMask)
{
	verboseOutput=verboseSelect;
	//if (argc>=1) frame = argv[1]);
	//if (argc>=2) singleRegionChoice = int(argv[2]);

	int step = 0;
	Mat3b frame;
	// Forcing resize to 640x480 -> all thresholds / pixel filters configured for this size.....
	// Note returned to original size at end...
    Size s = captureframe.size();
	resize(captureframe,captureframe,Size(640,480));

	
	// CHANGED HERE TO BGR
	//cvtColor(captureframe, captureframe, CV_RGB2BGR);
	if (verboseOutput)	imshow("Raw Image (A)",captureframe);
	/* THRESHOLD ON HSV*/
	// HSV data -> used to find skin
	cvtColor(captureframe, frame, CV_BGR2HSV);
	//cvtColor(captureframe, frame, CV_BGR2HLS);
	GaussianBlur(frame, frame, Size(imgBlurPixels,imgBlurPixels), 1, 1);
	//medianBlur(frame, frame, 15);
	for(int r=0; r<frame.rows; ++r){
		for(int c=0; c<frame.cols; ++c) 
			// 0<H<0.25  -   0.15<S<0.9    -    0.2<V<0.95   
			if( (frame(r,c)[0]>5) && (frame(r,c)[0] < 17) && (frame(r,c)[1]>38) && (frame(r,c)[1]<250) && (frame(r,c)[2]>51) && (frame(r,c)[2]<242) ); // do nothing
			else for(int i=0; i<3; ++i)	frame(r,c)[i] = 0;
	}

	if (verboseOutput)	imshow("Skin HSV (B)",frame);
	/* BGR CONVERSION AND THRESHOLD */
	Mat1b frame_gray;
	cvtColor(frame, frame, CV_HSV2BGR);
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
				
				
	// Adaptive thresholding technique
	// 1. Threshold data to find main areas of skin
	adaptiveThreshold(frame_gray,frame_gray,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY_INV,9,1);
	if (verboseOutput)	imshow("Adaptive_threshold (D1)",frame_gray);
	// 2. Fill in thresholded areas
	morphologyEx(frame_gray, frame_gray, CV_MOP_CLOSE, Mat1b(imgMorphPixels,imgMorphPixels,1), Point(-1, -1), 2);
	
	
	//GaussianBlur(frame_gray, frame_gray, Size((imgBlurPixels*2)+1,(imgBlurPixels*2)+1), 1, 1);
	GaussianBlur(frame_gray, frame_gray, Size(imgBlurPixels,imgBlurPixels), 1, 1);
	// Select single largest region from image, if singleRegionChoice is selected (1)
	
	if (singleRegionChoice)
	{
		*skinMask = cannySegmentation(frame_gray, -1);
	}
	else // Detect each separate block and remove blobs smaller than a few pixels
	{
		*skinMask = cannySegmentation(frame_gray, minPixelSize);
	}


	// Just return skin
	Mat frame_skin;
	captureframe.copyTo(frame_skin,*skinMask);  // Copy captureframe data to frame_skin, using mask from frame_ttt
	// Resize image to original before return
	resize(frame_skin,frame_skin,s);
	if (verboseOutput)	imshow("Skin segmented",frame_skin);
	return frame_skin;	
	waitKey(1);
}


Mat skinDetector::cannySegmentation(Mat img0, int minPixelSize)
{
	// Segments items in gray image (img0)
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


    Mat returnMask;
    returnMask=mask(tempRect);
    
    
    // show the images
    if (verboseOutput)	imshow("Canny Skin: Img in", img0);
    if (verboseOutput)	imshow("Canny Skin: Mask", returnMask);
    if (verboseOutput)	imshow("Canny Skin: Output", img1);
    

    return returnMask;
}


