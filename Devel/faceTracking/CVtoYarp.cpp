#include <opencv/cv.h>
#include <yarp/sig/all.h>

using namespace yarp::os;
using namespace cv;
using namespace yarp::sig;

void CVtoYarp(cv::Mat MatImage, ImageOf<PixelRgb> & yarpImage);


void CVtoYarp(Mat MatImage, ImageOf<PixelRgb> & yarpImage)
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
