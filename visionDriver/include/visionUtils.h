#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>
#include <yarp/sig/all.h>
#include <yarp/os/all.h>
#include <yarp/dev/all.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ctime>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/gpumat.hpp>
#include <iostream>
#include <math.h>
#include <algorithm>

using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::sig::draw;
using namespace yarp::sig::file;
using namespace cv;
using namespace yarp::dev;
using namespace std;
using namespace cv::gpu;


class visionUtils
{
    private:
        Mat srcImage;
        Mat src_gray;
        int thresh;
        int max_thresh;
        int minContourSize; // minimum contour size


    public:
        visionUtils();
        ~visionUtils();
        void convertCvToYarp(cv::Mat MatImage, ImageOf<PixelRgb> &yarpImage);
        Rect checkRoiInImage(Mat, Rect);
        Mat segmentEllipse(Mat, Mat, bool, Mat *);
        Mat skeletonDetect(Mat, int, bool );
        vector<Rect> segmentLineBoxFit(Mat, int, int, Mat *,  std::vector<std::vector<cv::Point> > *, bool);
        Mat cannySegmentation(Mat, int, bool);
        bool isHandMoving(Point, Point, int);
        
//        vector<Rect> getArmRects(Mat, int, Mat *, bool);
//        bool compareContourAreas(std::vector<cv::Point> c1, std::vector<cv::Point> c2);
};

