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


class visionUtils:
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
        convertCvToYarp(Mat , ImageOf<PixelRgb> &);
        Rect checkRoiInImage(Mat src, Rect roi);
        Mat segmentEllipse(Mat, Mat, bool)
}

