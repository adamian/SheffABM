#include <stdio.h>
#include <opencv/cv.h>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;
using std::cout;


class skinDetector
{
    private:
        bool singleRegionChoice; // On = Find single largest region of skin
        bool verboseOutput; // Turn on to show image processing steps

        int minPixelSize; // Minimum pixel size for keeping skin regions!

	    int imgBlurPixels;//7, 15; // Number of pixels to smooth over for final thresholding
	    int imgMorphPixels; //7, 9; // Number pixels to do morphing over Erode dilate etc....

    public:
        skinDetector();
        ~skinDetector();
        Mat detect(Mat, bool);
        Mat cannySegmentation(Mat, int);
};
