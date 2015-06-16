#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

/** Luke, Check Roi is within the image (e.g. for overlay / cutting 
roi = send in openCV roi e.g. rect....
src = send in colour image opencv Mat 
*/
Rect checkRoiInImage(Mat src, Rect roi)
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

