// g++ main.cpp -o output `pkg-config --cflags --libs opencv`
#include <string>
#include "opencv/highgui.h"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    Mat img = imread(argv[1]);
    Mat gray;
    Mat blur;
    Mat out;

    if(argc == 3){
        cvtColor(img, gray, COLOR_BGR2HSV);
        vector<Mat> channels;
        split(gray, channels);
        threshold(channels[1], out, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
    }
    else{
       cvtColor(img, gray, CV_BGR2GRAY);
       GaussianBlur(gray, blur,Size(5,5),0);
       threshold(blur, out, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
       medianBlur(out, out, 5);
    }
 //  adaptiveThreshold(blur, out, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 11, 1); // ADAPTIVE_THRESH_GAUSSIAN_C , ADAPTIVE_THRESH_MEAN_C



//    adaptiveThreshold(blur, out, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 11, 1);
//    double thresh_im = threshold(out, out, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
//    thresh = bitwise_or(adapt_thresh_im, thresh_im)

    imwrite("./gray.jpg", gray);
    imwrite("./blur.jpg", blur);
    imwrite("./black-white.jpg", out);
    imshow("input", img);
    imshow("intermediary", gray);
    imshow("black-white", out);
    waitKey(0);
    return 0;
}