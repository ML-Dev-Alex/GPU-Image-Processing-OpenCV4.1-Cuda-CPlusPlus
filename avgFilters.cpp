#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

static void help(){
    cout << "Usage: " << endl;
    cout << "./avgFilters <image_name>" << endl;
}


int main(int argc, const char* argv[]){
    const string filename = argc >= 2 ? argv[1] : "cover.jpg";

    Mat h_src = imread(filename, IMREAD_GRAYSCALE);

    if(h_src.empty()){
        help();
        cout << "Could not load " << filename << endl;
        return -1;
    }

    cuda::GpuMat d_img1, d_result3x3, d_result5x5, d_result7x7;

    d_img1.upload(h_src);

    cuda::resize(d_img1, d_img1, Size(), 0.2, 0.2);
    Ptr<cuda::Filter> filter3x3, filter5x5, filter7x7;

    filter3x3 = cuda::createBoxFilter(CV_8UC1, CV_8UC1, Size(3, 3));
    filter3x3->apply(d_img1, d_result3x3);

    filter5x5 = cuda::createBoxFilter(CV_8UC1, CV_8UC1, Size(5, 5));
    filter5x5->apply(d_img1, d_result5x5);

    filter7x7 = cuda::createBoxFilter(CV_8UC1, CV_8UC1, Size(7, 7));
    filter7x7->apply(d_img1, d_result7x7);

    Mat h_resized, h_result3x3, h_result5x5, h_result7x7;
    d_img1.download(h_resized);
    d_result3x3.download(h_result3x3);
    d_result5x5.download(h_result5x5);
    d_result7x7.download(h_result7x7);


    imshow("source (resized)", h_resized);
    imshow("3x3", h_result3x3);
    imshow("5x5", h_result5x5);
    imshow("7x7", h_result7x7);

    imwrite("Blurred3x3.png", h_result3x3);
    imwrite("Blurred5x5.png", h_result5x5);
    imwrite("Blurred7x7.png", h_result7x7);
    waitKey();

    return 0;
    
}

