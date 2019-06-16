#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

static void help(){
    cout << "Usage: " << endl;
    cout << "./threshold <image_name>" << endl;
}

static void process(Mat &h_input, Mat &h_output){
    cuda::GpuMat d_src, d_resoult;
    d_src.upload(h_input);
    int thresh = 150;
    int maxval = 255;
    cuda::threshold(d_src, d_resoult, thresh, maxval, THRESH_BINARY);
    d_resoult.download(h_output);
}

int main(int argc, char* argv[]){
    if (cuda::getCudaEnabledDeviceCount() < 1){
        cout << "ERROR: Could not find a Cuda enabled device on this machine." << endl;
        return -1;
    }

    const string filename = argc >= 2 ? argv[1] : "images/template.png";
    Mat h_src = imread(filename, IMREAD_UNCHANGED);
    if(h_src.empty()){
        help();
        cout << "Could not load " << filename << endl;
        return -1;
    }

    Mat h_resoult;

    imshow("Webcam", h_src);
    process(h_src, h_resoult);
    imshow("Resoult", h_resoult);
    imwrite("thresholdedImage.png", h_resoult);

    waitKey();
    return 0;
}
