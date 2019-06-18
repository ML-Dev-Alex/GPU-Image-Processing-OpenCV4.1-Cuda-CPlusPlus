#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

const int fps = 30;

void processVideo(Mat &h_input1, Mat &h_input2, Mat &h_output){
    cuda::GpuMat d_src1, d_src2, d_resoult, d_gray1, d_gray2, d_diff;
    d_src1.upload(h_input1);
    d_src2.upload(h_input2);

    cuda::cvtColor(d_src1, d_gray1, COLOR_BGR2GRAY);
    cuda::cvtColor(d_src2, d_gray2, COLOR_BGR2GRAY);
    
    cuda::subtract(d_gray1, d_gray2, d_diff);

    Ptr<cuda::Filter> gaussianFilter;
    gaussianFilter = cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(3, 3), 0.5);
    gaussianFilter->apply(d_diff, d_diff);

    Ptr<cuda::CannyEdgeDetector> canny;
    canny = cuda::createCannyEdgeDetector(70, 200);
    canny->detect(d_diff, d_resoult);
    // d_resoult.upload(h_input1);
    d_resoult.download(h_output);
}

int main(int argc, char* argv[]){
    if (cuda::getCudaEnabledDeviceCount() < 1){
        cout << "ERROR: Could not find a Cuda enabled device on this machine." << endl;
        return -1;
    } 

    Mat h_src1, h_src2;
    Mat h_resoult;

    VideoCapture cap(0);

    if (!cap.isOpened()){
        cout << "Video could not be opened." << endl;
        return -1;
    }


    // Default resolution of the frame is obtained.The default resolution is system dependent. 
    VideoWriter srcVideo, resoultVideo;
    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH); 
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT); 
    int codec = VideoWriter::fourcc('X', 'V', 'I', 'D'); 

    // Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file. 
    srcVideo.open("videos/originalMotion.mp4", codec, fps, Size(frame_width, frame_height)); 
    resoultVideo.open("videos/resoultMotion.mp4", codec, fps, Size(frame_width, frame_height), false); 
  
    while (cap.read(h_src1)){
        imshow("Webcam", h_src1);
        srcVideo.write(h_src1);
        if (waitKey((1000/fps)/2) >= 0)
            break;
        cap.read(h_src2);
        processVideo(h_src1, h_src2, h_resoult);
        imshow("Resoult", h_resoult);
        resoultVideo.write(h_resoult);
    }

    return 0;
}