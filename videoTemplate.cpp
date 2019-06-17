#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

const int fps = 30;

void processVideo(Mat &h_input, Mat &h_output){
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

    Mat h_src;
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
    int codec = VideoWriter::fourcc('M', 'J', 'P', 'G'); 

    // Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file. 
    srcVideo.open("videos/original.avi", codec, fps, Size(frame_width, frame_height)); 
    resoultVideo.open("videos/resoult.avi", codec, fps, Size(frame_width, frame_height)); 
  
    while (cap.read(h_src)){
        imshow("Webcam", h_src);
        srcVideo.write(h_src);
        if (waitKey(1000/fps) >= 0)
            break;
        processVideo(h_src, h_resoult);
        imshow("Resoult", h_resoult);
        resoultVideo.write(h_resoult);
    }

    return 0;
}