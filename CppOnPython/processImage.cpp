#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <iostream>

namespace py = pybind11;
using namespace std;


static void process(cv::Mat &h_input, cv::Mat &h_output){
    cv::cuda::GpuMat d_src, d_resoult;
    d_src.upload(h_input);
    cv::cuda::resize(d_src, d_resoult, cv::Size(500, 500));
    int thresh = 50;
    int maxval = 255;
    cv::cuda::threshold(d_resoult, d_resoult, thresh, maxval, cv::THRESH_BINARY);
    d_resoult.download(h_output);
}

cv::Mat thresholdImage(string filename="template.png"){
    cv::Mat h_src = cv::imread(filename, cv::IMREAD_UNCHANGED);
    if(h_src.empty()){
        cout << "Could not load " << filename << endl;
    }

    cv::Mat h_resoult;
    process(h_src, h_resoult);
    imwrite("resoult.png", h_resoult);
    
    vector<uchar> array;
    if (h_resoult.isContinuous()) {
        array.assign((uchar*)h_resoult.datastart, (uchar*)h_resoult.dataend);
    } else {
        for (int i = 0; i < h_resoult.rows; ++i) {
            array.insert(array.end(), h_resoult.ptr<uchar>(i), h_resoult.ptr<uchar>(i)+h_resoult.cols);
        }
    }
    return h_resoult;
}


PYBIND11_MODULE(imageProcessor, m){
    m.def("thresholdImage",
            &thresholdImage,
            "thersholds image",
            py::arg("s") = "template.png");

    pybind11::class_<cv::Mat>(m, "Image", pybind11::buffer_protocol())
    .def_buffer([](cv::Mat& im) -> pybind11::buffer_info{
        return pybind11::buffer_info(
            im.data,
            sizeof(unsigned char),
            pybind11::format_descriptor<unsigned char>::format(),
            3,
            { im.rows, im.cols, im.channels() },
            {
                sizeof(unsigned char) * im.channels() * im.cols,
                sizeof(unsigned char) * im.channels(),
                sizeof(unsigned char),
            }
        );
    });

    m.doc() = "Threshold plugin";
}