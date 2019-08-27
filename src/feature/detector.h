#ifndef _DETECTOR_H_
#define _DETECTOR_H_

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "data/frame.h"
#include "data/landmark.h"
#include <string>
#include <vector>

namespace omni_slam
{
namespace feature
{

class Detector
{
public:
    template<typename... Args>
    Detector(std::string type, Args... args)
        : type_(type)
    {
        if (type == "GFTT")
        {
            detector_ = cv::GFTTDetector::create(args...);
        }
        else if (type == "FAST")
        {
            detector_ = cv::FastFeatureDetector::create(args...);
        }
        else if (type == "SIFT")
        {
            detector_ = cv::xfeatures2d::SIFT::create(args...);
        }
        else if (type == "SURF")
        {
            detector_ = cv::xfeatures2d::SURF::create(args...);
        }
        else if (type == "ORB")
        {
            detector_ = cv::ORB::create(args...);
        }
        else if (type == "BRISK")
        {
            detector_ = cv::BRISK::create(args...);
        }
        else if (type == "STAR")
        {
            detector_ = cv::xfeatures2d::StarDetector::create(args...);
        }
        else if (type == "AKAZE")
        {
            detector_ = cv::AKAZE::create(args...);
        }
        else if (type == "KAZE")
        {
            detector_ = cv::KAZE::create(args...);
        }
        else if (type == "AGAST")
        {
            detector_ = cv::AgastFeatureDetector::create(args...);
        }
    }
    Detector(std::string type, std::map<std::string, double> args);
    int Detect(data::Frame &frame, std::vector<data::Landmark> &landmarks);
    int DetectInRectangularRegion(data::Frame &frame, std::vector<data::Landmark> &landmarks, cv::Point2f start, cv::Point2f end);
    int DetectInRadialRegion(data::Frame &frame, std::vector<data::Landmark> &landmarks, double start_r, double end_r, double start_t, double end_t);
    int DetectInRegion(data::Frame &frame, std::vector<data::Landmark> &landmarks, cv::Mat &mask);

    static bool IsDetectorTypeValid(std::string name);

private:
    cv::Ptr<cv::Feature2D> detector_;
    std::string type_;
};

}
}

#endif /* _DETECTOR_H_ */
