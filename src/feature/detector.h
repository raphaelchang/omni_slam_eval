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
            detector_.reset(cv::GFTTDetector::create(args...));
        }
        else if (type == "FAST")
        {
            detector_.reset(cv::FastFeatureDetector::create(args...));
        }
        else if (type == "SIFT")
        {
            detector_.reset(cv::xfeatures2d::SIFT::create(args...));
        }
        else if (type == "SURF")
        {
            detector_.reset(cv::xfeatures2d::SURF::create(args...));
        }
        else if (type == "ORB")
        {
            detector_.reset(cv::ORB::create(args...));
        }
        else if (type == "BRISK")
        {
            detector_.reset(cv::BRISK::create(args...));
        }
        else if (type == "STAR")
        {
            detector_.reset(cv::xfeatures2d::StarDetector::create(args...));
        }
        else if (type == "AKAZE")
        {
            detector_.reset(cv::AKAZE::create(args...));
        }
        else if (type == "KAZE")
        {
            detector_.reset(cv::KAZE::create(args...));
        }
        else if (type == "AGAST")
        {
            detector_.reset(cv::AgastFeatureDetector::create(args...));
        }
    }
    void Detect(data::Frame &frame, std::vector<data::Landmark> &landmarks);
    void DetectInRectangularRegion(data::Frame &frame, std::vector<data::Landmark> &landmarks, cv::Point2f start, cv::Point2f end);
    void DetectInRadialRegion(data::Frame &frame, std::vector<data::Landmark> &landmarks, double start_r, double end_r, double start_t, double end_t);
    void DetectInRegion(data::Frame &frame, std::vector<data::Landmark> &landmarks, cv::Mat &mask);

private:
    cv::Ptr<cv::Feature2D> detector_;
    std::string type_;
};

}
}

#endif /* _DETECTOR_H_ */
