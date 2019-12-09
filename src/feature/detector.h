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
        : detectorType_(type),
        descriptorType_(type)
    {
        if (type == "SIFT")
        {
            detector_ = cv::xfeatures2d::SIFT::create(args...);
            descriptor_ = cv::xfeatures2d::SIFT::create(args...);
        }
        else if (type == "SURF")
        {
            detector_ = cv::xfeatures2d::SURF::create(args...);
            descriptor_ = cv::xfeatures2d::SURF::create(args...);
        }
        else if (type == "ORB")
        {
            detector_ = cv::ORB::create(args...);
            descriptor_ = cv::ORB::create(args...);
        }
        else if (type == "BRISK")
        {
            detector_ = cv::BRISK::create(args...);
            descriptor_ = cv::BRISK::create(args...);
        }
        else if (type == "AKAZE")
        {
            detector_ = cv::AKAZE::create(args...);
            descriptor_ = cv::AKAZE::create(args...);
        }
        else if (type == "KAZE")
        {
            detector_ = cv::KAZE::create(args...);
            descriptor_ = cv::KAZE::create(args...);
        }
    }
    Detector(std::string detector_type, std::map<std::string, double> args);
    Detector(std::string detector_type, std::string descriptor_type, std::map<std::string, double> det_args, std::map<std::string, double> desc_args, bool local_unwarp = false);
    Detector(const Detector &other);

    int Detect(data::Frame &frame, std::vector<data::Landmark> &landmarks, bool stereo = false) const;
    int DetectInRectangularRegion(data::Frame &frame, std::vector<data::Landmark> &landmarks, cv::Point2f start, cv::Point2f end, bool stereo = false) const;
    int DetectInRadialRegion(data::Frame &frame, std::vector<data::Landmark> &landmarks, double start_r, double end_r, double start_t, double end_t, bool stereo = false) const;
    int DetectInRegion(data::Frame &frame, std::vector<data::Landmark> &landmarks, cv::Mat &mask, bool stereo = false) const;

    std::string GetDetectorType();
    std::string GetDescriptorType();

    static bool IsDetectorTypeValid(std::string name);
    static bool IsDescriptorTypeValid(std::string name);
    static bool IsDetectorDescriptorCombinationValid(std::string det, std::string desc);

private:
    cv::Ptr<cv::Feature2D> detector_;
    cv::Ptr<cv::Feature2D> descriptor_;
    std::string detectorType_;
    std::string descriptorType_;

    std::map<std::string, double> detectorArgs_;
    std::map<std::string, double> descriptorArgs_;

    bool localUnwarp_{false};
};

}
}

#endif /* _DETECTOR_H_ */
