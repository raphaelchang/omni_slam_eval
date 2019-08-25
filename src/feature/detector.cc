#include "detector.h"

namespace omni_slam
{
namespace feature
{

void Detector::Detect(data::Frame &frame, std::vector<data::Landmark> &landmarks)
{
    cv::Mat noarr;
    DetectInRegion(frame, landmarks, noarr);
}

void Detector::DetectInRectangularRegion(data::Frame &frame, std::vector<data::Landmark> &landmarks, cv::Point2f start, cv::Point2f end)
{
    cv::Mat mask = cv::Mat::zeros(frame.GetImage().size(), CV_8U);
    mask(cv::Rect(start, end)) = 255;
    DetectInRegion(frame, landmarks, mask);
}

void Detector::DetectInRadialRegion(data::Frame &frame, std::vector<data::Landmark> &landmarks, double start_r, double end_r, double start_t, double end_t)
{
    cv::Mat mask = cv::Mat::zeros(frame.GetImage().size(), CV_8U);
    for (int i = 0; i < mask.rows; i++)
    {
        for (int j = 0; j < mask.cols; j++)
        {
            double x = j - mask.cols / 2. + 0.5;
            double y = i - mask.rows / 2. + 0.5;
            double r = sqrt(x * x + y * y);
            double t = atan2(y, x);
            if (r >= start_r && r < end_r && t >= start_t && t < end_t)
            {
                mask.at<char>(i, j) = 255;
            }
        }
    }
    DetectInRegion(frame, landmarks, mask);
}

void Detector::DetectInRegion(data::Frame &frame, std::vector<data::Landmark> &landmarks, cv::Mat &mask)
{
    std::vector<cv::KeyPoint> kpts;
    detector_->detect(frame.GetImage(), kpts, mask);
    for (cv::KeyPoint kpt : kpts)
    {
        data::Feature feat(frame, kpt);
        data::Landmark landmark;
        landmark.AddObservation(feat);
        landmarks.push_back(landmark);
    }
}

}
}
