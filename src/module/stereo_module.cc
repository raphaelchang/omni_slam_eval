#include "stereo_module.h"

#include <algorithm>

using namespace std;

namespace omni_slam
{
namespace module
{

StereoModule::StereoModule(std::unique_ptr<stereo::StereoMatcher> &stereo)
    : stereo_(std::move(stereo))
{
}

StereoModule::StereoModule(std::unique_ptr<stereo::StereoMatcher> &&stereo)
    : StereoModule(stereo)
{
}

void StereoModule::Update(data::Frame &frame, std::vector<data::Landmark> &landmarks)
{
    stereo_->Match(frame, landmarks);

    if (frameNum_ == 0)
    {
        visualization_.Init(frame.GetImage().size());
    }
    for (data::Landmark &landmark : landmarks)
    {
        const data::Feature *feat1 = landmark.GetObservationByFrameID(frame.GetID());
        const data::Feature *feat2 = landmark.GetStereoObservationByFrameID(frame.GetID());
        if (feat1 != nullptr && feat2 != nullptr)
        {
            Matrix<double, 3, 4> framePose = frame.HasEstimatedPose() ? frame.GetEstimatedPose() : frame.GetPose();
            double depth = (landmark.GetEstimatedPosition() - framePose.block<3, 1>(0, 3)).norm();
            if (frame.HasPose() && landmark.HasGroundTruth())
            {
                double depthGnd = (landmark.GetGroundTruth() - frame.GetPose().block<3, 1>(0, 3)).norm();
                visualization_.AddMatch(feat1->GetKeypoint().pt, feat2->GetKeypoint().pt, depth, std::abs(depth - depthGnd) / depthGnd);
            }
            else
            {
                visualization_.AddMatch(feat1->GetKeypoint().pt, feat2->GetKeypoint().pt, depth, 0);
            }
        }
    }
    frameNum_++;
}

StereoModule::Stats& StereoModule::GetStats()
{
    return stats_;
}

void StereoModule::Visualize(cv::Mat &base_img, const cv::Mat &base_stereo_img)
{
    visualization_.Draw(base_img, base_stereo_img);
}

void StereoModule::Visualization::Init(cv::Size img_size)
{
    curMask_ = cv::Mat::zeros(cv::Size(img_size.width * 2, img_size.height), CV_8UC3);
    curDepth_ = cv::Mat::zeros(img_size, CV_8UC3);
}

void StereoModule::Visualization::AddMatch(cv::Point2f pt1, cv::Point2f pt2, double depth, double err)
{
    err = min(err, 1.0);
    depth = min(depth, maxDepth_);
    cv::circle(curMask_, pt1, 3, cv::Scalar(255, 0, 0), -1);
    cv::circle(curMask_, pt2 + cv::Point2f(curMask_.cols / 2., 0), 3, cv::Scalar(255, 0, 0), -1);
    cv::line(curMask_, pt1, pt2 + cv::Point2f(curMask_.cols / 2., 0), cv::Scalar(255, 0, 0), 1);
    double depthColor = 1 - depth / maxDepth_;
    depthColor *= 0.9 * depthColor;
    depthColor += 0.1;
    cv::circle(curDepth_, pt1, 2, cv::Scalar((depthColor - err) * 255, (depthColor - err) * 255, depthColor * 255));
}

void StereoModule::Visualization::Draw(cv::Mat &img, const cv::Mat &stereo_img)
{
    cv::hconcat(img, stereo_img, img);
    if (img.channels() == 1)
    {
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    }
    curMask_.copyTo(img, curMask_);
    curMask_ = cv::Mat::zeros(img.size(), CV_8UC3);
    cv::hconcat(img, curDepth_, img);
    curDepth_ = cv::Mat::zeros(stereo_img.size(), CV_8UC3);
}

}
}
