#include "stereo_eval.h"

#include "stereo/lk_stereo_matcher.h"
#include "module/tracking_module.h"

using namespace std;

namespace omni_slam
{
namespace ros
{

StereoEval::StereoEval(const ::ros::NodeHandle &nh, const ::ros::NodeHandle &nh_private)
    : TrackingEval<true>(nh, nh_private)
{
    int windowSize;
    int numScales;
    double errThresh;
    double epiThresh;

    this->nhp_.param("stereo_matcher_window_size", windowSize, 256);
    this->nhp_.param("stereo_matcher_num_scales", numScales, 5);
    this->nhp_.param("stereo_matcher_error_threshold", errThresh, 20.);
    this->nhp_.param("stereo_matcher_epipolar_threshold", epiThresh, 0.005);

    unique_ptr<stereo::StereoMatcher> stereo(new stereo::LKStereoMatcher(epiThresh, windowSize, numScales, errThresh));

    stereoModule_.reset(new module::StereoModule(stereo));
}

void StereoEval::InitPublishers()
{
    string outputTopic;
    this->nhp_.param("stereo_matched_image_topic", outputTopic, string("/omni_slam/stereo_matched"));
    matchedImagePublisher_ = this->imageTransport_.advertise(outputTopic, 2);
}

void StereoEval::ProcessFrame(unique_ptr<data::Frame> &&frame)
{
    this->trackingModule_->GetLandmarks().clear();
    this->trackingModule_->Update(frame);
    this->trackingModule_->Redetect();
    stereoModule_->Update(*this->trackingModule_->GetFrames().back(), this->trackingModule_->GetLandmarks());
}

void StereoEval::GetResultsData(std::map<std::string, std::vector<std::vector<double>>> &data)
{
    module::StereoModule::Stats &stats = stereoModule_->GetStats();
}

void StereoEval::Visualize(cv_bridge::CvImagePtr &base_img, cv_bridge::CvImagePtr &base_stereo_img)
{
    cv::Mat orig = base_img->image.clone();
    stereoModule_->Visualize(base_img->image, base_stereo_img->image);
    matchedImagePublisher_.publish(base_img->toImageMsg());
    base_img->image = orig;
}

}
}


