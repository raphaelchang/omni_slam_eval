#include "matching_eval.h"

#include "feature/matcher.h"
#include "feature/detector.h"

using namespace std;

namespace omni_slam
{
namespace ros
{

MatchingEval::MatchingEval(const ::ros::NodeHandle &nh, const ::ros::NodeHandle &nh_private)
    : EvalBase(nh, nh_private)
{
    map<string, double> detectorParams;
    map<string, double> descriptorParams;
    double matcherMaxDist;
    double overlapThresh;
    double distThresh;

    nhp_.param("detector_type", detectorType_, string("SIFT"));
    nhp_.getParam("detector_parameters", detectorParams);
    nhp_.param("descriptor_type", descriptorType_, string("SIFT"));
    nhp_.getParam("descriptor_parameters", descriptorParams);
    nhp_.param("matcher_max_dist", matcherMaxDist, 0.);
    nhp_.param("feature_overlap_threshold", overlapThresh, 0.5);
    nhp_.param("feature_distance_threshold", distThresh, 10.);

    unique_ptr<feature::Detector> detector;
    if (feature::Detector::IsDetectorTypeValid(detectorType_))
    {
        if (!feature::Detector::IsDetectorDescriptorCombinationValid(detectorType_, descriptorType_))
        {
            ROS_WARN("Invalid feature detector descriptor combination specified");
        }
        detector.reset(new feature::Detector(detectorType_, descriptorType_, detectorParams, descriptorParams));
    }
    else
    {
        ROS_ERROR("Invalid feature detector specified");
    }

    unique_ptr<feature::Matcher> matcher(new feature::Matcher(descriptorType_, matcherMaxDist));

    matchingModule_.reset(new module::MatchingModule(detector, matcher, overlapThresh, distThresh));
}

void MatchingEval::InitPublishers()
{
    EvalBase::InitPublishers();

    string outputTopic;
    nhp_.param("matched_image_topic", outputTopic, string("/omni_slam/matched"));
    matchedImagePublisher_ = imageTransport_.advertise(outputTopic, 2);
}

void MatchingEval::ProcessFrame(unique_ptr<data::Frame> &&frame)
{
    matchingModule_->Update(frame);
}

void MatchingEval::GetResultsData(std::map<std::string, std::vector<std::vector<double>>> &data)
{
    module::MatchingModule::Stats &stats = matchingModule_->GetStats();
    data["match_stats"] = stats.frameMatchStats;
    data["radial_overlaps_errors"] = stats.radialOverlapsErrors;
    data["delta_radius"] = {stats.deltaRadius};
}

bool MatchingEval::GetAttributes(std::map<std::string, std::string> &attributes)
{
    attributes["detector_type"] = detectorType_;
    attributes["descriptor_type"] = descriptorType_;
    return true;
}

void MatchingEval::Visualize(cv_bridge::CvImagePtr &base_img)
{
    matchingModule_->Visualize(base_img->image);
    matchedImagePublisher_.publish(base_img->toImageMsg());
}

}
}
