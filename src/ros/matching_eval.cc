#include "matching_eval.h"

#include "feature/matcher.h"
#include "feature/detector.h"
#include "odometry/five_point.h"

using namespace std;

namespace omni_slam
{
namespace ros
{

MatchingEval::MatchingEval(const ::ros::NodeHandle &nh, const ::ros::NodeHandle &nh_private)
    : EvalBase<>(nh, nh_private)
{
    map<string, double> detectorParams;
    map<string, double> descriptorParams;
    double matcherMaxDist;
    double overlapThresh;
    double distThresh;
    bool localUnwarp;
    double fivePointThreshold;
    int fivePointRansacIterations;

    nhp_.param("detector_type", detectorType_, string("SIFT"));
    nhp_.getParam("detector_parameters", detectorParams);
    nhp_.param("descriptor_type", descriptorType_, string("SIFT"));
    nhp_.getParam("descriptor_parameters", descriptorParams);
    nhp_.param("matcher_max_dist", matcherMaxDist, 0.);
    nhp_.param("feature_overlap_threshold", overlapThresh, 0.5);
    nhp_.param("feature_distance_threshold", distThresh, 10.);
    nhp_.param("local_unwarp", localUnwarp, false);
    nhp_.param("estimator_epipolar_threshold", fivePointThreshold, 0.01745240643);
    nhp_.param("estimator_iterations", fivePointRansacIterations, 1000);

    unique_ptr<feature::Detector> detector;
    if (feature::Detector::IsDetectorTypeValid(detectorType_))
    {
        if (!feature::Detector::IsDetectorDescriptorCombinationValid(detectorType_, descriptorType_))
        {
            ROS_WARN("Invalid feature detector descriptor combination specified");
        }
        detector.reset(new feature::Detector(detectorType_, descriptorType_, detectorParams, descriptorParams, localUnwarp));
    }
    else
    {
        ROS_ERROR("Invalid feature detector specified");
    }

    unique_ptr<feature::Matcher> matcher(new feature::Matcher(descriptorType_, matcherMaxDist));
    unique_ptr<odometry::FivePoint> estimator(new odometry::FivePoint(fivePointRansacIterations, fivePointThreshold, 0, false, 0));

    matchingModule_.reset(new module::MatchingModule(detector, matcher, estimator, overlapThresh, distThresh));
}

void MatchingEval::InitPublishers()
{
    EvalBase<>::InitPublishers();

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
    data["good_radial_distances"] = stats.goodRadialDistances;
    data["bad_radial_distances"] = stats.badRadialDistances;
    data["roc_curves"] = stats.rocCurves;
    data["precision_recall_curves"] = stats.precRecCurves;
    data["rotation_errors"] = stats.rotationErrors;
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
