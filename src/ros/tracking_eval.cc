#include "tracking_eval.h"

#include "feature/tracker.h"
#include "feature/detector.h"

using namespace std;

namespace omni_slam
{
namespace ros
{

TrackingEval::TrackingEval(const ::ros::NodeHandle &nh, const ::ros::NodeHandle &nh_private)
    : EvalBase(nh, nh_private)
{
    string detectorType;
    int trackerWindowSize;
    int trackerNumScales;
    double trackerDeltaPixelErrorThresh;
    double trackerErrorThresh;
    map<string, double> detectorParams;
    int minFeaturesRegion;

    nhp_.param("detector_type", detectorType, string("GFTT"));
    nhp_.param("tracker_window_size", trackerWindowSize, 128);
    nhp_.param("tracker_num_scales", trackerNumScales, 4);
    nhp_.param("tracker_delta_pixel_error_threshold", trackerDeltaPixelErrorThresh, 5.0);
    nhp_.param("tracker_error_threshold", trackerErrorThresh, 20.);
    nhp_.param("min_features_per_region", minFeaturesRegion, 5);
    nhp_.getParam("detector_parameters", detectorParams);

    unique_ptr<feature::Detector> detector;
    if (feature::Detector::IsDetectorTypeValid(detectorType))
    {
        detector.reset(new feature::Detector(detectorType, detectorParams));
    }
    else
    {
        ROS_ERROR("Invalid feature detector specified");
    }

    unique_ptr<feature::Tracker> tracker(new feature::Tracker(trackerWindowSize, trackerNumScales, trackerDeltaPixelErrorThresh, trackerErrorThresh));

    trackingModule_.reset(new module::TrackingModule(detector, tracker, minFeaturesRegion));
}

void TrackingEval::InitPublishers()
{
    EvalBase::InitPublishers();

    string outputTopic;
    nhp_.param("tracked_image_topic", outputTopic, string("/omni_slam/tracked"));
    trackedImagePublisher_ = imageTransport_.advertise(outputTopic, 2);
}

void TrackingEval::ProcessFrame(unique_ptr<data::Frame> &&frame)
{
    trackingModule_->Update(frame);
    trackingModule_->Redetect();
}

void TrackingEval::GetResultsData(std::map<std::string, std::vector<std::vector<double>>> &data)
{
    module::TrackingModule::Stats &stats = trackingModule_->GetStats();
    data["failures"] = {stats.failureRadDists};
    data["radial_errors"] = stats.radialErrors;
    data["length_errors"] = stats.frameErrors;
    data["track_counts"] = vector<vector<double>>();
    data["track_counts"].reserve(stats.frameTrackCounts.size());
    for (auto &&v : stats.frameTrackCounts)
    {
        data["track_counts"].emplace_back(begin(v), end(v));
    }
    data["track_lengths"] = {vector<double>(stats.trackLengths.begin(), stats.trackLengths.end())};
}

void TrackingEval::Visualize(cv_bridge::CvImagePtr &base_img)
{
    trackingModule_->Visualize(base_img->image);
    trackedImagePublisher_.publish(base_img->toImageMsg());
}

}
}
