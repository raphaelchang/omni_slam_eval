#include "tracking_eval.h"

#include "feature/lk_tracker.h"
#include "feature/descriptor_tracker.h"
#include "feature/detector.h"
#include "odometry/five_point.h"

using namespace std;

namespace omni_slam
{
namespace ros
{

template <bool Stereo>
TrackingEval<Stereo>::TrackingEval(const ::ros::NodeHandle &nh, const ::ros::NodeHandle &nh_private)
    : EvalBase<Stereo>(nh, nh_private)
{
    string detectorType;
    string descriptorType;
    int trackerWindowSize;
    int trackerNumScales;
    int keyframeInterval;
    double fivePointThreshold;
    int fivePointRansacIterations;
    double trackerDeltaPixelErrorThresh;
    double trackerErrorThresh;
    map<string, double> detectorParams;
    map<string, double> descriptorParams;
    int minFeaturesRegion;
    int maxFeaturesRegion;
    string trackerType;

    this->nhp_.param("detector_type", detectorType, string("GFTT"));
    this->nhp_.param("descriptor_type", descriptorType, string("ORB"));
    this->nhp_.param("tracker_window_size", trackerWindowSize, 128);
    this->nhp_.param("tracker_num_scales", trackerNumScales, 4);
    this->nhp_.param("tracker_checker_epipolar_threshold", fivePointThreshold, 0.01745240643);
    this->nhp_.param("tracker_checker_iterations", fivePointRansacIterations, 1000);
    this->nhp_.param("tracker_delta_pixel_error_threshold", trackerDeltaPixelErrorThresh, 5.0);
    this->nhp_.param("tracker_error_threshold", trackerErrorThresh, 20.);
    this->nhp_.param("min_features_per_region", minFeaturesRegion, 5);
    this->nhp_.param("max_features_per_region", maxFeaturesRegion, 5000);
    this->nhp_.getParam("detector_parameters", detectorParams);
    this->nhp_.getParam("descriptor_parameters", descriptorParams);
    this->nhp_.param("keyframe_interval", keyframeInterval, 1);
    this->nhp_.param("tracker_type", trackerType, string("lk"));

    unique_ptr<feature::Detector> detector;
    if (feature::Detector::IsDetectorTypeValid(detectorType))
    {
        if (trackerType == "lk")
        {
            detector.reset(new feature::Detector(detectorType, detectorParams));
        }
        else if (trackerType == "descriptor")
        {
            detector.reset(new feature::Detector(detectorType, descriptorType, detectorParams, descriptorParams));
        }
    }
    else
    {
        ROS_ERROR("Invalid feature detector specified");
    }

    unique_ptr<feature::Tracker> tracker;
    if (trackerType == "lk")
    {
        tracker.reset(new feature::LKTracker(trackerWindowSize, trackerNumScales, trackerDeltaPixelErrorThresh, trackerErrorThresh, keyframeInterval));
    }
    else if (trackerType == "descriptor")
    {
        tracker.reset(new feature::DescriptorTracker(detectorType, descriptorType, detectorParams, descriptorParams, trackerErrorThresh, keyframeInterval));
    }
    else
    {
        ROS_ERROR("Invalid tracker type specified");
    }

    unique_ptr<odometry::FivePoint> checker(new odometry::FivePoint(fivePointRansacIterations, fivePointThreshold, 0, false, 0));

    trackingModule_.reset(new module::TrackingModule(detector, tracker, checker, minFeaturesRegion, maxFeaturesRegion));
}

template <bool Stereo>
void TrackingEval<Stereo>::InitPublishers()
{
    if (!pubInitialized_)
    {
        EvalBase<Stereo>::InitPublishers();

        string outputTopic;
        this->nhp_.param("tracked_image_topic", outputTopic, string("/omni_slam/tracked"));
        trackedImagePublisher_ = this->imageTransport_.advertise(outputTopic, 2);

        pubInitialized_ = true;
    }
}

template <bool Stereo>
void TrackingEval<Stereo>::ProcessFrame(unique_ptr<data::Frame> &&frame)
{
    trackingModule_->Update(frame);
    trackingModule_->Redetect();

    visualized_ = false;
}

template <bool Stereo>
void TrackingEval<Stereo>::GetResultsData(std::map<std::string, std::vector<std::vector<double>>> &data)
{
    module::TrackingModule::Stats &stats = trackingModule_->GetStats();
    data["failures"] = stats.failureRadDists;
    data["successes"] = stats.successRadDists;
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

template <bool Stereo>
void TrackingEval<Stereo>::Visualize(cv_bridge::CvImagePtr &base_img)
{
    if (!visualized_)
    {
        cv::Mat orig = base_img->image.clone();
        trackingModule_->Visualize(base_img->image);
        trackedImagePublisher_.publish(base_img->toImageMsg());
        base_img->image = orig;
        visualized_ = true;
    }
}

template class TrackingEval<true>;
template class TrackingEval<false>;

}
}
