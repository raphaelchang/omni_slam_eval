#include "odometry_module.h"

using namespace std;

namespace omni_slam
{
namespace module
{

OdometryModule::OdometryModule(std::unique_ptr<odometry::PoseEstimator> &pose_estimator, std::unique_ptr<optimization::BundleAdjuster> &bundle_adjuster)
    : poseEstimator_(std::move(pose_estimator)),
    bundleAdjuster_(std::move(bundle_adjuster))
{
}

OdometryModule::OdometryModule(std::unique_ptr<odometry::PoseEstimator> &&pose_estimator, std::unique_ptr<optimization::BundleAdjuster> &&bundle_adjuster)
    : OdometryModule(pose_estimator, bundle_adjuster)
{
}

void OdometryModule::Update(std::vector<data::Landmark> &landmarks, std::vector<std::unique_ptr<data::Frame>> &frames)
{
    if (landmarks.size() == 0)
    {
        return;
    }
    poseEstimator_->Compute(landmarks, *frames.back(), **next(frames.rbegin()));
}

void OdometryModule::BundleAdjust(std::vector<data::Landmark> &landmarks)
{
    bundleAdjuster_->Optimize(landmarks);
}

OdometryModule::Stats& OdometryModule::GetStats()
{
    return stats_;
}

}
}
