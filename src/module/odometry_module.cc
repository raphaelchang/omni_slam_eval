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

void OdometryModule::Update(std::vector<data::Landmark> &landmarks, std::unique_ptr<data::Frame> &cur_frame, const data::Frame *keyframe)
{
    if (landmarks.size() == 0)
    {
        frameNum_++;
        return;
    }
    vector<int> inliers;
    poseEstimator_->Compute(landmarks, *cur_frame, *keyframe, inliers);

    unordered_set<int> inlierSet(inliers.begin(), inliers.end());
    int imsize = max(cur_frame->GetImage().rows, cur_frame->GetImage().cols);
    for (int i = 0; i < landmarks.size(); i++)
    {
        if (inlierSet.find(i) != inlierSet.end())
        {
            const data::Feature *feat = landmarks[i].GetObservationByFrameID(cur_frame->GetID());
            double x = feat->GetKeypoint().pt.x - cur_frame->GetImage().cols / 2. + 0.5;
            double y = feat->GetKeypoint().pt.y - cur_frame->GetImage().rows / 2. + 0.5;
            double r = sqrt(x * x + y * y) / imsize;
            stats_.inlierRadDists.push_back(std::vector<double>{(double)frameNum_, r});
        }
        else if (landmarks[i].IsObservedInFrame(cur_frame->GetID()))
        {
            const data::Feature *feat = landmarks[i].GetObservationByFrameID(cur_frame->GetID());
            double x = feat->GetKeypoint().pt.x - cur_frame->GetImage().cols / 2. + 0.5;
            double y = feat->GetKeypoint().pt.y - cur_frame->GetImage().rows / 2. + 0.5;
            double r = sqrt(x * x + y * y) / imsize;
            stats_.outlierRadDists.push_back(std::vector<double>{(double)frameNum_, r});
        }
    }
    frameNum_++;
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
