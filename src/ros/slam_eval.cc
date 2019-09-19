#include "slam_eval.h"

#include "odometry/pnp.h"
#include "optimization/bundle_adjuster.h"
#include "module/tracking_module.h"

#include "geometry_msgs/PoseStamped.h"
#include "nav_msgs/Path.h"

using namespace std;

namespace omni_slam
{
namespace ros
{

SLAMEval::SLAMEval(const ::ros::NodeHandle &nh, const ::ros::NodeHandle &nh_private)
    : OdometryEval<true>(nh, nh_private), ReconstructionEval<true>(nh, nh_private), StereoEval(nh, nh_private)
{
    this->nhp_.param("local_bundle_adjustment_window", baSlidingWindow_, 0);
}

void SLAMEval::InitPublishers()
{
    OdometryEval<true>::InitPublishers();
    ReconstructionEval<true>::InitPublishers();
    StereoEval::InitPublishers();
}

void SLAMEval::ProcessFrame(unique_ptr<data::Frame> &&frame)
{
    trackingModule_->Update(frame);
    odometryModule_->Update(trackingModule_->GetLandmarks(), *trackingModule_->GetFrames().back());
    reconstructionModule_->Update(trackingModule_->GetLandmarks());
    if (baSlidingWindow_ > 0 && (frameNum_ + 1) % baSlidingWindow_ == 0)
    {
        std::vector<int> frameIds;
        frameIds.reserve(baSlidingWindow_);
        for (auto it = trackingModule_->GetFrames().rbegin(); it != trackingModule_->GetFrames().rend(); ++it)
        {
            frameIds.push_back((*it)->GetID());
            if (frameIds.size() >= baSlidingWindow_)
            {
                break;
            }
        }
        reconstructionModule_->BundleAdjust(trackingModule_->GetLandmarks(), frameIds);
    }
    trackingModule_->Redetect();
    stereoModule_->Update(*trackingModule_->GetFrames().back(), trackingModule_->GetLandmarks());
    frameNum_++;
}

void SLAMEval::Finish()
{
    ReconstructionEval<true>::Finish();
    PublishOdometry();
}

void SLAMEval::GetResultsData(std::map<std::string, std::vector<std::vector<double>>> &data)
{
}

void SLAMEval::Visualize(cv_bridge::CvImagePtr &base_img)
{
    visualized_ = false;
    ReconstructionEval<true>::Visualize(base_img);
    OdometryEval<true>::Visualize(base_img);
}

void SLAMEval::Visualize(cv_bridge::CvImagePtr &base_img, cv_bridge::CvImagePtr &base_stereo_img)
{
    Visualize(base_img);
    StereoEval::Visualize(base_img, base_stereo_img);
}

}
}

