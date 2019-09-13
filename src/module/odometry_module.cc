#include "odometry_module.h"

using namespace std;

namespace omni_slam
{
namespace module
{

OdometryModule::OdometryModule(std::unique_ptr<odometry::PNP> &pnp, std::unique_ptr<optimization::BundleAdjuster> &bundle_adjuster)
    : pnp_(std::move(pnp)),
    bundleAdjuster_(std::move(bundle_adjuster))
{
}

OdometryModule::OdometryModule(std::unique_ptr<odometry::PNP> &&pnp, std::unique_ptr<optimization::BundleAdjuster> &&bundle_adjuster)
    : OdometryModule(pnp, bundle_adjuster)
{
}

void OdometryModule::Update(std::vector<data::Landmark> &landmarks, data::Frame &frame)
{
    pnp_->Compute(landmarks, frame);
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
