#include "odometry_module.h"

using namespace std;

namespace omni_slam
{
namespace module
{

OdometryModule::OdometryModule(std::unique_ptr<odometry::PNP> &pnp)
    : pnp_(std::move(pnp))
{
}

OdometryModule::OdometryModule(std::unique_ptr<odometry::PNP> &&pnp)
    : OdometryModule(pnp)
{
}

void OdometryModule::Update(std::vector<data::Landmark> &landmarks, data::Frame &frame)
{
    pnp_->Compute(landmarks, frame);
}

OdometryModule::Stats& OdometryModule::GetStats()
{
    return stats_;
}

}
}
