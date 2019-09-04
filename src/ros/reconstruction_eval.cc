#include "reconstruction_eval.h"

#include "reconstruction/triangulator.h"
#include "module/tracking_module.h"

using namespace std;

namespace omni_slam
{
namespace ros
{

ReconstructionEval::ReconstructionEval(const ::ros::NodeHandle &nh, const ::ros::NodeHandle &nh_private)
    : TrackingEval(nh, nh_private)
{
    double maxReprojError;
    double minTriAngle;

    nhp_.param("max_reprojection_error", maxReprojError, 0.0);
    nhp_.param("min_triangulation_angle", minTriAngle, 1.0);

    unique_ptr<reconstruction::Triangulator> triangulator(new reconstruction::Triangulator(maxReprojError, minTriAngle));

    reconstructionModule_.reset(new module::ReconstructionModule(triangulator));
}

void ReconstructionEval::InitPublishers()
{
    TrackingEval::InitPublishers();
}

void ReconstructionEval::ProcessFrame(unique_ptr<data::Frame> &&frame)
{
    TrackingEval::ProcessFrame(std::move(frame));
    reconstructionModule_->Update(trackingModule_->GetLandmarks());
}

void ReconstructionEval::GetResultsData(std::map<std::string, std::vector<std::vector<double>>> &data)
{
    module::ReconstructionModule::Stats &stats = reconstructionModule_->GetStats();
}

void ReconstructionEval::Visualize(cv_bridge::CvImagePtr &base_img)
{
    TrackingEval::Visualize(base_img);
}

}
}
