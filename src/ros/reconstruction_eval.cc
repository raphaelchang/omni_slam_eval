#include "reconstruction_eval.h"

#include "reconstruction/triangulator.h"
#include "optimization/bundle_adjuster.h"
#include "module/tracking_module.h"

#include <pcl_conversions/pcl_conversions.h>
#include "sensor_msgs/PointCloud2.h"

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
    int baMaxIter;
    bool logCeres;

    nhp_.param("max_reprojection_error", maxReprojError, 0.0);
    nhp_.param("min_triangulation_angle", minTriAngle, 1.0);
    nhp_.param("bundle_adjustment_max_iterations", baMaxIter, 500);
    nhp_.param("bundle_adjustment_logging", logCeres, false);

    unique_ptr<reconstruction::Triangulator> triangulator(new reconstruction::Triangulator(maxReprojError, minTriAngle));
    unique_ptr<optimization::BundleAdjuster> bundleAdjuster(new optimization::BundleAdjuster(baMaxIter, logCeres));

    reconstructionModule_.reset(new module::ReconstructionModule(triangulator, bundleAdjuster));
}

void ReconstructionEval::InitPublishers()
{
    TrackingEval::InitPublishers();
    string outputTopic;
    nhp_.param("point_cloud_topic", outputTopic, string("/omni_slam/reconstructed"));
    pointCloudPublisher_ = nh_.advertise<sensor_msgs::PointCloud2>(outputTopic, 2);
}

void ReconstructionEval::ProcessFrame(unique_ptr<data::Frame> &&frame)
{
    TrackingEval::ProcessFrame(std::move(frame));
    reconstructionModule_->Update(trackingModule_->GetLandmarks());
}

void ReconstructionEval::Finish()
{
    reconstructionModule_->BundleAdjust(trackingModule_->GetLandmarks());

    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    cv::Mat noarr;
    reconstructionModule_->Visualize(noarr, cloud);
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);
    msg.header.frame_id = "map";
    msg.header.stamp = ::ros::Time::now();
    pointCloudPublisher_.publish(msg);
}

void ReconstructionEval::GetResultsData(std::map<std::string, std::vector<std::vector<double>>> &data)
{
    module::ReconstructionModule::Stats &stats = reconstructionModule_->GetStats();
}

void ReconstructionEval::Visualize(cv_bridge::CvImagePtr &base_img)
{
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    reconstructionModule_->Visualize(base_img->image, cloud);
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);
    msg.header.frame_id = "map";
    msg.header.stamp = ::ros::Time::now();
    pointCloudPublisher_.publish(msg);
    TrackingEval::Visualize(base_img);
}

}
}
