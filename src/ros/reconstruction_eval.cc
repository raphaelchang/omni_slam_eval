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

template <bool Stereo>
ReconstructionEval<Stereo>::ReconstructionEval(const ::ros::NodeHandle &nh, const ::ros::NodeHandle &nh_private)
    : TrackingEval<Stereo>(nh, nh_private)
{
    double maxReprojError;
    double minTriAngle;
    int baMaxIter;
    double baLossCoeff;
    bool logCeres;
    int numCeresThreads;

    this->nhp_.param("output_frame", cameraFrame_, std::string("map"));
    this->nhp_.param("max_reprojection_error", maxReprojError, 0.0);
    this->nhp_.param("min_triangulation_angle", minTriAngle, 1.0);
    this->nhp_.param("bundle_adjustment_max_iterations", baMaxIter, 500);
    this->nhp_.param("bundle_adjustment_loss_coefficient", baLossCoeff, 0.1);
    this->nhp_.param("bundle_adjustment_logging", logCeres, false);
    this->nhp_.param("bundle_adjustment_num_threads", numCeresThreads, 1);

    unique_ptr<reconstruction::Triangulator> triangulator(new reconstruction::Triangulator(maxReprojError, minTriAngle));
    unique_ptr<optimization::BundleAdjuster> bundleAdjuster(new optimization::BundleAdjuster(baMaxIter, baLossCoeff, numCeresThreads, logCeres));

    reconstructionModule_.reset(new module::ReconstructionModule(triangulator, bundleAdjuster));
}

template <bool Stereo>
void ReconstructionEval<Stereo>::InitPublishers()
{
    TrackingEval<Stereo>::InitPublishers();
    string outputTopic;
    this->nhp_.param("point_cloud_topic", outputTopic, string("/omni_slam/reconstructed"));
    pointCloudPublisher_ = this->nh_.template advertise<sensor_msgs::PointCloud2>(outputTopic, 2);
}

template <bool Stereo>
void ReconstructionEval<Stereo>::ProcessFrame(unique_ptr<data::Frame> &&frame)
{
    this->trackingModule_->Update(frame);
    reconstructionModule_->Update(this->trackingModule_->GetLandmarks());
    this->trackingModule_->Redetect();

    this->visualized_ = false;
}

template <bool Stereo>
void ReconstructionEval<Stereo>::Finish()
{
    ROS_INFO("Performing bundle adjustment...");
    reconstructionModule_->BundleAdjust(this->trackingModule_->GetLandmarks());

    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    cv::Mat noarr;
    reconstructionModule_->Visualize(noarr, cloud);
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);
    msg.header.frame_id = cameraFrame_;
    msg.header.stamp = ::ros::Time::now();
    pointCloudPublisher_.publish(msg);
}

template <bool Stereo>
void ReconstructionEval<Stereo>::GetResultsData(std::map<std::string, std::vector<std::vector<double>>> &data)
{
    module::ReconstructionModule::Stats &stats = reconstructionModule_->GetStats();
    for (const data::Landmark &landmark : this->trackingModule_->GetLandmarks())
    {
        if (landmark.HasGroundTruth() && landmark.HasEstimatedPosition() && landmark.GetStereoObservations().size() == 0)
        {
            Vector3d gnd = landmark.GetGroundTruth();
            Vector3d est = landmark.GetEstimatedPosition();
            data["landmarks"].emplace_back(std::vector<double>{est(0), est(1), est(2), gnd(0), gnd(1), gnd(2), (double)landmark.GetNumFramesForEstimate()});
        }
    }
}

template <bool Stereo>
void ReconstructionEval<Stereo>::Visualize(cv_bridge::CvImagePtr &base_img)
{
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    reconstructionModule_->Visualize(base_img->image, cloud);
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);
    msg.header.frame_id = cameraFrame_;
    msg.header.stamp = ::ros::Time::now();
    pointCloudPublisher_.publish(msg);
    TrackingEval<Stereo>::Visualize(base_img);
}

template class ReconstructionEval<true>;
template class ReconstructionEval<false>;

}
}
