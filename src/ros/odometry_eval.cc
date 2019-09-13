#include "odometry_eval.h"

#include "odometry/pnp.h"
#include "module/tracking_module.h"

#include "geometry_msgs/PoseStamped.h"

using namespace std;

namespace omni_slam
{
namespace ros
{

OdometryEval::OdometryEval(const ::ros::NodeHandle &nh, const ::ros::NodeHandle &nh_private)
    : TrackingEval(nh, nh_private)
{
    double reprojThresh;
    int iterations;
    int baMaxIter;
    bool logCeres;

    nhp_.param("pnp_inlier_threshold", reprojThresh, 10.);
    nhp_.param("pnp_iterations", iterations, 1000);
    nhp_.param("bundle_adjustment_max_iterations", baMaxIter, 500);
    nhp_.param("bundle_adjustment_logging", logCeres, false);

    unique_ptr<odometry::PNP> pnp(new odometry::PNP(iterations, reprojThresh));

    odometryModule_.reset(new module::OdometryModule(pnp));
}

void OdometryEval::InitPublishers()
{
    TrackingEval::InitPublishers();
    string outputTopic;
    string outputGndTopic;
    nhp_.param("odometry_estimate_topic", outputTopic, string("/omni_slam/odometry"));
    nhp_.param("odometry_ground_truth_topic", outputGndTopic, string("/omni_slam/odometry_truth"));
    odometryPublisher_ = nh_.advertise<geometry_msgs::PoseStamped>(outputTopic, 2);
    odometryGndPublisher_ = nh_.advertise<geometry_msgs::PoseStamped>(outputGndTopic, 2);
}

void OdometryEval::ProcessFrame(unique_ptr<data::Frame> &&frame)
{
    trackingModule_->Update(frame);
    odometryModule_->Update(trackingModule_->GetLandmarks(), *trackingModule_->GetFrames().back());
    trackingModule_->Redetect();
}

void OdometryEval::Finish()
{
}

void OdometryEval::GetResultsData(std::map<std::string, std::vector<std::vector<double>>> &data)
{
    module::OdometryModule::Stats &stats = odometryModule_->GetStats();
}

void OdometryEval::Visualize(cv_bridge::CvImagePtr &base_img)
{
    TrackingEval::Visualize(base_img);
    geometry_msgs::PoseStamped poseMsg;
    poseMsg.header.stamp = ::ros::Time::now();
    poseMsg.header.frame_id = "map";
    if (trackingModule_->GetFrames().back()->HasEstimatedPose())
    {
        const Matrix<double, 3, 4> &pose = trackingModule_->GetFrames().back()->GetEstimatedPose();
        poseMsg.pose.position.x = pose(0, 3);
        poseMsg.pose.position.y = pose(1, 3);
        poseMsg.pose.position.z = pose(2, 3);
        Quaterniond quat(pose.block<3, 3>(0, 0));
        poseMsg.pose.orientation.x = quat.normalized().x();
        poseMsg.pose.orientation.y = quat.normalized().y();
        poseMsg.pose.orientation.z = quat.normalized().z();
        poseMsg.pose.orientation.w = quat.normalized().w();
        odometryPublisher_.publish(poseMsg);
    }
    const Matrix<double, 3, 4> &poseGnd = trackingModule_->GetFrames().back()->GetPose();
    poseMsg.pose.position.x = poseGnd(0, 3);
    poseMsg.pose.position.y = poseGnd(1, 3);
    poseMsg.pose.position.z = poseGnd(2, 3);
    Quaterniond quatGnd(poseGnd.block<3, 3>(0, 0));
    poseMsg.pose.orientation.x = quatGnd.normalized().x();
    poseMsg.pose.orientation.y = quatGnd.normalized().y();
    poseMsg.pose.orientation.z = quatGnd.normalized().z();
    poseMsg.pose.orientation.w = quatGnd.normalized().w();
    odometryGndPublisher_.publish(poseMsg);
}

}
}
