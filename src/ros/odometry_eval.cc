#include "odometry_eval.h"

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

template <bool Stereo>
OdometryEval<Stereo>::OdometryEval(const ::ros::NodeHandle &nh, const ::ros::NodeHandle &nh_private)
    : TrackingEval<Stereo>(nh, nh_private)
{
    double reprojThresh;
    int iterations;
    int baMaxIter;
    double baLossCoeff;
    bool logCeres;
    int numCeresThreads;

    this->nhp_.param("output_frame", cameraFrame_, std::string("map"));
    this->nhp_.param("pnp_inlier_threshold", reprojThresh, 10.);
    this->nhp_.param("pnp_iterations", iterations, 1000);
    this->nhp_.param("bundle_adjustment_max_iterations", baMaxIter, 500);
    this->nhp_.param("bundle_adjustment_loss_coefficient", baLossCoeff, 0.1);
    this->nhp_.param("bundle_adjustment_logging", logCeres, false);
    this->nhp_.param("bundle_adjustment_num_threads", numCeresThreads, 1);

    unique_ptr<odometry::PNP> pnp(new odometry::PNP(iterations, reprojThresh, numCeresThreads));
    unique_ptr<optimization::BundleAdjuster> bundleAdjuster(new optimization::BundleAdjuster(baMaxIter, baLossCoeff, numCeresThreads, logCeres));

    odometryModule_.reset(new module::OdometryModule(pnp, bundleAdjuster));
}

template <bool Stereo>
void OdometryEval<Stereo>::InitPublishers()
{
    TrackingEval<Stereo>::InitPublishers();
    string outputTopic;
    string outputOptTopic;
    string outputGndTopic;
    string outputPathTopic;
    string outputPathOptTopic;
    string outputPathGndTopic;
    this->nhp_.param("odometry_estimate_topic", outputTopic, string("/omni_slam/odometry"));
    this->nhp_.param("odometry_optimized_topic", outputOptTopic, string("/omni_slam/odometry_optimized"));
    this->nhp_.param("odometry_ground_truth_topic", outputGndTopic, string("/omni_slam/odometry_truth"));
    this->nhp_.param("path_estimate_topic", outputPathTopic, string("/omni_slam/odometry_path"));
    this->nhp_.param("path_optimized_topic", outputOptTopic, string("/omni_slam/odometry_path_optimized"));
    this->nhp_.param("path_ground_truth_topic", outputPathGndTopic, string("/omni_slam/odometry_path_truth"));
    odometryPublisher_ = this->nh_.template advertise<geometry_msgs::PoseStamped>(outputTopic, 2);
    odometryOptPublisher_ = this->nh_.template advertise<geometry_msgs::PoseStamped>(outputOptTopic, 2);
    odometryGndPublisher_ = this->nh_.template advertise<geometry_msgs::PoseStamped>(outputGndTopic, 2);
    pathPublisher_ = this->nh_.template advertise<nav_msgs::Path>(outputPathTopic, 2);
    pathOptPublisher_ = this->nh_.template advertise<nav_msgs::Path>(outputPathOptTopic, 2);
    pathGndPublisher_ = this->nh_.template advertise<nav_msgs::Path>(outputPathGndTopic, 2);
}

template <bool Stereo>
void OdometryEval<Stereo>::ProcessFrame(unique_ptr<data::Frame> &&frame)
{
    this->trackingModule_->Update(frame);
    odometryModule_->Update(this->trackingModule_->GetLandmarks(), *this->trackingModule_->GetFrames().back());
    this->trackingModule_->Redetect();

    this->visualized_ = false;
}

template <bool Stereo>
void OdometryEval<Stereo>::Finish()
{
    ROS_INFO("Performing bundle adjustment...");
    odometryModule_->BundleAdjust(this->trackingModule_->GetLandmarks());
    PublishOdometry(true);
}

template <bool Stereo>
void OdometryEval<Stereo>::GetResultsData(std::map<std::string, std::vector<std::vector<double>>> &data)
{
    module::OdometryModule::Stats &stats = odometryModule_->GetStats();
}

template <bool Stereo>
void OdometryEval<Stereo>::Visualize(cv_bridge::CvImagePtr &base_img)
{
    TrackingEval<Stereo>::Visualize(base_img);
    PublishOdometry();
}

template <bool Stereo>
void OdometryEval<Stereo>::PublishOdometry(bool optimized)
{
    geometry_msgs::PoseStamped poseMsg;
    poseMsg.header.frame_id = cameraFrame_;
    if (this->trackingModule_->GetFrames().back()->HasEstimatedPose() || this->trackingModule_->GetFrames().size() == 1)
    {
        const Matrix<double, 3, 4> &pose = (this->trackingModule_->GetFrames().size() == 1 && !this->trackingModule_->GetFrames().back()->HasEstimatedPose()) ? this->trackingModule_->GetFrames().back()->GetPose() : this->trackingModule_->GetFrames().back()->GetEstimatedPose();
        poseMsg.pose.position.x = pose(0, 3);
        poseMsg.pose.position.y = pose(1, 3);
        poseMsg.pose.position.z = pose(2, 3);
        Quaterniond quat(pose.block<3, 3>(0, 0));
        poseMsg.pose.orientation.x = quat.normalized().x();
        poseMsg.pose.orientation.y = quat.normalized().y();
        poseMsg.pose.orientation.z = quat.normalized().z();
        poseMsg.pose.orientation.w = quat.normalized().w();
        poseMsg.header.stamp = ::ros::Time(this->trackingModule_->GetFrames().back()->GetTime());
        if (optimized)
        {
            odometryOptPublisher_.publish(poseMsg);
        }
        else
        {
            odometryPublisher_.publish(poseMsg);
        }
    }
    const Matrix<double, 3, 4> &poseGnd = this->trackingModule_->GetFrames().back()->GetPose();
    poseMsg.header.frame_id = cameraFrame_;
    poseMsg.pose.position.x = poseGnd(0, 3);
    poseMsg.pose.position.y = poseGnd(1, 3);
    poseMsg.pose.position.z = poseGnd(2, 3);
    Quaterniond quatGnd(poseGnd.block<3, 3>(0, 0));
    poseMsg.pose.orientation.x = quatGnd.normalized().x();
    poseMsg.pose.orientation.y = quatGnd.normalized().y();
    poseMsg.pose.orientation.z = quatGnd.normalized().z();
    poseMsg.pose.orientation.w = quatGnd.normalized().w();
    poseMsg.header.stamp = ::ros::Time(this->trackingModule_->GetFrames().back()->GetTime());
    odometryGndPublisher_.publish(poseMsg);

    nav_msgs::Path path;
    path.header.stamp = ::ros::Time::now();
    path.header.frame_id = cameraFrame_;
    nav_msgs::Path pathGnd;
    pathGnd.header.stamp = ::ros::Time::now();
    pathGnd.header.frame_id = cameraFrame_;
    bool first = true;
    for (const std::unique_ptr<data::Frame> &frame : this->trackingModule_->GetFrames())
    {
        if (frame->HasEstimatedPose() || first)
        {
            geometry_msgs::PoseStamped poseMsg;
            poseMsg.header.stamp = ::ros::Time(frame->GetTime());
            poseMsg.header.frame_id = cameraFrame_;
            const Matrix<double, 3, 4> &pose = (first && !frame->HasEstimatedPose()) ? frame->GetPose() : frame->GetEstimatedPose();
            poseMsg.pose.position.x = pose(0, 3);
            poseMsg.pose.position.y = pose(1, 3);
            poseMsg.pose.position.z = pose(2, 3);
            Quaterniond quat(pose.block<3, 3>(0, 0));
            poseMsg.pose.orientation.x = quat.normalized().x();
            poseMsg.pose.orientation.y = quat.normalized().y();
            poseMsg.pose.orientation.z = quat.normalized().z();
            poseMsg.pose.orientation.w = quat.normalized().w();
            path.poses.push_back(poseMsg);
        }
        if (frame->HasPose())
        {
            geometry_msgs::PoseStamped poseMsg;
            poseMsg.header.stamp = ::ros::Time(frame->GetTime());
            poseMsg.header.frame_id = cameraFrame_;
            const Matrix<double, 3, 4> &pose = frame->GetPose();
            poseMsg.pose.position.x = pose(0, 3);
            poseMsg.pose.position.y = pose(1, 3);
            poseMsg.pose.position.z = pose(2, 3);
            Quaterniond quat(pose.block<3, 3>(0, 0));
            poseMsg.pose.orientation.x = quat.normalized().x();
            poseMsg.pose.orientation.y = quat.normalized().y();
            poseMsg.pose.orientation.z = quat.normalized().z();
            poseMsg.pose.orientation.w = quat.normalized().w();
            pathGnd.poses.push_back(poseMsg);
        }
        first = false;
    }
    if (optimized)
    {
        pathOptPublisher_.publish(path);
    }
    else
    {
        pathPublisher_.publish(path);
    }
    pathGndPublisher_.publish(pathGnd);
}

template class OdometryEval<true>;
template class OdometryEval<false>;

}
}
