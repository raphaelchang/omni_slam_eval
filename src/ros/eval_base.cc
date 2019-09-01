#include "eval_base.h"

#include <rosbag/bag.h>
#include <rosbag/view.h>

#include "camera/double_sphere.h"
#include "camera/perspective.h"
#include "util/tf_util.h"
#include "util/hdf_file.h"

using namespace Eigen;

namespace omni_slam
{
namespace ros
{

EvalBase::EvalBase(const ::ros::NodeHandle &nh, const ::ros::NodeHandle &nh_private)
    : nh_(nh), nhp_(nh_private), imageTransport_(nh)
{
    std::string cameraModel;
    nhp_.param("camera_model", cameraModel, std::string("double_sphere"));
    nhp_.getParam("camera_parameters", cameraParams_);
    nhp_.param("image_topic", imageTopic_, std::string("/camera/image_raw"));
    nhp_.param("depth_image_topic", depthImageTopic_, std::string("/depth_camera/image_raw"));
    nhp_.param("pose_topic", poseTopic_, std::string("/pose"));

    if (cameraModel == "double_sphere")
    {
        cameraModel_.reset(new camera::DoubleSphere(cameraParams_["fx"], cameraParams_["fy"], cameraParams_["cx"], cameraParams_["cy"], cameraParams_["chi"], cameraParams_["alpha"]));
    }
    else if (cameraModel == "perspective")
    {
        cameraModel_.reset(new camera::Perspective(cameraParams_["fx"], cameraParams_["fy"], cameraParams_["cx"], cameraParams_["cy"]));
    }
    else
    {
        ROS_ERROR("Invalid camera model specified");
    }
}

void EvalBase::InitSubscribers()
{
    imageSubscriber_.subscribe(imageTransport_, imageTopic_, 3);
    depthImageSubscriber_.subscribe(imageTransport_, depthImageTopic_, 3);
    poseSubscriber_.subscribe(nh_, poseTopic_, 10);
    sync_.reset(new message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, geometry_msgs::PoseStamped>>(message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, geometry_msgs::PoseStamped>(5), imageSubscriber_, depthImageSubscriber_, poseSubscriber_));
    sync_->registerCallback(boost::bind(&EvalBase::FrameCallback, this, _1, _2, _3));
}

void EvalBase::InitPublishers()
{
}

void EvalBase::FrameCallback(const sensor_msgs::ImageConstPtr &image, const sensor_msgs::ImageConstPtr &depth_image, const geometry_msgs::PoseStamped::ConstPtr &pose)
{
    cv_bridge::CvImagePtr cvImage;
    cv_bridge::CvImagePtr cvDepthImage;
    try
    {
        cvImage = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
        cvDepthImage = cv_bridge::toCvCopy(depth_image, sensor_msgs::image_encodings::TYPE_16UC1);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    Quaterniond q(pose->pose.orientation.w, pose->pose.orientation.x, pose->pose.orientation.y, pose->pose.orientation.z);
    Vector3d t(pose->pose.position.x, pose->pose.position.y, pose->pose.position.z);
    Matrix<double, 3, 4> posemat = util::TFUtil::QuaternionTranslationToPoseMatrix(q, t);
    cv::Mat monoImg;
    cv::cvtColor(cvImage->image, monoImg, CV_BGR2GRAY);
    cv::Mat depthFloatImg;
    cvDepthImage->image.convertTo(depthFloatImg, CV_64FC1, 500. / 65535);

    ProcessFrame(std::unique_ptr<data::Frame>(new data::Frame(frameNum_, monoImg, depthFloatImg, posemat, pose->header.stamp.toSec(), *cameraModel_)));

    Visualize(cvImage);

    frameNum_++;
}

bool EvalBase::GetAttributes(std::map<std::string, std::string> &attributes)
{
    return false;
}

bool EvalBase::GetAttributes(std::map<std::string, double> &attributes)
{
    return false;
}

void EvalBase::Visualize(cv_bridge::CvImagePtr &base_img)
{
}

void EvalBase::Run()
{
    std::string bagFile;
    std::string resultsFile;
    nhp_.param("bag_file", bagFile, std::string(""));
    nhp_.param("results_file", resultsFile, std::string(""));
    if (bagFile == "")
    {
        InitSubscribers();
        InitPublishers();
        ::ros::spin();
        ROS_INFO("Saving results...");
        std::map<std::string, std::vector<std::vector<double>>> results;
        GetResultsData(results);
        omni_slam::util::HDFFile out(resultsFile);
        for (auto &dataset : results)
        {
            out.AddDataset(dataset.first, dataset.second);
        }
    }
    else
    {
        InitPublishers();
        int rate;
        nhp_.param("rate", rate, 1);

        ROS_INFO("Processing bag...");
        rosbag::Bag bag;
        bag.open(bagFile);
        sensor_msgs::ImageConstPtr imageMsg = nullptr;
        sensor_msgs::ImageConstPtr depthMsg = nullptr;
        geometry_msgs::PoseStamped::ConstPtr poseMsg = nullptr;
        int runNext = 0;
        int skip = 0;
        for (rosbag::MessageInstance const m : rosbag::View(bag))
        {
            if (!::ros::ok())
            {
                break;
            }
            if (m.getTopic() == depthImageTopic_)
            {
                depthMsg = m.instantiate<sensor_msgs::Image>();
                runNext++;
            }
            else if (m.getTopic() == imageTopic_)
            {
                imageMsg = m.instantiate<sensor_msgs::Image>();
                runNext++;
            }
            else if (runNext >= 2 && m.getTopic() == poseTopic_)
            {
                poseMsg = m.instantiate<geometry_msgs::PoseStamped>();
                runNext = 0;
                if (skip == 0)
                {
                    if (imageMsg != nullptr && depthMsg != nullptr && poseMsg != nullptr)
                    {
                        FrameCallback(imageMsg, depthMsg, poseMsg);
                    }
                }
                skip++;
                if (skip >= rate)
                {
                    skip = 0;
                }
            }
        }

        ROS_INFO("Saving results...");
        std::map<std::string, std::vector<std::vector<double>>> results;
        GetResultsData(results);
        omni_slam::util::HDFFile out(resultsFile);
        for (auto &dataset : results)
        {
            out.AddDataset(dataset.first, dataset.second);
        }
        out.AddAttribute("bag_file", bagFile);
        out.AddAttribute("rate", rate);
        out.AddAttributes(cameraParams_);
        std::map<std::string, double> numAttr;
        if (GetAttributes(numAttr))
        {
            out.AddAttributes(numAttr);
        }
        std::map<std::string, std::string> strAttr;
        if (GetAttributes(strAttr))
        {
            out.AddAttributes(strAttr);
        }
    }
}

}
}
