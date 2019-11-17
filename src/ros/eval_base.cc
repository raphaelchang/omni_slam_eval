#include "eval_base.h"

#include <rosbag/bag.h>
#include <rosbag/view.h>

#include "camera/double_sphere.h"
#include "camera/perspective.h"
#include "util/tf_util.h"
#include "util/hdf_file.h"

#define USE_GROUND_TRUTH

using namespace Eigen;

namespace omni_slam
{
namespace ros
{

template <>
EvalBase<false>::EvalBase(const ::ros::NodeHandle &nh, const ::ros::NodeHandle &nh_private)
    : nh_(nh), nhp_(nh_private), imageTransport_(nh)
{
    std::string cameraModel;
    nhp_.param("camera_model", cameraModel, std::string("double_sphere"));
    nhp_.getParam("camera_parameters", cameraParams_);
    nhp_.param("image_topic", imageTopic_, std::string("/camera/image_raw"));
    nhp_.param("depth_image_topic", depthImageTopic_, std::string("/depth_camera/image_raw"));
    nhp_.param("pose_topic", poseTopic_, std::string("/pose"));
    nhp_.param("vignette", vignette_, 0.0);
    nhp_.param("vignette_expansion", vignetteExpansion_, 0.01);

    if (cameraModel == "double_sphere")
    {
        cameraModel_.reset(new camera::DoubleSphere<>(cameraParams_["fx"], cameraParams_["fy"], cameraParams_["cx"], cameraParams_["cy"], cameraParams_["chi"], cameraParams_["alpha"], vignette_));
    }
    else if (cameraModel == "perspective")
    {
        cameraModel_.reset(new camera::Perspective<>(cameraParams_["fx"], cameraParams_["fy"], cameraParams_["cx"], cameraParams_["cy"]));
    }
    else
    {
        ROS_ERROR("Invalid camera model specified");
    }
}

template <>
EvalBase<true>::EvalBase(const ::ros::NodeHandle &nh, const ::ros::NodeHandle &nh_private)
    : nh_(nh), nhp_(nh_private), imageTransport_(nh)
{
    std::string cameraModel;
    std::map<std::string, double> stereoCameraParams;
    nhp_.param("camera_model", cameraModel, std::string("double_sphere"));
    nhp_.getParam("camera_parameters", cameraParams_);
    nhp_.getParam("stereo_camera_parameters", stereoCameraParams);
    nhp_.param("image_topic", imageTopic_, std::string("/camera/image_raw"));
    nhp_.param("depth_image_topic", depthImageTopic_, std::string("/depth_camera/image_raw"));
    nhp_.param("pose_topic", poseTopic_, std::string("/pose"));
    nhp_.param("stereo_image_topic", stereoImageTopic_, std::string("/camera2/image_raw"));
    std::vector<double> stereoT;
    stereoT.reserve(3);
    std::vector<double> stereoR;
    stereoR.reserve(4);
    nhp_.getParam("stereo_camera_parameters/tf_t", stereoT);
    nhp_.getParam("stereo_camera_parameters/tf_r", stereoR);
    Quaterniond q(stereoR[3], stereoR[0], stereoR[1], stereoR[2]);
    Vector3d t(stereoT[0], stereoT[1], stereoT[2]);
    stereoPose_ = util::TFUtil::QuaternionTranslationToPoseMatrix(q, t);
    nhp_.param("vignette", vignette_, 0.0);
    nhp_.param("vignette_expansion", vignetteExpansion_, 0.01);

    if (cameraModel == "double_sphere")
    {
        cameraModel_.reset(new camera::DoubleSphere<>(cameraParams_["fx"], cameraParams_["fy"], cameraParams_["cx"], cameraParams_["cy"], cameraParams_["chi"], cameraParams_["alpha"], vignette_));
        stereoCameraModel_.reset(new camera::DoubleSphere<>(cameraParams_["fx"], cameraParams_["fy"], cameraParams_["cx"], cameraParams_["cy"], cameraParams_["chi"], cameraParams_["alpha"], vignette_));
    }
    else if (cameraModel == "perspective")
    {
        cameraModel_.reset(new camera::Perspective<>(cameraParams_["fx"], cameraParams_["fy"], cameraParams_["cx"], cameraParams_["cy"]));
        stereoCameraModel_.reset(new camera::Perspective<>(stereoCameraParams["fx"], stereoCameraParams["fy"], stereoCameraParams["cx"], stereoCameraParams["cy"]));
    }
    else
    {
        ROS_ERROR("Invalid camera model specified");
    }
}

template <>
void EvalBase<false>::InitSubscribers()
{
    imageSubscriber_.subscribe(imageTransport_, imageTopic_, 3);
    depthImageSubscriber_.subscribe(imageTransport_, depthImageTopic_, 3);
    poseSubscriber_.subscribe(nh_, poseTopic_, 10);
    sync_.reset(new message_filters::Synchronizer<MessageFilter>(MessageFilter(5), imageSubscriber_, depthImageSubscriber_, poseSubscriber_));
    sync_->registerCallback(boost::bind(&EvalBase<false>::FrameCallback, this, _1, _2, _3));
}

template <>
void EvalBase<true>::InitSubscribers()
{
    imageSubscriber_.subscribe(imageTransport_, imageTopic_, 3);
    stereoImageSubscriber_.subscribe(imageTransport_, stereoImageTopic_, 3);
    depthImageSubscriber_.subscribe(imageTransport_, depthImageTopic_, 3);
    poseSubscriber_.subscribe(nh_, poseTopic_, 10);
    sync_.reset(new message_filters::Synchronizer<MessageFilter>(MessageFilter(5), imageSubscriber_, stereoImageSubscriber_, depthImageSubscriber_, poseSubscriber_));
    sync_->registerCallback(boost::bind(&EvalBase<true>::FrameCallback, this, _1, _2, _3, _4));
}

template <bool Stereo>
void EvalBase<Stereo>::InitPublishers()
{
}

template <bool Stereo>
void EvalBase<Stereo>::FrameCallback(const sensor_msgs::ImageConstPtr &image, const sensor_msgs::ImageConstPtr &depth_image, const geometry_msgs::PoseStamped::ConstPtr &pose)
{
    cv_bridge::CvImagePtr cvImage;
    cv_bridge::CvImagePtr cvDepthImage;
    try
    {
        cvImage = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
        if (depth_image != nullptr)
        {
            cvDepthImage = cv_bridge::toCvCopy(depth_image, sensor_msgs::image_encodings::TYPE_16UC1);
        }
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    Quaterniond q(pose->pose.orientation.w, pose->pose.orientation.x, pose->pose.orientation.y, pose->pose.orientation.z);
    Vector3d t(pose->pose.position.x, pose->pose.position.y, pose->pose.position.z);
    Matrix<double, 3, 4> posemat = util::TFUtil::QuaternionTranslationToPoseMatrix(q, t);
    if (vignette_ > 0)
    {
        cv::Mat mask = cv::Mat::zeros(cvImage->image.size(), CV_32FC3);
        cv::circle(mask, cv::Point2f(cvImage->image.cols / 2 - 0.5, cvImage->image.rows / 2 - 0.5), std::max(cvImage->image.rows, cvImage->image.cols) * (vignette_ + vignetteExpansion_) / 2, cv::Scalar::all(1), -1);
        cv::GaussianBlur(mask, mask, cv::Size(51, 51), 0);
        cv::multiply(cvImage->image, mask, cvImage->image, 1, CV_8UC3);
    }
    cv::Mat monoImg;
    cv::cvtColor(cvImage->image, monoImg, CV_BGR2GRAY);
    cv::Mat depthFloatImg;
    if (depth_image != nullptr)
    {
        cvDepthImage->image.convertTo(depthFloatImg, CV_64FC1, 500. / 65535);
    }

#ifdef USE_GROUND_TRUTH
    if (depth_image != nullptr)
    {
        ProcessFrame(std::unique_ptr<data::Frame>(new data::Frame(monoImg, depthFloatImg, posemat, pose->header.stamp.toSec(), *cameraModel_)));
    }
    else
    {
        ProcessFrame(std::unique_ptr<data::Frame>(new data::Frame(monoImg, posemat, pose->header.stamp.toSec(), *cameraModel_)));
    }
#else
    if (first_)
    {
        ProcessFrame(std::unique_ptr<data::Frame>(new data::Frame(monoImg, posemat, pose->header.stamp.toSec(), *cameraModel_)));
        first_ = false;
    }
    else
    {
        ProcessFrame(std::unique_ptr<data::Frame>(new data::Frame(monoImg, pose->header.stamp.toSec(), *cameraModel_)));
    }
#endif
    Visualize(cvImage);
}

template <bool Stereo>
void EvalBase<Stereo>::FrameCallback(const sensor_msgs::ImageConstPtr &image, const sensor_msgs::ImageConstPtr &stereo_image, const sensor_msgs::ImageConstPtr &depth_image, const geometry_msgs::PoseStamped::ConstPtr &pose)
{
    cv_bridge::CvImagePtr cvImage;
    cv_bridge::CvImagePtr cvStereoImage;
    cv_bridge::CvImagePtr cvDepthImage;
    try
    {
        cvImage = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
        cvStereoImage = cv_bridge::toCvCopy(stereo_image, sensor_msgs::image_encodings::BGR8);
        if (depth_image != nullptr)
        {
            cvDepthImage = cv_bridge::toCvCopy(depth_image, sensor_msgs::image_encodings::TYPE_16UC1);
        }
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    Quaterniond q(pose->pose.orientation.w, pose->pose.orientation.x, pose->pose.orientation.y, pose->pose.orientation.z);
    Vector3d t(pose->pose.position.x, pose->pose.position.y, pose->pose.position.z);
    Matrix<double, 3, 4> posemat = util::TFUtil::QuaternionTranslationToPoseMatrix(q, t);
    if (vignette_ > 0)
    {
        cv::Mat mask = cv::Mat::zeros(cvImage->image.size(), CV_32FC3);
        cv::circle(mask, cv::Point2f(cvImage->image.cols / 2 - 0.5, cvImage->image.rows / 2 - 0.5), std::max(cvImage->image.rows, cvImage->image.cols) * (vignette_ + vignetteExpansion_) / 2, cv::Scalar::all(1), -1);
        cv::GaussianBlur(mask, mask, cv::Size(51, 51), 0);
        cv::multiply(cvImage->image, mask, cvImage->image, 1, CV_8UC3);
        cv::multiply(cvStereoImage->image, mask, cvStereoImage->image, 1, CV_8UC3);
    }
    cv::Mat monoImg;
    cv::cvtColor(cvImage->image, monoImg, CV_BGR2GRAY);
    cv::Mat monoImg2;
    cv::cvtColor(cvStereoImage->image, monoImg2, CV_BGR2GRAY);
    cv::equalizeHist(monoImg, monoImg);
    cv::equalizeHist(monoImg2, monoImg2);
    cv::Mat depthFloatImg;
    if (depth_image != nullptr)
    {
        cvDepthImage->image.convertTo(depthFloatImg, CV_64FC1, 500. / 65535);
    }

#ifdef USE_GROUND_TRUTH
    if (depth_image != nullptr)
    {
        ProcessFrame(std::unique_ptr<data::Frame>(new data::Frame(monoImg, monoImg2, depthFloatImg, posemat, stereoPose_, pose->header.stamp.toSec(), *cameraModel_, *stereoCameraModel_)));
    }
    else
    {
        ProcessFrame(std::unique_ptr<data::Frame>(new data::Frame(monoImg, monoImg2, posemat, stereoPose_, pose->header.stamp.toSec(), *cameraModel_, *stereoCameraModel_)));
    }
#else
    if (first_)
    {
        ProcessFrame(std::unique_ptr<data::Frame>(new data::Frame(monoImg, monoImg2, posemat, stereoPose_, pose->header.stamp.toSec(), *cameraModel_, *stereoCameraModel_)));
        first_ = false;
    }
    else
    {
        ProcessFrame(std::unique_ptr<data::Frame>(new data::Frame(monoImg, monoImg2, stereoPose_, pose->header.stamp.toSec(), *cameraModel_, *stereoCameraModel_)));
    }
#endif

    Visualize(cvImage, cvStereoImage);
}

template <bool Stereo>
void EvalBase<Stereo>::Finish()
{
}

template <bool Stereo>
bool EvalBase<Stereo>::GetAttributes(std::map<std::string, std::string> &attributes)
{
    return false;
}

template <bool Stereo>
bool EvalBase<Stereo>::GetAttributes(std::map<std::string, double> &attributes)
{
    return false;
}

template <bool Stereo>
void EvalBase<Stereo>::Visualize(cv_bridge::CvImagePtr &base_img)
{
}

template <bool Stereo>
void EvalBase<Stereo>::Visualize(cv_bridge::CvImagePtr &base_img, cv_bridge::CvImagePtr &base_stereo_img)
{
    Visualize(base_img);
}

template <bool Stereo>
void EvalBase<Stereo>::Run()
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
        sensor_msgs::ImageConstPtr stereoMsg = nullptr;
        sensor_msgs::ImageConstPtr depthMsg = nullptr;
        geometry_msgs::PoseStamped::ConstPtr poseMsg = nullptr;
        nav_msgs::Odometry::ConstPtr odomMsg = nullptr;
        geometry_msgs::TransformStamped::ConstPtr tfMsg = nullptr;
        int runNext = 0;
        int numMsgs = depthImageTopic_ == "" ? 1 : 2;
        int skip = 0;
        int finished = true;
        for (rosbag::MessageInstance const m : rosbag::View(bag))
        {
            if (!::ros::ok())
            {
                finished = false;
                break;
            }
            if (depthImageTopic_ != "" && m.getTopic() == depthImageTopic_)
            {
                depthMsg = m.instantiate<sensor_msgs::Image>();
                runNext++;
            }
            else if (Stereo && m.getTopic() == stereoImageTopic_)
            {
                stereoMsg = m.instantiate<sensor_msgs::Image>();
                runNext++;
            }
            else if (m.getTopic() == imageTopic_)
            {
                imageMsg = m.instantiate<sensor_msgs::Image>();
                runNext++;
            }
            else if (runNext >= (Stereo ? numMsgs + 1 : numMsgs) && m.getTopic() == poseTopic_)
            {
                poseMsg = m.instantiate<geometry_msgs::PoseStamped>();
                geometry_msgs::PoseStamped::Ptr pose(new geometry_msgs::PoseStamped());
                if (poseMsg == nullptr)
                {
                    odomMsg = m.instantiate<nav_msgs::Odometry>();
                    if (odomMsg == nullptr)
                    {
                        tfMsg = m.instantiate<geometry_msgs::TransformStamped>();
                        if (tfMsg != nullptr)
                        {
                            pose->header = tfMsg->header;
                            pose->pose.position.x = -tfMsg->transform.translation.y;
                            pose->pose.position.y = tfMsg->transform.translation.x;
                            pose->pose.position.z = tfMsg->transform.translation.z;
                            pose->pose.orientation = tfMsg->transform.rotation;
                        }
                    }
                    else
                    {
                        pose->pose = odomMsg->pose.pose;
                        pose->header = odomMsg->header;
                    }
                }
                else
                {
                    *pose = *poseMsg;
                }
                runNext = 0;
                if (skip == 0)
                {
                    if (imageMsg != nullptr && (depthImageTopic_ == "" || depthMsg != nullptr) && (poseMsg != nullptr || odomMsg != nullptr || tfMsg != nullptr) && (!Stereo || stereoMsg != nullptr))
                    {
                        if (Stereo)
                        {
                            FrameCallback(imageMsg, stereoMsg, depthMsg, pose);
                        }
                        else
                        {
                            FrameCallback(imageMsg, depthMsg, pose);
                        }
                    }
                }
                skip++;
                if (skip >= rate)
                {
                    skip = 0;
                }
            }
        }
        if (finished)
        {
            Finish();
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
        out.AddAttribute("fov", cameraModel_->GetFOV() * 180 / M_PI);
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

template class EvalBase<true>;
template class EvalBase<false>;

}
}
