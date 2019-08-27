#include "omni_slam_eval_tracking_node.h"

#include <rosbag/bag.h>
#include <rosbag/view.h>

#include "camera/double_sphere.h"
#include "camera/perspective.h"
#include "util/tf_util.h"

using namespace std;
using namespace Eigen;

namespace omni_slam
{

TrackingNode::TrackingNode(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private, bool realtime)
    : nh_(nh), nhp_(nh_private), imageTransport_(nh)
{
    string imageTopic;
    string depthImageTopic;
    string poseTopic;
    string outputTopic;
    string cameraModel;
    string detectorType;
    int trackerWindowSize;
    int trackerNumScales;
    double trackerDeltaPixelErrorThresh;
    double trackerErrorThresh;
    map<string, double> cameraParams;
    map<string, double> detectorParams;

    nhp_.param("image_topic", imageTopic, string("/camera/image_raw"));
    nhp_.param("depth_image_topic", depthImageTopic, string("/depth_camera/image_raw"));
    nhp_.param("pose_topic", poseTopic, string("/pose"));
    nhp_.param("tracked_image_topic", outputTopic, string("/omni_slam/tracked"));
    nhp_.param("camera_model", cameraModel, string("double_sphere"));
    nhp_.param("detector_type", detectorType, string("GFTT"));
    nhp_.param("tracker_window_size", trackerWindowSize, 128);
    nhp_.param("tracker_num_scales", trackerNumScales, 4);
    nhp_.param("tracker_delta_pixel_error_threshold", trackerDeltaPixelErrorThresh, 5.0);
    nhp_.param("tracker_error_threshold", trackerErrorThresh, 20.);
    nhp_.param("min_features_per_region", minFeaturesRegion_, 5);
    nhp_.getParam("camera_parameters", cameraParams);
    nhp_.getParam("detector_parameters", detectorParams);

    if (cameraModel == "double_sphere")
    {
        cameraModel_.reset(new camera::DoubleSphere(cameraParams["fx"], cameraParams["fy"], cameraParams["cx"], cameraParams["cy"], cameraParams["chi"], cameraParams["alpha"]));
    }
    else if (cameraModel == "perspective")
    {
        cameraModel_.reset(new camera::Perspective(cameraParams["fx"], cameraParams["fy"], cameraParams["cx"], cameraParams["cy"]));
    }
    else
    {
        ROS_ERROR("Invalid camera model specified");
    }

    if (feature::Detector::IsDetectorTypeValid(detectorType))
    {
        detector_.reset(new feature::Detector(detectorType, detectorParams));
    }
    else
    {
        ROS_ERROR("Invalid feature detector specified");
    }

    tracker_.reset(new feature::Tracker(trackerWindowSize, trackerNumScales, trackerDeltaPixelErrorThresh, trackerErrorThresh));

    trackedImagePublisher_ = imageTransport_.advertise(outputTopic, 2);

    if (realtime)
    {
        imageSubscriber_.subscribe(imageTransport_, imageTopic, 3);
        depthImageSubscriber_.subscribe(imageTransport_, depthImageTopic, 3);
        poseSubscriber_.subscribe(nh_, poseTopic, 10);
        sync_.reset(new message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, geometry_msgs::PoseStamped>>(message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, geometry_msgs::PoseStamped>(5), imageSubscriber_, depthImageSubscriber_, poseSubscriber_));
        sync_->registerCallback(boost::bind(&TrackingNode::FrameCallback, this, _1, _2, _3));
    }
}

void TrackingNode::FrameCallback(const sensor_msgs::ImageConstPtr &image, const sensor_msgs::ImageConstPtr &depth_image, const geometry_msgs::PoseStamped::ConstPtr &pose)
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
    frames_.emplace_back(frameNum_, monoImg, depthFloatImg, posemat, pose->header.stamp.toSec(), *cameraModel_);

    if (frameNum_ == 0)
    {
        int imsize = std::max(frames_.back().GetImage().rows, frames_.back().GetImage().cols);
        for (int i = 0; i < rs_.size() - 1; i++)
        {
            for (int j = 0; j < ts_.size() - 1; j++)
            {
                detector_->DetectInRadialRegion(frames_.back(), landmarks_, rs_[i] * imsize, rs_[i+1] * imsize, ts_[j], ts_[j+1]);
            }
        }
        tracker_->Init(frames_.back());
        visMask_ = cv::Mat::zeros(cvImage->image.size(), CV_8UC3);
        cv::RNG rng(123);
        for (int i = 0; i < landmarks_.size(); i++)
        {
            colors_.emplace_back(rng.uniform(0, 200), rng.uniform(0, 200), rng.uniform(0, 200));
        }
        frameNum_++;
        return;
    }

    tracker_->Track(landmarks_, frames_.back());

    int i = 0;
    map<pair<int, int>, int> regionCount;
    int imsize = std::max(frames_.back().GetImage().rows, frames_.back().GetImage().cols);
    for (data::Landmark& landmark : landmarks_)
    {
        data::Feature *obs;
        if ((obs = landmark.GetObservationByFrameID(frames_.back().GetID())) != nullptr)
        {
            Vector2d pixel_gnd;
            if (frames_.back().GetCameraModel().ProjectToImage(util::TFUtil::WorldFrameToCameraFrame(util::TFUtil::TransformPoint(frames_.back().GetInversePose(), landmark.GetGroundTruth())), pixel_gnd))
            {
                data::Feature *obsPrev = landmark.GetObservationByFrameID(next(frames_.rbegin())->GetID());
                Vector2d pixel;
                pixel << obs->GetKeypoint().pt.x, obs->GetKeypoint().pt.y;
                double error = (pixel - pixel_gnd).norm();
                cv::Scalar color(0, (int)(255 * (1 - error / 10)), (int)(255 * (error / 10)));
                cv::line(visMask_, obsPrev->GetKeypoint().pt, obs->GetKeypoint().pt, color, 1);
                cv::circle(cvImage->image, obs->GetKeypoint().pt, 1, color, -1);
                cv::circle(cvImage->image, cv::Point2f(pixel_gnd(0), pixel_gnd(1)), 3, colors_[i], -1);
            }
            double x = obs->GetKeypoint().pt.x - frames_.back().GetImage().cols / 2. + 0.5;
            double y = obs->GetKeypoint().pt.y - frames_.back().GetImage().rows / 2. + 0.5;
            double r = sqrt(x * x + y * y) / imsize;
            double t = atan2(y, x);
            vector<double>::const_iterator ri = upper_bound(rs_.begin(), rs_.end(), r);
            vector<double>::const_iterator ti = upper_bound(ts_.begin(), ts_.end(), t);
            int rinx = min((int)(ri - rs_.begin()), (int)(rs_.size() - 1)) - 1;
            int tinx = min((int)(ti - ts_.begin()), (int)(ts_.size() - 1)) - 1;
            if (regionCount.find({rinx, tinx}) == regionCount.end())
            {
                regionCount[{rinx, tinx}] = 0;
            }
            regionCount[{rinx, tinx}]++;
        }
        else if ((obs = landmark.GetObservationByFrameID(next(frames_.rbegin())->GetID())) != nullptr) // Failed in current frame
        {

        }
        i++;
    }
    for (int i = 0; i < rs_.size() - 1; i++)
    {
        for (int j = 0; j < ts_.size() - 1; j++)
        {
            if (regionCount.find({i, j}) == regionCount.end() || regionCount[{i, j}] < minFeaturesRegion_)
            {
                detector_->DetectInRadialRegion(frames_.back(), landmarks_, rs_[i] * imsize, rs_[i+1] * imsize, ts_[j], ts_[j+1]);
            }
        }
    }

    cvImage->image += visMask_;
    trackedImagePublisher_.publish(cvImage->toImageMsg());

    next(frames_.rbegin())->CompressImages();
    frameNum_++;
}

}

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "omni_slam_eval_tracking_node");
    ros::NodeHandle nh("~");
    string bagFile;
    nh.param("bag_file", bagFile, string(""));
    if (bagFile == "")
    {
        omni_slam::TrackingNode node(true);
        ros::spin();
    }
    else
    {
        string imageTopic;
        string depthImageTopic;
        string poseTopic;
        int rate;
        nh.param("image_topic", imageTopic, string("/camera/image_raw"));
        nh.param("depth_image_topic", depthImageTopic, string("/depth_camera/image_raw"));
        nh.param("pose_topic", poseTopic, string("/pose"));
        nh.param("rate", rate, 1);
        omni_slam::TrackingNode node(false);

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
            if (!ros::ok())
            {
                break;
            }
            if (m.getTopic() == depthImageTopic)
            {
                depthMsg = m.instantiate<sensor_msgs::Image>();
                runNext++;
            }
            else if (m.getTopic() == imageTopic)
            {
                imageMsg = m.instantiate<sensor_msgs::Image>();
                runNext++;
            }
            else if (runNext >= 2 && m.getTopic() == poseTopic)
            {
                poseMsg = m.instantiate<geometry_msgs::PoseStamped>();
                runNext = 0;
                if (skip == 0)
                {
                    if (imageMsg != nullptr && depthMsg != nullptr && poseMsg != nullptr)
                    {
                        node.FrameCallback(imageMsg, depthMsg, poseMsg);
                    }
                }
                skip++;
                if (skip >= rate)
                {
                    skip = 0;
                }
            }
        }
    }
    return 0;
}
