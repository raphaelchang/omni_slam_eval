#ifndef _OMNI_SLAM_EVAL_TRACKING_NODE_H_
#define _OMNI_SLAM_EVAL_TRACKING_NODE_H_

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Image.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/synchronizer.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <opencv2/opencv.hpp>
#include <vector>

#include "feature/tracker.h"
#include "feature/detector.h"
#include "data/frame.h"
#include "data/landmark.h"
#include "camera/camera_model.h"

namespace omni_slam
{

class TrackingNode
{
public:
    TrackingNode(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private, bool realtime);
    TrackingNode(bool realtime) : TrackingNode(ros::NodeHandle(), ros::NodeHandle("~"), realtime) {}
    void FrameCallback(const sensor_msgs::ImageConstPtr &image, const sensor_msgs::ImageConstPtr &depth_image, const geometry_msgs::PoseStamped::ConstPtr &pose);
    void GetResultsData(std::map<std::string, std::vector<std::vector<double>>> &data);

private:
    ros::NodeHandle nh_;
    ros::NodeHandle nhp_;

    image_transport::ImageTransport imageTransport_;
    image_transport::SubscriberFilter imageSubscriber_;
    image_transport::SubscriberFilter depthImageSubscriber_;
    message_filters::Subscriber<geometry_msgs::PoseStamped> poseSubscriber_;
    std::unique_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, geometry_msgs::PoseStamped>>> sync_;

    image_transport::Publisher trackedImagePublisher_;

    std::unique_ptr<camera::CameraModel> cameraModel_;
    std::unique_ptr<feature::Detector> detector_;
    std::unique_ptr<feature::Tracker> tracker_;

    std::list<data::Frame> frames_;
    std::vector<data::Landmark> landmarks_;

    cv::Mat visMask_;
    std::vector<cv::Scalar> colors_;

    int frameNum_{0};

    const std::vector<double> rs_{0, 0.1, 0.2, 0.3, 0.4, 0.5};
    const std::vector<double> ts_{-M_PI, -3 * M_PI / 4, -M_PI / 2, -M_PI / 4, 0, M_PI / 4, M_PI / 2, 3 * M_PI / 4, M_PI};
    int minFeaturesRegion_;

    std::vector<std::vector<double>> radErrors_;
    std::vector<std::vector<double>> frameErrors_;
    std::vector<std::vector<double>> frameTrackCounts_;
    std::vector<int> trackLengths_;
    std::vector<double> failureRadDists_;
};

}

#endif /* _OMNI_SLAM_EVAL_TRACKING_NODE_H_ */
