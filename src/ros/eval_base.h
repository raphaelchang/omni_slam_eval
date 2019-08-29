#ifndef _EVAL_BASE_H_
#define _EVAL_BASE_H_

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

#include "data/frame.h"
#include "camera/camera_model.h"

namespace omni_slam
{
namespace ros
{

class EvalBase
{
public:
    EvalBase(const ::ros::NodeHandle &nh, const ::ros::NodeHandle &nh_private);
    EvalBase() : EvalBase(::ros::NodeHandle(), ::ros::NodeHandle("~")) {}

    void Run();

protected:
    virtual void InitSubscribers();
    virtual void InitPublishers();

    ::ros::NodeHandle nh_;
    ::ros::NodeHandle nhp_;

    image_transport::ImageTransport imageTransport_;

    std::unique_ptr<camera::CameraModel> cameraModel_;

    int frameNum_{0};

private:
    void FrameCallback(const sensor_msgs::ImageConstPtr &image, const sensor_msgs::ImageConstPtr &depth_image, const geometry_msgs::PoseStamped::ConstPtr &pose);

    virtual void ProcessFrame(std::unique_ptr<data::Frame> &&frame) = 0;
    virtual void GetResultsData(std::map<std::string, std::vector<std::vector<double>>> &data) = 0;
    virtual void Visualize(cv_bridge::CvImagePtr &base_img);

    image_transport::SubscriberFilter imageSubscriber_;
    image_transport::SubscriberFilter depthImageSubscriber_;
    message_filters::Subscriber<geometry_msgs::PoseStamped> poseSubscriber_;
    std::unique_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, geometry_msgs::PoseStamped>>> sync_;

    std::string imageTopic_;
    std::string depthImageTopic_;
    std::string poseTopic_;
    std::map<std::string, double> cameraParams_;
};

}
}
#endif /* _EVAL_BASE_H_ */
