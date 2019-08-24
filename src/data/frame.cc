#include "frame.h"

namespace omni_slam
{
namespace data
{

Frame::Frame(const int id, cv::Mat &image, cv::Mat &depth_image, Matrix<double, 3, 4>  &pose, camera::CameraModel &camera_model)
    : id_(id),
    image_(image.clone()),
    depthImage_(depth_image.clone()),
    pose_(pose),
    cameraModel_(camera_model)
{
    hasPose_ = true;
    hasDepth_ = true;
}

Frame::Frame(const int id, cv::Mat &image, cv::Mat &depth_image, camera::CameraModel &camera_model)
    : id_(id),
    image_(image.clone()),
    depthImage_(depth_image.clone()),
    cameraModel_(camera_model)
{
    hasPose_ = false;
    hasDepth_ = true;
}

Frame::Frame(const int id, cv::Mat &image, Matrix<double, 3, 4>  &pose, camera::CameraModel &camera_model)
    : id_(id),
    image_(image.clone()),
    pose_(pose),
    cameraModel_(camera_model)
{
    hasPose_ = true;
    hasDepth_ = false;
}

Frame::Frame(const int id, cv::Mat &image, camera::CameraModel &camera_model)
    : id_(id),
    cameraModel_(camera_model)
{
    hasPose_ = false;
    hasDepth_ = false;
}

const Matrix<double, 3, 4>& Frame::GetPose()
{
    return pose_;
}

const cv::Mat& Frame::GetImage()
{
    return image_;
}

const cv::Mat& Frame::GetDepthImage()
{
    return depthImage_;
}

const int Frame::GetID()
{
    return id_;
}

const camera::CameraModel& Frame::GetCameraModel()
{
    return cameraModel_;
}

void Frame::SetPose(Matrix<double, 3, 4> &pose)
{
    pose_ = pose;
    hasPose_ = true;
}

void Frame::SetDepthImage(cv::Mat &depth_image)
{
    depthImage_ = depth_image.clone();
    hasDepth_ = true;
}

bool Frame::HasPose()
{
    return hasPose_;
}

bool Frame::HasDepthImage()
{
    return hasDepth_;
}

}
}
