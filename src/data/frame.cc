#include "frame.h"
#include "util/tf_util.h"

namespace omni_slam
{
namespace data
{

Frame::Frame(const int id, cv::Mat &image, cv::Mat &depth_image, Matrix<double, 3, 4>  &pose, double time, camera::CameraModel &camera_model)
    : id_(id),
    image_(image.clone()),
    depthImage_(depth_image.clone()),
    pose_(pose),
    invPose_(util::TFUtil::InversePoseMatrix(pose)),
    timeSec_(time),
    cameraModel_(camera_model)
{
    hasPose_ = true;
    hasDepth_ = true;
}

Frame::Frame(const int id, cv::Mat &image, cv::Mat &depth_image, double time, camera::CameraModel &camera_model)
    : id_(id),
    image_(image.clone()),
    depthImage_(depth_image.clone()),
    timeSec_(time),
    cameraModel_(camera_model)
{
    hasPose_ = false;
    hasDepth_ = true;
}

Frame::Frame(const int id, cv::Mat &image, Matrix<double, 3, 4>  &pose, double time, camera::CameraModel &camera_model)
    : id_(id),
    image_(image.clone()),
    pose_(pose),
    invPose_(util::TFUtil::InversePoseMatrix(pose)),
    timeSec_(time),
    cameraModel_(camera_model)
{
    hasPose_ = true;
    hasDepth_ = false;
}

Frame::Frame(const int id, cv::Mat &image, double time, camera::CameraModel &camera_model)
    : id_(id),
    timeSec_(time),
    cameraModel_(camera_model)
{
    hasPose_ = false;
    hasDepth_ = false;
}

const Matrix<double, 3, 4>& Frame::GetPose()
{
    return pose_;
}

const Matrix<double, 3, 4>& Frame::GetInversePose()
{
    return invPose_;
}

cv::Mat& Frame::GetImage()
{
    if (isCompressed_)
    {
        DecompressImages();
    }
    return image_;
}

cv::Mat& Frame::GetDepthImage()
{
    if (isCompressed_)
    {
        DecompressImages();
    }
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
    invPose_ = util::TFUtil::InversePoseMatrix(pose);
    hasPose_ = true;
}

void Frame::SetDepthImage(cv::Mat &depth_image)
{
    if (isCompressed_)
    {
        std::vector<int> param = {cv::IMWRITE_PNG_COMPRESSION, 5};
        cv::imencode(".png", depth_image, depthImageComp_, param);
    }
    else
    {
        depthImageComp_ = depth_image.clone();
    }
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

void Frame::CompressImages()
{
    if (isCompressed_)
    {
        return;
    }
    std::vector<int> param = {cv::IMWRITE_PNG_COMPRESSION, 5};
    cv::imencode(".png", image_, imageComp_, param);
    if (hasDepth_)
    {
        cv::imencode(".png", depthImage_, depthImageComp_, param);
    }
    image_.release();
    depthImage_.release();
    isCompressed_ = true;
}

void Frame::DecompressImages()
{
    if (!isCompressed_)
    {
        return;
    }
    image_ = cv::imdecode(cv::Mat(1, imageComp_.size(), CV_8UC1, imageComp_.data()), CV_LOAD_IMAGE_UNCHANGED);
    if (hasDepth_)
    {
        depthImage_ = cv::imdecode(cv::Mat(1, depthImageComp_.size(), CV_8UC1, depthImageComp_.data()), CV_LOAD_IMAGE_UNCHANGED);
    }
    imageComp_.clear();
    depthImageComp_.clear();
    isCompressed_ = false;
}

bool Frame::IsCompressed()
{
    return isCompressed_;
}

}
}
