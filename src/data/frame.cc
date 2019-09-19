#include "frame.h"
#include "util/tf_util.h"

namespace omni_slam
{
namespace data
{

int Frame::lastFrameId_ = 0;

Frame::Frame(cv::Mat &image, cv::Mat &stereo_image, cv::Mat &depth_image, Matrix<double, 3, 4>  &pose, Matrix<double, 3, 4> &stereo_pose, double time, camera::CameraModel<> &camera_model)
    : id_(lastFrameId_++),
    image_(image.clone()),
    stereoImage_(stereo_image.clone()),
    depthImage_(depth_image.clone()),
    pose_(pose),
    invPose_(util::TFUtil::InversePoseMatrix(pose)),
    stereoPose_(stereo_pose),
    timeSec_(time),
    cameraModel_(camera_model)
{
    hasPose_ = true;
    hasDepth_ = true;
    hasStereo_ = true;
}

Frame::Frame(cv::Mat &image, cv::Mat &stereo_image, Matrix<double, 3, 4> &stereo_pose, double time, camera::CameraModel<> &camera_model)
    : id_(lastFrameId_++),
    image_(image.clone()),
    stereoImage_(stereo_image.clone()),
    stereoPose_(stereo_pose),
    timeSec_(time),
    cameraModel_(camera_model)
{
    hasPose_ = false;
    hasDepth_ = false;
    hasStereo_ = true;
}

Frame::Frame(cv::Mat &image, cv::Mat &depth_image, double time, camera::CameraModel<> &camera_model)
    : id_(lastFrameId_++),
    image_(image.clone()),
    depthImage_(depth_image.clone()),
    timeSec_(time),
    cameraModel_(camera_model)
{
    hasPose_ = false;
    hasDepth_ = true;
    hasStereo_ = false;
}

Frame::Frame(cv::Mat &image, cv::Mat &stereo_image, cv::Mat &depth_image, Matrix<double, 3, 4> &stereo_pose, double time, camera::CameraModel<> &camera_model)
    : id_(lastFrameId_++),
    image_(image.clone()),
    stereoImage_(stereo_image.clone()),
    depthImage_(depth_image.clone()),
    stereoPose_(stereo_pose),
    timeSec_(time),
    cameraModel_(camera_model)
{
    hasPose_ = false;
    hasDepth_ = true;
    hasStereo_ = true;
}

Frame::Frame(cv::Mat &image, Matrix<double, 3, 4>  &pose, double time, camera::CameraModel<> &camera_model)
    : id_(lastFrameId_++),
    image_(image.clone()),
    pose_(pose),
    invPose_(util::TFUtil::InversePoseMatrix(pose)),
    timeSec_(time),
    cameraModel_(camera_model)
{
    hasPose_ = true;
    hasDepth_ = false;
    hasStereo_ = false;
}

Frame::Frame(cv::Mat &image, cv::Mat &stereo_image, Matrix<double, 3, 4>  &pose, Matrix<double, 3, 4> &stereo_pose, double time, camera::CameraModel<> &camera_model)
    : id_(lastFrameId_++),
    image_(image.clone()),
    stereoImage_(stereo_image.clone()),
    pose_(pose),
    invPose_(util::TFUtil::InversePoseMatrix(pose)),
    stereoPose_(stereo_pose),
    timeSec_(time),
    cameraModel_(camera_model)
{
    hasPose_ = true;
    hasDepth_ = false;
    hasStereo_ = true;
}

Frame::Frame(cv::Mat &image, Matrix<double, 3, 4>  &pose, cv::Mat &depth_image, double time, camera::CameraModel<> &camera_model)
    : id_(lastFrameId_++),
    image_(image.clone()),
    depthImage_(depth_image.clone()),
    pose_(pose),
    invPose_(util::TFUtil::InversePoseMatrix(pose)),
    timeSec_(time),
    cameraModel_(camera_model)
{
    hasPose_ = true;
    hasDepth_ = true;
    hasStereo_ = false;
}

Frame::Frame(cv::Mat &image, double time, camera::CameraModel<> &camera_model)
    : id_(lastFrameId_++),
    timeSec_(time),
    cameraModel_(camera_model)
{
    hasPose_ = false;
    hasDepth_ = false;
    hasStereo_ = false;
}

Frame::Frame(const Frame &other)
    : image_(other.image_.clone()),
    depthImage_(other.depthImage_.clone()),
    stereoImage_(other.stereoImage_.clone()),
    imageComp_(other.imageComp_),
    depthImageComp_(other.depthImageComp_),
    stereoImageComp_(other.stereoImageComp_),
    cameraModel_(other.cameraModel_),
    id_(other.id_),
    pose_(other.pose_),
    invPose_(other.invPose_),
    timeSec_(other.timeSec_),
    hasPose_(other.hasPose_),
    hasDepth_(other.hasDepth_),
    hasStereo_(other.hasStereo_),
    isCompressed_(other.isCompressed_)
{
}

const Matrix<double, 3, 4>& Frame::GetPose() const
{
    return pose_;
}

const Matrix<double, 3, 4>& Frame::GetInversePose() const
{
    return invPose_;
}

const cv::Mat& Frame::GetImage()
{
    if (isCompressed_)
    {
        DecompressImages();
    }
    return image_;
}

const cv::Mat& Frame::GetDepthImage()
{
    if (isCompressed_)
    {
        DecompressImages();
    }
    return depthImage_;
}

const cv::Mat& Frame::GetStereoImage()
{
    if (isCompressed_)
    {
        DecompressImages();
    }
    return stereoImage_;
}

const int Frame::GetID() const
{
    return id_;
}

const Matrix<double, 3, 4>& Frame::GetStereoPose() const
{
    return stereoPose_;
}

const camera::CameraModel<>& Frame::GetCameraModel() const
{
    return cameraModel_;
}

const Matrix<double, 3, 4>& Frame::GetEstimatedPose() const
{
    return poseEstimate_;
}

const Matrix<double, 3, 4>& Frame::GetEstimatedInversePose() const
{
    return invPoseEstimate_;
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

void Frame::SetStereoImage(cv::Mat &stereo_image)
{
    if (isCompressed_)
    {
        std::vector<int> param = {cv::IMWRITE_PNG_COMPRESSION, 5};
        cv::imencode(".png", stereo_image, stereoImageComp_, param);
    }
    else
    {
        stereoImageComp_ = stereo_image.clone();
    }
    hasStereo_ = true;
}

void Frame::SetStereoPose(Matrix<double, 3, 4> &pose)
{
    stereoPose_ = pose;
}

void Frame::SetEstimatedPose(const Matrix<double, 3, 4> &pose, const std::vector<int> &landmark_ids)
{
    poseEstimate_ = pose;
    invPoseEstimate_ = util::TFUtil::InversePoseMatrix(pose);
    estLandmarkIds_ = std::unordered_set<int>(landmark_ids.begin(), landmark_ids.end());
    hasPoseEstimate_ = true;
}

void Frame::SetEstimatedPose(const Matrix<double, 3, 4> &pose)
{
    poseEstimate_ = pose;
    invPoseEstimate_ = util::TFUtil::InversePoseMatrix(pose);
    hasPoseEstimate_ = true;
}

void Frame::SetEstimatedInversePose(const Matrix<double, 3, 4> &pose, const std::vector<int> &landmark_ids)
{
    invPoseEstimate_ = pose;
    poseEstimate_ = util::TFUtil::InversePoseMatrix(pose);
    estLandmarkIds_ = std::unordered_set<int>(landmark_ids.begin(), landmark_ids.end());
    hasPoseEstimate_ = true;
}

void Frame::SetEstimatedInversePose(const Matrix<double, 3, 4> &pose)
{
    invPoseEstimate_ = pose;
    poseEstimate_ = util::TFUtil::InversePoseMatrix(pose);
    hasPoseEstimate_ = true;
}

bool Frame::HasPose() const
{
    return hasPose_;
}

bool Frame::HasDepthImage() const
{
    return hasDepth_;
}

bool Frame::HasStereoImage() const
{
    return hasStereo_;
}

bool Frame::HasEstimatedPose() const
{
    return hasPoseEstimate_;
}

bool Frame::IsEstimatedByLandmark(const int landmark_id) const
{
    return estLandmarkIds_.find(landmark_id) != estLandmarkIds_.end();
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
    if (hasStereo_)
    {
        cv::imencode(".png", stereoImage_, stereoImageComp_, param);
    }
    image_.release();
    depthImage_.release();
    stereoImage_.release();
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
    if (hasStereo_)
    {
        stereoImage_ = cv::imdecode(cv::Mat(1, stereoImageComp_.size(), CV_8UC1, stereoImageComp_.data()), CV_LOAD_IMAGE_UNCHANGED);
    }
    imageComp_.clear();
    depthImageComp_.clear();
    stereoImageComp_.clear();
    isCompressed_ = false;
}

bool Frame::IsCompressed() const
{
    return isCompressed_;
}

const double Frame::GetTime() const
{
    return timeSec_;
}

}
}
