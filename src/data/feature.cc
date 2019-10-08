#include "feature.h"

#include "util/tf_util.h"

namespace omni_slam
{
namespace data
{

Feature::Feature(Frame &frame, cv::KeyPoint kpt, cv::Mat descriptor, bool stereo)
    : frame_(frame),
    kpt_(kpt),
    descriptor_(descriptor),
    stereo_(stereo)
{
}

Feature::Feature(Frame &frame, cv::KeyPoint kpt, bool stereo)
    : frame_(frame),
    kpt_(kpt),
    stereo_(stereo)
{
}

const Frame& Feature::GetFrame() const
{
    return frame_;
}

const cv::KeyPoint& Feature::GetKeypoint() const
{
    return kpt_;
}

const cv::Mat& Feature::GetDescriptor() const
{
    return descriptor_;
}

void Feature::SetDescriptor(const cv::Mat& descriptor)
{
    descriptor_ = descriptor.clone();
}

Vector3d Feature::GetBearing() const
{
    Vector3d cameraFramePt;
    Vector2d pixelPt;
    pixelPt << kpt_.pt.x + 0.5, kpt_.pt.y + 0.5;
    if (stereo_)
    {
        frame_.GetStereoCameraModel().UnprojectToBearing(pixelPt, cameraFramePt);
    }
    else
    {
        frame_.GetCameraModel().UnprojectToBearing(pixelPt, cameraFramePt);
    }
    return util::TFUtil::CameraFrameToWorldFrame(cameraFramePt);
}

Vector3d Feature::GetWorldPoint()
{
    if (worldPointCached_)
    {
        return worldPoint_;
    }
    Vector3d worldFramePt = GetBearing();
    bool wasCompressed = frame_.IsCompressed();
    worldFramePt *= frame_.GetDepthImage().at<double>((int)(kpt_.pt.y + 0.5), (int)(kpt_.pt.x + 0.5));
    if (wasCompressed)
    {
        frame_.CompressImages();
    }
    worldPoint_ = util::TFUtil::TransformPoint(frame_.GetPose(), worldFramePt);
    worldPointCached_ = true;
    return worldPoint_;
}

Vector3d Feature::GetEstimatedWorldPoint()
{
    if (worldPointEstimateCached_)
    {
        return worldPointEstimate_;
    }
    Vector3d worldFramePt = GetBearing();
    bool wasCompressed = frame_.IsCompressed();
    worldFramePt *= frame_.GetDepthImage().at<double>((int)(kpt_.pt.y + 0.5), (int)(kpt_.pt.x + 0.5));
    if (wasCompressed)
    {
        frame_.CompressImages();
    }
    worldPointEstimate_ = util::TFUtil::TransformPoint(frame_.GetEstimatedPose(), worldFramePt);
    worldPointEstimateCached_ = true;
    return worldPointEstimate_;
}

bool Feature::HasWorldPoint() const
{
    return frame_.HasPose() && frame_.HasDepthImage();
}

bool Feature::HasEstimatedWorldPoint() const
{
    return frame_.HasEstimatedPose() && frame_.HasDepthImage();
}

}
}
