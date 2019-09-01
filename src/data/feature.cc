#include "feature.h"

#include "util/tf_util.h"

namespace omni_slam
{
namespace data
{

Feature::Feature(Frame &frame, cv::KeyPoint kpt, cv::Mat descriptor)
    : frame_(frame),
    kpt_(kpt),
    descriptor_(descriptor)
{
}

Feature::Feature(Frame &frame, cv::KeyPoint kpt)
    : frame_(frame),
    kpt_(kpt)
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

Vector3d Feature::GetWorldPoint()
{
    if (worldPointCached_)
    {
        return worldPoint_;
    }
    Vector3d cameraFramePt;
    Vector2d pixelPt;
    pixelPt << kpt_.pt.x + 0.5, kpt_.pt.y + 0.5;
    frame_.GetCameraModel().UnprojectToBearing(pixelPt, cameraFramePt);
    bool wasCompressed = frame_.IsCompressed();
    cameraFramePt *= frame_.GetDepthImage().at<double>((int)(kpt_.pt.y + 0.5), (int)(kpt_.pt.x + 0.5));
    if (wasCompressed)
    {
        frame_.CompressImages();
    }
    Vector3d worldFramePt = util::TFUtil::CameraFrameToWorldFrame(cameraFramePt);
    worldPoint_ = util::TFUtil::TransformPoint(frame_.GetPose(), worldFramePt);
    worldPointCached_ = true;
    return worldPoint_;
}

bool Feature::HasWorldPoint() const
{
    return frame_.HasPose() && frame_.HasDepthImage();
}

}
}
