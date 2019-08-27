#include "feature.h"

#include "util/tf_util.h"

namespace omni_slam
{
namespace data
{

Feature::Feature(Frame &frame, cv::KeyPoint kpt, cv::Mat descriptor)
    : frame_(frame),
    kpt_(kpt),
    descriptor_(descriptor.clone())
{
}

Feature::Feature(Frame &frame, cv::KeyPoint kpt)
    : frame_(frame),
    kpt_(kpt)
{
}

Frame& Feature::GetFrame()
{
    return frame_;
}

cv::KeyPoint& Feature::GetKeypoint()
{
    return kpt_;
}

cv::Mat& Feature::GetDescriptor()
{
    return descriptor_;
}

Vector3d Feature::GetWorldPoint()
{
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
    return util::TFUtil::TransformPoint(frame_.GetPose(), worldFramePt);
}

bool Feature::HasWorldPoint()
{
    return frame_.HasPose() && frame_.HasDepthImage();
}

}
}
