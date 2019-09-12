#ifndef _FEATURE_H_
#define _FEATURE_H_

#include "frame.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

using namespace Eigen;

namespace omni_slam
{
namespace data
{

class Feature
{
public:
    Feature(Frame &frame, cv::KeyPoint kpt, cv::Mat descriptor);
    Feature(Frame &frame, cv::KeyPoint kpt);

    const Frame& GetFrame() const;
    const cv::KeyPoint& GetKeypoint() const;
    const cv::Mat& GetDescriptor() const;

    Vector3d GetBearing() const;
    Vector3d GetWorldPoint();
    Vector3d GetEstimatedWorldPoint();
    bool HasWorldPoint() const;
    bool HasEstimatedWorldPoint() const;

    bool worldPointCached_{false};
    bool worldPointEstimateCached_{false};
private:
    Frame &frame_;
    cv::KeyPoint kpt_;
    cv::Mat descriptor_;
    Vector3d worldPoint_;
    Vector3d worldPointEstimate_;
};

}
}

#endif /* _FEATURE_H_ */
