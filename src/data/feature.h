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

    Frame& GetFrame();
    cv::KeyPoint& GetKeypoint();
    cv::Mat& GetDescriptor();

    Vector3d GetWorldPoint();
    bool HasWorldPoint();

private:
    Frame &frame_;
    cv::KeyPoint kpt_;
    cv::Mat descriptor_;
};

}
}

#endif /* _FEATURE_H_ */
