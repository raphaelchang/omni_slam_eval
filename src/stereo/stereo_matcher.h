#ifndef _STEREO_MATCHER_H_
#define _STEREO_MATCHER_H_

#include "data/landmark.h"
#include <Eigen/Dense>

using namespace Eigen;

namespace omni_slam
{
namespace stereo
{

class StereoMatcher
{
public:
    StereoMatcher(double epipolar_thresh);

    int Match(data::Frame &frame, std::vector<data::Landmark> &landmarks) const;

private:
    virtual void FindMatches(const cv::Mat &image1, const cv::Mat &image2, const std::vector<cv::KeyPoint> &pts1, std::vector<cv::KeyPoint> &pts2, std::vector<int> &matchedIndices) const = 0;
    Vector3d TriangulateDLT(const Vector3d &x1, const Vector3d &x2, const Matrix<double, 3, 4> &pose1, const Matrix<double, 3, 4> &pose2) const;

    double epipolarThresh_;
};

}
}

#endif /* _STEREO_MATCHER_H_ */
