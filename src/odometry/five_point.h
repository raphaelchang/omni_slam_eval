#ifndef _FIVE_POINT_H_
#define _FIVE_POINT_H_

#include <Eigen/Dense>
#include "data/landmark.h"

using namespace Eigen;

namespace omni_slam
{
namespace odometry
{

class FivePoint
{
public:
    FivePoint(int ransac_iterations, double epipolar_threshold);

    int Compute(const std::vector<data::Landmark> &landmarks, const data::Frame &frame1, const data::Frame &frame2, Matrix3d &E, std::vector<int> &inlier_indices) const;
    int Compute(const std::vector<data::Landmark> &landmarks, const data::Frame &frame1, const data::Frame &frame2, Matrix3d &E) const;

private:
    int RANSAC(const std::vector<Vector3d> &x1, const std::vector<Vector3d> &x2, Matrix3d &E) const;
    void FivePointRelativePose(const std::vector<Vector3d> &x1, const std::vector<Vector3d> &x2, std::vector<Matrix3d> &Es) const;
    std::vector<int> GetInlierIndices(const std::vector<Vector3d> &x1, const std::vector<Vector3d> &x2, const Matrix3d &E) const;

    int ransacIterations_;
    double epipolarThreshold_;

};

}
}

#endif /* _FIVE_POINT_H_ */
