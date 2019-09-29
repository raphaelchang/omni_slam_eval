#ifndef _FIVE_POINT_H_
#define _FIVE_POINT_H_

#include "pose_estimator.h"
#include <Eigen/Dense>
#include "data/landmark.h"

using namespace Eigen;

namespace omni_slam
{
namespace odometry
{

class FivePoint : public PoseEstimator
{
public:
    FivePoint(int ransac_iterations, double epipolar_threshold, int num_ceres_threads = 1);

    int Compute(const std::vector<data::Landmark> &landmarks, data::Frame &cur_frame, data::Frame &prev_frame, std::vector<int> &inlier_indices) const;
    int ComputeE(const std::vector<data::Landmark> &landmarks, const data::Frame &frame1, const data::Frame &frame2, Matrix3d &E, std::vector<int> &inlier_indices) const;

private:
    int RANSAC(const std::vector<Vector3d> &x1, const std::vector<Vector3d> &x2, Matrix3d &E) const;
    void FivePointRelativePose(const std::vector<Vector3d> &x1, const std::vector<Vector3d> &x2, std::vector<Matrix3d> &Es) const;
    std::vector<int> GetInlierIndices(const std::vector<Vector3d> &x1, const std::vector<Vector3d> &x2, const Matrix3d &E) const;
    void EssentialToPoses(const Matrix3d &E, std::vector<Matrix3d> &rs, std::vector<Vector3d> &ts) const;
    void ComputeTranslation(const std::vector<Vector3d> &xs, const std::vector<const data::Feature*> &ys, const Matrix3d &R, Vector3d &t) const;
    Vector3d TriangulateDLT(const Vector3d &x1, const Vector3d &x2, const Matrix<double, 3, 4> &pose1, const Matrix<double, 3, 4> &pose2) const;

    int ransacIterations_;
    double epipolarThreshold_;
    int numCeresThreads_;
};

}
}

#endif /* _FIVE_POINT_H_ */
