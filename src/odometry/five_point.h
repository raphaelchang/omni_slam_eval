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
    FivePoint(int ransac_iterations, double epipolar_threshold, int trans_ransac_iterations, double reprojection_threshold, bool fix_translation_vector, int num_ceres_threads = 1);

    int Compute(const std::vector<data::Landmark> &landmarks, data::Frame &cur_frame, const data::Frame &prev_frame, std::vector<int> &inlier_indices) const;
    int Compute(const std::vector<data::Landmark> &landmarks, data::Frame &cur_frame, const data::Frame &prev_frame, std::vector<int> &inlier_indices, Matrix<double, 3, 4> &pose) const;
    int ComputeE(const std::vector<data::Landmark> &landmarks, const data::Frame &frame1, const data::Frame &frame2, Matrix3d &E, std::vector<int> &inlier_indices, bool stereo = false) const;

private:
    int ERANSAC(const std::vector<Vector3d> &x1, const std::vector<Vector3d> &x2, Matrix3d &E) const;
    void FivePointRelativePose(const std::vector<Vector3d> &x1, const std::vector<Vector3d> &x2, std::vector<Matrix3d> &Es) const;
    std::vector<int> GetEInlierIndices(const std::vector<Vector3d> &x1, const std::vector<Vector3d> &x2, const Matrix3d &E) const;
    void EssentialToPoses(const Matrix3d &E, std::vector<Matrix3d> &rs, std::vector<Vector3d> &ts) const;
    int ComputeTranslation(const std::vector<Vector3d> &xs, const std::vector<std::pair<const data::Feature*, const data::Feature*>> &ys, const Matrix3d &R, Vector3d &t, const Vector3d &tvec, std::vector<int> &inlier_indices) const;
    int TranslationRANSAC(const std::vector<Vector3d> &xs, const std::vector<std::pair<const data::Feature*, const data::Feature*>> &ys, const Matrix3d &R, Vector3d &t, const Vector3d &tvec) const;
    bool OptimizeTranslation(const std::vector<Vector3d> &xs, const std::vector<std::pair<const data::Feature*, const data::Feature*>> &ys, const Matrix3d &R, Vector3d &t, const Vector3d &tvec) const;
    std::vector<int> GetTranslationInlierIndices(const std::vector<Vector3d> &xs, const std::vector<std::pair<const data::Feature*, const data::Feature*>> &ys, const Matrix3d &R, const Vector3d &t) const;
    Vector3d TriangulateDLT(const Vector3d &x1, const Vector3d &x2, const Matrix<double, 3, 4> &pose1, const Matrix<double, 3, 4> &pose2) const;

    int ransacIterations_;
    int transIterations_;
    double epipolarThreshold_;
    double reprojectionThreshold_;
    bool fixTransVec_;
    int numCeresThreads_;
};

}
}

#endif /* _FIVE_POINT_H_ */
