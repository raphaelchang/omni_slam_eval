#ifndef _PNP_H_
#define _PNP_H_

#include "pose_estimator.h"
#include <Eigen/Dense>
#include "data/landmark.h"
#include "data/frame.h"
#include "camera/camera_model.h"
#include <vector>

using namespace Eigen;

namespace omni_slam
{
namespace odometry
{

class PNP : public PoseEstimator
{
public:
    PNP(int ransac_iterations, double reprojection_threshold, int num_refine_threads = 1);
    int Compute(const std::vector<data::Landmark> &landmarks, data::Frame &cur_frame, data::Frame &prev_frame, std::vector<int> &inlier_indices) const;

private:
    int RANSAC(const std::vector<Vector3d> &xs, const std::vector<Vector3d> &ys, const std::vector<Vector2d> &yns, const camera::CameraModel<> &camera_model, Matrix<double, 3, 4> &pose) const;
    bool Refine(const std::vector<Vector3d> &xs, const std::vector<const data::Feature*> &features, const std::vector<int> indices, Matrix<double, 3, 4> &pose) const;
    double P4P(const std::vector<Vector3d> &xs, const std::vector<Vector3d> &ys, const std::vector<Vector2d> &yns, std::vector<int> indices, const camera::CameraModel<> &camera_model, Matrix<double, 3, 4> &pose) const;
    std::vector<int> GetInlierIndices(const std::vector<Vector3d> &xs, const std::vector<Vector2d> &yns, const Matrix<double, 3, 4> &pose, const camera::CameraModel<> &camera_model) const;

    int ransacIterations_;
    double reprojThreshold_;
    int numRefineThreads_;
};

}
}

#endif /* _PNP_H_ */
