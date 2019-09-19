#include "pnp.h"
#include "lambda_twist.h"

#include <ceres/ceres.h>
#include "optimization/reprojection_error.h"
#include "camera/double_sphere.h"
#include "camera/perspective.h"

#include "util/tf_util.h"

#include <random>
#include <set>

namespace omni_slam
{
namespace odometry
{

PNP::PNP(int ransac_iterations, double reprojection_threshold, int num_refine_threads)
    : ransacIterations_(ransac_iterations),
    reprojThreshold_(reprojection_threshold),
    numRefineThreads_(num_refine_threads)
{
}

int PNP::Compute(const std::vector<data::Landmark> &landmarks, data::Frame &frame, std::vector<int> &inlier_indices) const
{
    std::vector<Vector3d> xs;
    std::vector<Vector2d> yns;
    std::vector<Vector3d> ys;
    std::vector<const data::Feature*> features;
    std::map<int, int> indexToId;
    std::map<int, int> indexToLandmarkIndex;
    int i = 0;
    for (const data::Landmark &landmark : landmarks)
    {
        if (landmark.IsObservedInFrame(frame.GetID()))
        {
            if (landmark.HasEstimatedPosition())
            {
                xs.push_back(landmark.GetEstimatedPosition());
            }
            else if (landmark.HasGroundTruth())
            {
                xs.push_back(landmark.GetGroundTruth());
            }
            else
            {
                continue;
            }
            const data::Feature *feat = landmark.GetObservationByFrameID(frame.GetID());
            Vector2d pix;
            pix << feat->GetKeypoint().pt.x, feat->GetKeypoint().pt.y;
            yns.push_back(pix);
            ys.push_back(feat->GetBearing());
            features.push_back(feat);
            indexToLandmarkIndex[xs.size() - 1] = i;
            indexToId[xs.size() - 1] = landmark.GetID();
        }
        i++;
    }
    if (xs.size() < 4)
    {
        return 0;
    }
    Matrix<double, 3, 4> pose;
    int inliers = RANSAC(xs, ys, yns, frame.GetCameraModel(), pose);
    std::vector<int> indices = GetInlierIndices(xs, yns, pose, frame.GetCameraModel());
    if (inliers > 3)
    {
        Refine(xs, features, indices, pose);
    }
    std::vector<int> inlierIds;
    inlierIds.reserve(indices.size());
    inlier_indices.clear();
    inlier_indices.reserve(indices.size());
    for (int inx : indices)
    {
        inlierIds.push_back(indexToId[inx]);
        inlier_indices.push_back(indexToLandmarkIndex[inx]);
    }
    frame.SetEstimatedInversePose(pose, inlierIds);
    return inliers;
}

int PNP::Compute(const std::vector<data::Landmark> &landmarks, data::Frame &frame) const
{
    std::vector<int> temp;
    return Compute(landmarks, frame, temp);
}

int PNP::RANSAC(const std::vector<Vector3d> &xs, const std::vector<Vector3d> &ys, const std::vector<Vector2d> &yns, const camera::CameraModel<> &camera_model, Matrix<double, 3, 4> &pose) const
{
    int maxInliers = 0;
    #pragma omp parallel for
    for (int i = 0; i < ransacIterations_; i++)
    {
        std::random_device rd;
        std::mt19937 eng(rd());
        std::uniform_int_distribution<> distr(0, xs.size() - 1);
        std::set<int> randset;
        while (randset.size() < 4)
        {
            randset.insert(distr(eng));
        }
        std::vector<int> indices;
        std::copy(randset.begin(), randset.end(), std::back_inserter(indices));

        Matrix<double, 3, 4> iterPose;
        double err = P4P(xs, ys, yns, indices, camera_model, iterPose);
        if (std::isnan(err))
        {
            continue;
        }

        int inliers = GetInlierIndices(xs, yns, iterPose, camera_model).size();
        #pragma omp critical
        {
            if (inliers > maxInliers)
            {
                maxInliers = inliers;
                pose = iterPose;
            }
        }
    }
    return maxInliers;
}

bool PNP::Refine(const std::vector<Vector3d> &xs, const std::vector<const data::Feature*> &features, const std::vector<int> indices, Matrix<double, 3, 4> &pose) const
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function = nullptr;
    std::vector<double> landmarks;
    landmarks.reserve(3 * indices.size());
    Quaterniond quat(pose.block<3, 3>(0, 0));
    quat.normalize();
    Vector3d t(pose.block<3, 1>(0, 3));
    std::vector<double> quatData(quat.coeffs().data(), quat.coeffs().data() + 4);
    std::vector<double> tData(t.data(), t.data() + 3);
    for (int i : indices)
    {
        const Vector3d &x = xs[i];
        landmarks.push_back(x(0));
        landmarks.push_back(x(1));
        landmarks.push_back(x(2));
        ceres::CostFunction *cost_function = nullptr;
        if (features[i]->GetFrame().GetCameraModel().GetType() == camera::CameraModel<>::kPerspective)
        {
            cost_function = optimization::ReprojectionError<camera::Perspective>::Create(*features[i]);
        }
        else if (features[i]->GetFrame().GetCameraModel().GetType() == camera::CameraModel<>::kDoubleSphere)
        {
            cost_function = optimization::ReprojectionError<camera::DoubleSphere>::Create(*features[i]);
        }
        if (cost_function != nullptr)
        {
            problem.AddResidualBlock(cost_function, loss_function, &quatData[0], &tData[0], &landmarks[landmarks.size() - 3]);
            problem.SetParameterBlockConstant(&landmarks[landmarks.size() - 3]);
        }
    }
    ceres::Solver::Options options;
    options.max_num_iterations = 10;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.function_tolerance = 1e-6;
    options.gradient_tolerance = 1e-6;
    options.num_threads = numRefineThreads_;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (!summary.IsSolutionUsable())
    {
        return false;
    }
    const Quaterniond quatRes = Map<const Quaterniond>(&quatData[0]);
    const Vector3d tRes = Map<const Vector3d>(&tData[0]);
    pose = util::TFUtil::QuaternionTranslationToPoseMatrix(quatRes, tRes);
    return true;
}

double PNP::P4P(const std::vector<Vector3d> &xs, const std::vector<Vector3d> &ys, const std::vector<Vector2d> &yns, std::vector<int> indices, const camera::CameraModel<> &camera_model, Matrix<double, 3, 4> &pose) const
{
    std::vector<Matrix3d> Rs(4);
    std::vector<Vector3d> Ts(4);

    int valid = LambdaTwist::P3P<double, 5>(ys[indices[0]], ys[indices[1]], ys[indices[2]], xs[indices[0]], xs[indices[1]], xs[indices[2]], Rs, Ts);

    const Vector2d &y = yns[indices[3]];
    const Vector3d &x = xs[indices[3]];

    if (valid == 0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double e0 = std::numeric_limits<double>::max();

    for (int v = 0; v < valid; ++v)
    {
        Matrix<double, 3, 4> tmp;
        tmp.block<3, 3>(0, 0) = Rs[v];
        tmp.block<3, 1>(0, 3) = Ts[v];

        Vector2d xr;
        if (!camera_model.ProjectToImage(util::TFUtil::WorldFrameToCameraFrame(util::TFUtil::TransformPoint(tmp, x)), xr))
        {
            continue;
        }
        double e = (xr - y).array().abs().sum();
        if (std::isnan(e))
        {
            continue;
        }
        if (e < e0 && ((tmp - tmp).array() == (tmp - tmp).array()).all() && ((tmp.array() == tmp.array()).all()))
        {
            pose = tmp;
            e0 = e;
        }
    }

    double error = 0;
    for (int i = 0; i < 4; i++)
    {
        Vector2d xr;
        camera_model.ProjectToImage(util::TFUtil::WorldFrameToCameraFrame(util::TFUtil::TransformPoint(pose, xs[indices[i]])), xr);
        double e = (xr - yns[indices[i]]).squaredNorm();
        error += e;
    }
    return error;
}

std::vector<int> PNP::GetInlierIndices(const std::vector<Vector3d> &xs, const std::vector<Vector2d> &yns, const Matrix<double, 3, 4> &pose, const camera::CameraModel<> &camera_model) const
{
    std::vector<int> indices;
    double thresh = reprojThreshold_ * reprojThreshold_;
    for (int i = 0; i < xs.size(); i++)
    {
        Vector2d xr;
        if (!camera_model.ProjectToImage(util::TFUtil::WorldFrameToCameraFrame(util::TFUtil::TransformPoint(pose, xs[i])), xr))
        {
            continue;
        }
        double err = (xr - yns[i]).squaredNorm();
        if (err < thresh)
        {
            indices.push_back(i);
        }
    }
    return indices;
}

}
}

