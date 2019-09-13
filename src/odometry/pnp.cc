#include "pnp.h"
#include "lambda_twist.h"

#include "util/tf_util.h"

#include <random>
#include <set>

namespace omni_slam
{
namespace odometry
{

PNP::PNP(int ransac_iterations, double reprojection_threshold)
    : ransacIterations_(ransac_iterations),
    reprojThreshold_(reprojection_threshold)
{
}

int PNP::Compute(const std::vector<data::Landmark> &landmarks, data::Frame &frame) const
{
    std::vector<Vector3d> xs;
    std::vector<Vector2d> yns;
    std::vector<Vector3d> ys;
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
        }
    }
    Matrix<double, 3, 4> pose;
    int inliers = RANSAC(xs, ys, yns, frame.GetCameraModel(), pose);
    frame.SetEstimatedInversePose(pose);
    return inliers;
}

int PNP::RANSAC(const std::vector<Vector3d> &xs, const std::vector<Vector3d> &ys, const std::vector<Vector2d> &yns, const camera::CameraModel<> &camera_model, Matrix<double, 3, 4> &pose) const
{
    int maxInliers = 0;
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

        int inliers = CountInliers(xs, yns, iterPose, camera_model);
        if (inliers > maxInliers)
        {
            maxInliers = inliers;
            pose = iterPose;
        }
    }
    return maxInliers;
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

int PNP::CountInliers(const std::vector<Vector3d> &xs, const std::vector<Vector2d> &yns, const Matrix<double, 3, 4> &pose, const camera::CameraModel<> &camera_model) const
{
    int inliers = 0;
    double thresh = reprojThreshold_ * reprojThreshold_;
    for (int i = 0; i < xs.size(); i++)
    {
        Vector2d xr;
        if (!camera_model.ProjectToImage(util::TFUtil::WorldFrameToCameraFrame(util::TFUtil::TransformPoint(pose, xs[i])), xr))
        {
            continue;
        }
        double err = (xr - yns[i]).squaredNorm();
        inliers += (err < thresh) ? 1 : 0;
    }
    return inliers;
}

}
}

