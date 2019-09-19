#include "triangulator.h"

#include "util/tf_util.h"

namespace omni_slam
{
namespace reconstruction
{

Triangulator::Triangulator(double max_reprojection_error, double min_triangulation_angle)
    : cosMinTriangulationAngle_(cos(min_triangulation_angle * M_PI / 180)),
    maxReprojError_(max_reprojection_error)
{
}

int Triangulator::Triangulate(std::vector<data::Landmark> &landmarks) const
{
    int numSuccess = 0;
    #pragma omp parallel for reduction(+:numSuccess)
    for (auto it = landmarks.begin(); it < landmarks.end(); it++)
    {
        data::Landmark &landmark = *it;
        if (landmark.HasEstimatedPosition() && landmark.GetStereoObservationByFrameID(landmark.GetFirstFrameID()) != nullptr)
        {
            continue;
        }
        Vector3d point;
        if (TriangulateNViews(landmark.GetObservations(), point))
        {
            if (CheckReprojectionErrors(landmark.GetObservations(), point))
            {
                std::vector<int> frameIds;
                frameIds.reserve(landmark.GetObservations().size());
                for (data::Feature &obs : landmark.GetObservations())
                {
                    frameIds.push_back(obs.GetFrame().GetID());
                }
                landmark.SetEstimatedPosition(point, frameIds);
                numSuccess++;
            }
        }
    }
    return numSuccess;
}

bool Triangulator::TriangulateNViews(const std::vector<data::Feature> &views, Vector3d &point) const
{
    Matrix4d cost = Matrix4d::Zero();
    std::vector<Vector3d> bearings;
    for (const data::Feature &view : views)
    {
        Matrix<double, 3, 4> pose;
        Matrix<double, 3, 4> invpose;
        if (view.GetFrame().HasEstimatedPose())
        {
            pose = view.GetFrame().GetEstimatedInversePose();
            invpose = view.GetFrame().GetEstimatedPose();
        }
        else if (view.GetFrame().HasPose())
        {
            pose = view.GetFrame().GetInversePose();
            invpose = view.GetFrame().GetPose();
        }
        else
        {
            continue;
        }
        Vector3d bearing = view.GetBearing().normalized();
        bearings.push_back(util::TFUtil::RotatePoint(util::TFUtil::GetRotationFromPoseMatrix(invpose), bearing));
        const Matrix<double, 3, 4> A = pose - bearing * bearing.transpose() * pose;
        cost += A.transpose() * A;
    }
    if (!CheckAngleCoverage(bearings))
    {
        return false;
    }
    Eigen::SelfAdjointEigenSolver<Matrix4d> solver(cost);
    point = solver.eigenvectors().col(0).hnormalized();
    return solver.info() == Eigen::Success;
}

bool Triangulator::CheckReprojectionErrors(const std::vector<data::Feature> &views, const Vector3d &point) const
{
    for (const data::Feature &view : views)
    {
        Matrix<double, 3, 4> pose;
        if (view.GetFrame().HasEstimatedPose())
        {
            pose = view.GetFrame().GetEstimatedInversePose();
        }
        else if (view.GetFrame().HasPose())
        {
            pose = view.GetFrame().GetInversePose();
        }
        else
        {
            continue;
        }
        Vector2d reprojPix;
        if (view.GetFrame().GetCameraModel().ProjectToImage(util::TFUtil::WorldFrameToCameraFrame(util::TFUtil::TransformPoint(pose, point)), reprojPix))
        {
            if (maxReprojError_ > 0)
            {
                Vector2d kpt;
                kpt << view.GetKeypoint().pt.x, view.GetKeypoint().pt.y;
                double reprojError = (reprojPix - kpt).norm();
                if (reprojError > maxReprojError_)
                {
                    return false;
                }
            }
        }
        else
        {
            return false;
        }
    }
    return true;
}

bool Triangulator::CheckAngleCoverage(const std::vector<Vector3d> &bearings) const
{
    for (int i = 0; i < bearings.size(); i++)
    {
        for (int j = i + 1; j < bearings.size(); j++)
        {
            if (fabs(bearings[i].dot(bearings[j])) < cosMinTriangulationAngle_)
            {
                return true;
            }
        }
    }
    return false;
}

}
}
