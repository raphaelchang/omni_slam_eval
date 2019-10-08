#include "stereo_matcher.h"

#include "util/tf_util.h"

namespace omni_slam
{
namespace stereo
{

StereoMatcher::StereoMatcher(double epipolar_thresh)
    : epipolarThresh_(epipolar_thresh)
{
}

int StereoMatcher::Match(data::Frame &frame, std::vector<data::Landmark> &landmarks) const
{
    if (!frame.HasStereoImage())
    {
        return 0;
    }
    Matrix<double, 3, 4> framePose;
    if (frame.HasEstimatedPose())
    {
        framePose = frame.GetEstimatedPose();
    }
    else if (frame.HasPose())
    {
        framePose = frame.GetPose();
    }
    else
    {
        return 0;
    }
    std::vector<cv::KeyPoint> pointsToMatch;
    std::vector<cv::Mat> descriptors;
    std::vector<Vector3d> bearings1;
    std::vector<int> origInx;
    for (int i = 0; i < landmarks.size(); i++)
    {
        data::Landmark &landmark = landmarks[i];
        if (landmark.HasEstimatedPosition())
        {
            continue;
        }
        const data::Feature *feat = landmark.GetObservationByFrameID(frame.GetID());
        if (feat != nullptr)
        {
            pointsToMatch.push_back(feat->GetKeypoint());
            descriptors.push_back(feat->GetDescriptor());
            bearings1.push_back(util::TFUtil::WorldFrameToCameraFrame(feat->GetBearing().normalized()));
            origInx.push_back(i);
        }
    }
    if (pointsToMatch.size() == 0)
    {
        return 0;
    }

    std::vector<cv::KeyPoint> matchedPoints;
    std::vector<int> matchedIndices;
    FindMatches(frame.GetImage(), frame.GetStereoImage(), pointsToMatch, matchedPoints, matchedIndices);

    Matrix<double, 3, 4> I = util::TFUtil::IdentityPoseMatrix<double>();
    Matrix3d E = util::TFUtil::GetEssentialMatrixFromPoses(I, frame.GetStereoPose());
    int good = 0;
    #pragma omp parallel for reduction(+:good)
    for (auto it = matchedIndices.begin(); it < matchedIndices.end(); it++)
    {
        int inx = *it;
        Vector3d &bearing1 = bearings1[inx];
        data::Feature feat(frame, matchedPoints[inx], descriptors[inx], true);
        Vector3d bearing2 = util::TFUtil::WorldFrameToCameraFrame(feat.GetBearing().normalized());

        RowVector3d epiplane1 = bearing2.transpose() * E;
        epiplane1.normalize();
        double epiErr1 = std::abs(epiplane1 * bearing1);
        Vector3d epiplane2 = E * bearing1;
        epiplane2.normalize();
        double epiErr2 = std::abs(bearing2.transpose() * epiplane2);
        if (epiErr1 < epipolarThresh_ && epiErr2 < epipolarThresh_)
        {
            landmarks[origInx[inx]].SetEstimatedPosition(util::TFUtil::TransformPoint(framePose, util::TFUtil::CameraFrameToWorldFrame(TriangulateDLT(bearing1, bearing2, I, frame.GetStereoPose()))), std::vector<int>({frame.GetID()}));
            landmarks[origInx[inx]].AddStereoObservation(feat);
            good++;
        }
    }
    return good;
}

Vector3d StereoMatcher::TriangulateDLT(const Vector3d &x1, const Vector3d &x2, const Matrix<double, 3, 4> &pose1, const Matrix<double, 3, 4> &pose2) const
{
    Matrix4d design;
    design.row(0) = x1(0) * pose1.row(2) - x1(2) * pose1.row(0);
    design.row(1) = x1(1) * pose1.row(2) - x1(2) * pose1.row(1);
    design.row(2) = x2(0) * pose2.row(2) - x2(2) * pose2.row(0);
    design.row(3) = x2(1) * pose2.row(2) - x2(2) * pose2.row(1);

    Eigen::JacobiSVD<Matrix4d> svd(design, Eigen::ComputeFullV);
    return svd.matrixV().col(3).hnormalized();
}

}
}
