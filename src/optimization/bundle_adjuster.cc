#include "bundle_adjuster.h"

#include "reprojection_error.h"

#include "camera/double_sphere.h"
#include "camera/unified.h"
#include "camera/perspective.h"

namespace omni_slam
{
namespace optimization
{

BundleAdjuster::BundleAdjuster(int max_iterations, double loss_coeff, int num_threads, bool log)
    : lossCoeff_(loss_coeff)
{
    problem_.reset(new ceres::Problem());
    solverOptions_.max_num_iterations = max_iterations;
    solverOptions_.linear_solver_type = ceres::ITERATIVE_SCHUR;
    solverOptions_.preconditioner_type = ceres::SCHUR_JACOBI;
    solverOptions_.num_threads = num_threads;
    solverOptions_.use_inner_iterations = true;
    solverOptions_.minimizer_progress_to_stdout = log;
    solverOptions_.logging_type = log ? ceres::PER_MINIMIZER_ITERATION : ceres::SILENT;
}

bool BundleAdjuster::Optimize(std::vector<data::Landmark> &landmarks, const std::vector<int> &frame_ids)
{
    std::vector<double> landmarkEstimates;
    landmarkEstimates.reserve(3 * landmarks.size());
    std::map<int, std::pair<std::vector<double>, std::vector<double>>> framePoses;
    std::map<int, data::Frame*> estFrames;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(lossCoeff_);
    for (const data::Landmark &landmark : landmarks)
    {
        if (frame_ids.size() > 0)
        {
            bool observed = false;
            for (int id : frame_ids)
            {
                if (landmark.IsObservedInFrame(id))
                {
                    observed = true;
                    break;
                }
            }
            if (!observed)
            {
                continue;
            }
        }
        bool hasEstCameraPoses = false;
        for (const data::Feature &feature : landmark.GetObservations())
        {
            if (frame_ids.size() > 0)
            {
                if (std::find(frame_ids.begin(), frame_ids.end(), feature.GetFrame().GetID()) == frame_ids.end())
                {
                    continue;
                }
            }
            if (feature.GetFrame().HasEstimatedPose() && feature.GetFrame().IsEstimatedByLandmark(landmark.GetID()))
            {
                hasEstCameraPoses = true;
            }
        }
        if (!landmark.HasEstimatedPosition() && landmark.HasGroundTruth() && hasEstCameraPoses)
        {
            Vector3d gnd = landmark.GetGroundTruth();
            landmarkEstimates.push_back(gnd(0));
            landmarkEstimates.push_back(gnd(1));
            landmarkEstimates.push_back(gnd(2));
            problem_->AddParameterBlock(&landmarkEstimates[landmarkEstimates.size() - 3], 3);
            problem_->SetParameterBlockConstant(&landmarkEstimates[landmarkEstimates.size() - 3]);
        }
        else if (landmark.HasEstimatedPosition())
        {
            Vector3d est = landmark.GetEstimatedPosition();
            landmarkEstimates.push_back(est(0));
            landmarkEstimates.push_back(est(1));
            landmarkEstimates.push_back(est(2));
            problem_->AddParameterBlock(&landmarkEstimates[landmarkEstimates.size() - 3], 3);
        }
        else
        {
            continue;
        }
        for (const data::Feature &feature : landmark.GetObservations())
        {
            if (frame_ids.size() > 0)
            {
                if (std::find(frame_ids.begin(), frame_ids.end(), feature.GetFrame().GetID()) == frame_ids.end())
                {
                    continue;
                }
            }
            if (!feature.GetFrame().HasEstimatedPose() && feature.GetFrame().HasPose())
            {
                if (!landmark.HasEstimatedPosition())
                {
                    continue;
                }
                if (!landmark.IsEstimatedByFrame(feature.GetFrame().GetID()))
                {
                    continue;
                }
                if (framePoses.find(feature.GetFrame().GetID()) == framePoses.end())
                {
                    const Matrix<double, 3, 4> &pose = feature.GetFrame().GetInversePose();
                    Quaterniond quat(pose.block<3, 3>(0, 0));
                    quat.normalize();
                    const Vector3d &t = pose.block<3, 1>(0, 3);
                    framePoses[feature.GetFrame().GetID()] = {std::vector<double>(quat.coeffs().data(), quat.coeffs().data() + 4), std::vector<double>(t.data(), t.data() + 3)};
                    problem_->AddParameterBlock(&framePoses[feature.GetFrame().GetID()].first[0], 4);
                    problem_->AddParameterBlock(&framePoses[feature.GetFrame().GetID()].second[0], 3);
                    problem_->SetParameterBlockConstant(&framePoses[feature.GetFrame().GetID()].first[0]);
                    problem_->SetParameterBlockConstant(&framePoses[feature.GetFrame().GetID()].second[0]);
                }
            }
            else if (feature.GetFrame().HasEstimatedPose())
            {
                if (!(feature.GetFrame().IsEstimatedByLandmark(landmark.GetID()) || (landmark.HasEstimatedPosition() && landmark.IsEstimatedByFrame(feature.GetFrame().GetID()))))
                {
                    continue;
                }
                if (framePoses.find(feature.GetFrame().GetID()) == framePoses.end())
                {
                    const Matrix<double, 3, 4> &pose = feature.GetFrame().GetEstimatedInversePose();
                    Quaterniond quat(pose.block<3, 3>(0, 0));
                    quat.normalize();
                    const Vector3d &t = pose.block<3, 1>(0, 3);
                    framePoses[feature.GetFrame().GetID()] = {std::vector<double>(quat.coeffs().data(), quat.coeffs().data() + 4), std::vector<double>(t.data(), t.data() + 3)};
                    problem_->AddParameterBlock(&framePoses[feature.GetFrame().GetID()].first[0], 4);
                    problem_->AddParameterBlock(&framePoses[feature.GetFrame().GetID()].second[0], 3);
                    problem_->SetParameterization(&framePoses[feature.GetFrame().GetID()].first[0], new ceres::EigenQuaternionParameterization());
                    estFrames[feature.GetFrame().GetID()] = const_cast<data::Frame*>(&feature.GetFrame());
                }
            }
            else
            {
                continue;
            }
            ceres::CostFunction *cost_function = nullptr;
            const data::Feature *stereoFeat = feature.GetFrame().HasStereoImage() ? landmark.GetStereoObservationByFrameID(feature.GetFrame().GetID()) : nullptr;
            if (feature.GetFrame().GetCameraModel().GetType() == camera::CameraModel<>::kPerspective)
            {
                if (stereoFeat != nullptr)
                {
                    cost_function = ReprojectionError<camera::Perspective>::Create(feature, *stereoFeat);
                }
                else
                {
                    cost_function = ReprojectionError<camera::Perspective>::Create(feature);
                }
            }
            else if (feature.GetFrame().GetCameraModel().GetType() == camera::CameraModel<>::kDoubleSphere)
            {
                if (stereoFeat != nullptr)
                {
                    cost_function = ReprojectionError<camera::DoubleSphere>::Create(feature, *stereoFeat);
                }
                else
                {
                    cost_function = ReprojectionError<camera::DoubleSphere>::Create(feature);
                }
            }
            else if (feature.GetFrame().GetCameraModel().GetType() == camera::CameraModel<>::kUnified)
            {
                if (stereoFeat != nullptr)
                {
                    cost_function = ReprojectionError<camera::Unified>::Create(feature, *stereoFeat);
                }
                else
                {
                    cost_function = ReprojectionError<camera::Unified>::Create(feature);
                }
            }
            if (cost_function != nullptr)
            {
                problem_->AddResidualBlock(cost_function, loss_function, &framePoses[feature.GetFrame().GetID()].first[0], &framePoses[feature.GetFrame().GetID()].second[0], &landmarkEstimates[landmarkEstimates.size() - 3]);
            }
        }
    }

    ceres::Solver::Summary summary;
    ceres::Solve(solverOptions_, problem_.get(), &summary);
    if (solverOptions_.minimizer_progress_to_stdout)
    {
        std::cout << summary.FullReport() << std::endl;
    }
    if (!summary.IsSolutionUsable())
    {
        return false;
    }

    int inx = 0;
    for (data::Landmark &landmark : landmarks)
    {
        if (frame_ids.size() > 0)
        {
            bool observed = false;
            for (int id : frame_ids)
            {
                if (landmark.IsObservedInFrame(id))
                {
                    observed = true;
                    break;
                }
            }
            if (!observed)
            {
                continue;
            }
        }
        bool hasEstCameraPoses = false;
        for (const data::Feature &feature : landmark.GetObservations())
        {
            if (frame_ids.size() > 0)
            {
                if (std::find(frame_ids.begin(), frame_ids.end(), feature.GetFrame().GetID()) == frame_ids.end())
                {
                    continue;
                }
            }
            if (feature.GetFrame().HasEstimatedPose() && feature.GetFrame().IsEstimatedByLandmark(landmark.GetID()))
            {
                hasEstCameraPoses = true;
            }
        }
        if (!landmark.HasEstimatedPosition() && landmark.HasGroundTruth() && hasEstCameraPoses)
        {
            inx++;
        }
        else if (landmark.HasEstimatedPosition())
        {
            Vector3d est;
            est << landmarkEstimates[inx * 3], landmarkEstimates[inx * 3 + 1], landmarkEstimates[inx * 3 + 2];
            landmark.SetEstimatedPosition(est);
            inx++;
        }
    }
    for (auto &frame : estFrames)
    {
        const Quaterniond quat = Map<const Quaterniond>(&framePoses[frame.first].first[0]);
        const Vector3d t = Map<const Vector3d>(&framePoses[frame.first].second[0]);
        const Matrix<double, 3, 4> pose = util::TFUtil::QuaternionTranslationToPoseMatrix(quat, t);
        frame.second->SetEstimatedInversePose(pose);
    }
    problem_.reset(new ceres::Problem());
    return true;
}

bool BundleAdjuster::Optimize(std::vector<data::Landmark> &landmarks)
{
    std::vector<int> tmp;
    Optimize(landmarks, tmp);
}

}
}
