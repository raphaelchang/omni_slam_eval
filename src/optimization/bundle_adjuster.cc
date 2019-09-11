#include "bundle_adjuster.h"

#include "reprojection_error.h"

namespace omni_slam
{
namespace optimization
{

BundleAdjuster::BundleAdjuster(int max_iterations, bool log)
{
    problem_.reset(new ceres::Problem());
    solverOptions_.max_num_iterations = max_iterations;
    solverOptions_.linear_solver_type = ceres::DENSE_SCHUR;
    solverOptions_.minimizer_progress_to_stdout = log;
    solverOptions_.logging_type = log ? ceres::PER_MINIMIZER_ITERATION : ceres::SILENT;
    //solverOptions_.check_gradients = true;
}

bool BundleAdjuster::Optimize(std::vector<data::Landmark> &landmarks)
{
    int numLandmarks = 0;
    for (const data::Landmark &landmark : landmarks)
    {
        if (landmark.HasEstimatedPosition() || landmark.HasGroundTruth())
        {
            numLandmarks++;
        }
    }
    std::vector<double> landmarkEstimates(3 * numLandmarks, 0);
    std::map<int, std::vector<double>> framePoses;
    std::map<int, data::Frame*> estFrames;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    for (const data::Landmark &landmark : landmarks)
    {
        bool hasEstCameraPoses = false;
        for (const data::Feature &feature : landmark.GetObservations())
        {
            if (feature.GetFrame().HasEstimatedPose())
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
            if (!feature.GetFrame().HasEstimatedPose() && feature.GetFrame().HasPose())
            {
                if (!landmark.HasEstimatedPosition())
                {
                    continue;
                }
                if (framePoses.find(feature.GetFrame().GetID()) == framePoses.end())
                {
                    framePoses[feature.GetFrame().GetID()].insert(framePoses[feature.GetFrame().GetID()].end(), feature.GetFrame().GetInversePose().data(), feature.GetFrame().GetInversePose().data() + 12);
                    problem_->AddParameterBlock(&framePoses[feature.GetFrame().GetID()][0], 12);
                    problem_->SetParameterBlockConstant(&framePoses[feature.GetFrame().GetID()][0]);
                }
            }
            else if (feature.GetFrame().HasEstimatedPose())
            {
                if (framePoses.find(feature.GetFrame().GetID()) == framePoses.end())
                {
                    framePoses[feature.GetFrame().GetID()].insert(framePoses[feature.GetFrame().GetID()].end(), feature.GetFrame().GetEstimatedInversePose().data(), feature.GetFrame().GetEstimatedInversePose().data() + 12);
                    problem_->AddParameterBlock(&framePoses[feature.GetFrame().GetID()][0], 12);
                    estFrames[feature.GetFrame().GetID()] = const_cast<data::Frame*>(&feature.GetFrame());
                }
            }
            else
            {
                continue;
            }
            ceres::CostFunction *cost_function = ReprojectionError::Create(feature);
            problem_->AddResidualBlock(cost_function, loss_function, &framePoses[feature.GetFrame().GetID()][0], &landmarkEstimates[landmarkEstimates.size() - 3]);
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
        bool hasEstCameraPoses = false;
        for (const data::Feature &feature : landmark.GetObservations())
        {
            if (feature.GetFrame().HasEstimatedPose())
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
        const Matrix<double, 3, 4> pose = Map<const Matrix<double, 3, 4>>(&framePoses[frame.first][0]);
        frame.second->SetEstimatedInversePose(pose);
    }
    return true;
}

}
}
