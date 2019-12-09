#include "five_point.h"

#include "util/math_util.h"
#include "util/tf_util.h"
#include <set>
#include <random>
#include <ceres/ceres.h>
#include "optimization/reprojection_error.h"
#include "optimization/scale_parameterization.h"
#include "camera/double_sphere.h"
#include "camera/unified.h"
#include "camera/perspective.h"

namespace omni_slam
{
namespace odometry
{

FivePoint::FivePoint(int ransac_iterations, double epipolar_threshold, int trans_ransac_iterations, double reprojection_threshold, bool fix_translation_vector, int num_ceres_threads)
    : ransacIterations_(ransac_iterations),
    epipolarThreshold_(epipolar_threshold),
    transIterations_(trans_ransac_iterations),
    reprojectionThreshold_(reprojection_threshold),
    fixTransVec_(fix_translation_vector),
    numCeresThreads_(num_ceres_threads)
{
}

int FivePoint::Compute(const std::vector<data::Landmark> &landmarks, data::Frame &cur_frame, const data::Frame &prev_frame, std::vector<int> &inlier_indices) const
{
    Matrix<double, 3, 4> poseTmp;
    return Compute(landmarks, cur_frame, prev_frame, inlier_indices, poseTmp);
}

int FivePoint::Compute(const std::vector<data::Landmark> &landmarks, data::Frame &cur_frame, const data::Frame &prev_frame, std::vector<int> &inlier_indices, Matrix<double, 3, 4> &pose) const
{
    Matrix3d E;
    int inliers = ComputeE(landmarks, prev_frame, cur_frame, E, inlier_indices);
    if (inliers == 0)
    {
        return 0;
    }
    std::vector<Matrix3d> rs;
    std::vector<Vector3d> ts;
    EssentialToPoses(E, rs, ts);
    Matrix<double, 3, 4> I = util::TFUtil::IdentityPoseMatrix<double>();
    int bestPoints = 0;
    Matrix<double, 3, 4> bestPose;
    std::vector<Vector3d> bestXs;
    std::vector<std::pair<const data::Feature*, const data::Feature*>> bestFeats;
    std::vector<int> bestIds;
    for (int i = 0; i < rs.size(); i++)
    {
        Matrix<double, 3, 4> P;
        P.block<3, 3>(0, 0) = rs[i];
        P.block<3, 1>(0, 3) = ts[i];
        int goodPoints = 0;
        std::vector<Vector3d> xs;
        std::vector<std::pair<const data::Feature*, const data::Feature*>> feats;
        std::vector<int> ids;
        for (int inx : inlier_indices)
        {
            const data::Landmark &landmark = landmarks[inx];
            const data::Feature *feat1 = landmark.GetObservationByFrameID(prev_frame.GetID());
            const data::Feature *feat2 = landmark.GetObservationByFrameID(cur_frame.GetID());
            Vector3d X = TriangulateDLT(feat1->GetBearing().normalized(), feat2->GetBearing().normalized(), I, P);

            Vector2d pix;
            if (cur_frame.GetCameraModel().ProjectToImage(util::TFUtil::WorldFrameToCameraFrame(X), pix) && prev_frame.GetCameraModel().ProjectToImage(util::TFUtil::WorldFrameToCameraFrame(util::TFUtil::TransformPoint(P, X)), pix))
            {
                goodPoints++;
                if (prev_frame.HasStereoImage() && landmark.HasEstimatedPosition() && landmark.GetStereoObservationByFrameID(cur_frame.GetID()) != nullptr)
                {
                    const data::Feature *stereoFeat = landmark.GetStereoObservationByFrameID(cur_frame.GetID());
                    xs.push_back(landmark.GetEstimatedPosition());
                    feats.push_back({feat2, stereoFeat});
                    ids.push_back(landmark.GetID());
                }
            }
        }
        if (goodPoints > bestPoints)
        {
            bestPoints = goodPoints;
            bestPose = P;
            bestXs = xs;
            bestFeats = feats;
            bestIds = ids;
        }
    }
    pose = bestPose;
    if (bestXs.size() == 0)
    {
        return inliers;
    }
    Vector3d tvec = bestPose.block<3, 1>(0, 3);
    Vector3d t = bestPose.block<3, 1>(0, 3);
    Matrix<double, 3, 4> estPose;
    if (prev_frame.HasEstimatedPose())
    {
        estPose = util::TFUtil::CombineTransforms(bestPose, prev_frame.GetEstimatedInversePose());
        t = estPose.block<3, 1>(0, 3);
    }
    else if (prev_frame.HasPose())
    {
        estPose = util::TFUtil::CombineTransforms(bestPose, prev_frame.GetInversePose());
        t = estPose.block<3, 1>(0, 3);
    }
    else
    {
        estPose = bestPose;
    }
    std::vector<int> transInliers;
    inliers = ComputeTranslation(bestXs, bestFeats, util::TFUtil::GetRotationFromPoseMatrix(estPose), t, tvec, transInliers);
    std::vector<int> transInlierIds;
    transInlierIds.reserve(inliers);
    for (int i : transInliers)
    {
        transInlierIds.push_back(bestIds[i]);
    }
    estPose.block<3, 1>(0, 3) = t;
    cur_frame.SetEstimatedInversePose(estPose, transInlierIds);
    pose = estPose;

    return inliers;
}

int FivePoint::ComputeE(const std::vector<data::Landmark> &landmarks, const data::Frame &frame1, const data::Frame &frame2, Matrix3d &E, std::vector<int> &inlier_indices, bool stereo) const
{
    std::vector<Vector3d> x1;
    std::vector<Vector3d> x2;
    std::map<int, int> indexToLandmarkIndex;
    int i = 0;
    for (const data::Landmark &landmark : landmarks)
    {
        const data::Feature *feat1 = stereo ? landmark.GetStereoObservationByFrameID(frame1.GetID()) : landmark.GetObservationByFrameID(frame1.GetID());
        const data::Feature *feat2 = stereo ? landmark.GetStereoObservationByFrameID(frame2.GetID()) : landmark.GetObservationByFrameID(frame2.GetID());
        if (feat1 != nullptr && feat2 != nullptr)
        {
            x1.push_back(feat1->GetBearing().normalized());
            x2.push_back(feat2->GetBearing().normalized());
            indexToLandmarkIndex[x1.size() - 1] = i;
        }
        i++;
    }
    if (x1.size() < 5)
    {
        return 0;
    }
    int inliers = ERANSAC(x1, x2, E);
    std::vector<int> indices = GetEInlierIndices(x1, x2, E);
    inlier_indices.clear();
    inlier_indices.reserve(indices.size());
    for (int inx : indices)
    {
        inlier_indices.push_back(indexToLandmarkIndex[inx]);
    }
    return inliers;
}

int FivePoint::ERANSAC(const std::vector<Vector3d> &x1, const std::vector<Vector3d> &x2, Matrix3d &E) const
{
    int maxInliers = 0;
    #pragma omp parallel for
    for (int i = 0; i < ransacIterations_; i++)
    {
        std::random_device rd;
        std::mt19937 eng(rd());
        std::uniform_int_distribution<> distr(0, x1.size() - 1);
        std::unordered_set<int> randset;
        while (randset.size() < 5)
        {
            randset.insert(distr(eng));
        }
        std::vector<int> indices;
        std::copy(randset.begin(), randset.end(), std::back_inserter(indices));

        std::vector<Matrix3d> iterE;
        std::vector<Vector3d> x1s(5);
        std::vector<Vector3d> x2s(5);
        for (int j = 0; j < 5; j++)
        {
            x1s[j] = x1[indices[j]];
            x2s[j] = x2[indices[j]];
        }
        FivePointRelativePose(x1s, x2s, iterE);

        for (Matrix3d &e : iterE)
        {
            int inliers = GetEInlierIndices(x1, x2, e).size();
            #pragma omp critical
            {
                if (inliers > maxInliers)
                {
                    maxInliers = inliers;
                    E = e;
                }
            }
        }
    }
    return maxInliers;
}

std::vector<int> FivePoint::GetEInlierIndices(const std::vector<Vector3d> &x1, const std::vector<Vector3d> &x2, const Matrix3d &E) const
{
    std::vector<int> indices;
    for (int i = 0; i < x1.size(); i++)
    {
        RowVector3d epiplane1 = x2[i].transpose() * E;
        epiplane1.normalize();
        double epiErr1 = std::abs(epiplane1 * x1[i]);
        Vector3d epiplane2 = E * x1[i];
        epiplane2.normalize();
        double epiErr2 = std::abs(x2[i].transpose() * epiplane2);
        if (epiErr1 < epipolarThreshold_ && epiErr2 < epipolarThreshold_)
        {
            indices.push_back(i);
        }
    }
    return indices;
}

void FivePoint::FivePointRelativePose(const std::vector<Vector3d> &x1, const std::vector<Vector3d> &x2, std::vector<Matrix3d> &Es) const
{
    MatrixXd epipolarConstraint(x1.size(), 9);
    for (int i = 0; i < x1.size(); i++)
    {
        epipolarConstraint.row(i) << x2[i](0) * x1[i].transpose(), x2[i](1) * x1[i].transpose(), x2[i](2) * x1[i].transpose();
    }
    Eigen::SelfAdjointEigenSolver<MatrixXd> solver(epipolarConstraint.transpose() * epipolarConstraint);
    Matrix<double, 9, 4> basis = solver.eigenvectors().leftCols<4>();
    Matrix<double, 1, 4> E[3][3] = {
        basis.row(0), basis.row(3), basis.row(6),
        basis.row(1), basis.row(4), basis.row(7),
        basis.row(2), basis.row(5), basis.row(8)
    };

    Matrix<double, 1, 10> EET[3][3];
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            EET[i][j] = 2 * (util::MathUtil::MultiplyDegOnePoly(E[i][0], E[j][0]) + util::MathUtil::MultiplyDegOnePoly(E[i][1], E[j][1]) + util::MathUtil::MultiplyDegOnePoly(E[i][2], E[j][2]));
        }
    }
    Matrix<double, 1, 10> trace = EET[0][0] + EET[1][1] + EET[2][2];
    Matrix<double, 9, 20> traceConstraint;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            traceConstraint.row(3 * i + j) = util::MathUtil::MultiplyDegTwoDegOnePoly(EET[i][0], E[0][j]) + util::MathUtil::MultiplyDegTwoDegOnePoly(EET[i][1], E[1][j]) + util::MathUtil::MultiplyDegTwoDegOnePoly(EET[i][2], E[2][j]) - 0.5 * util::MathUtil::MultiplyDegTwoDegOnePoly(trace, E[i][j]);
        }
    }
    Matrix<double, 1, 20> determinantConstraint = util::MathUtil::MultiplyDegTwoDegOnePoly(Matrix<double, 1, 10>(util::MathUtil::MultiplyDegOnePoly(E[0][1], E[1][2]) - util::MathUtil::MultiplyDegOnePoly(E[0][2], E[1][1])), E[2][0])
        + util::MathUtil::MultiplyDegTwoDegOnePoly(Matrix<double, 1, 10>(util::MathUtil::MultiplyDegOnePoly(E[0][2], E[1][0]) - util::MathUtil::MultiplyDegOnePoly(E[0][0], E[1][2])), E[2][1])
        + util::MathUtil::MultiplyDegTwoDegOnePoly(Matrix<double, 1, 10>(util::MathUtil::MultiplyDegOnePoly(E[0][0], E[1][1]) - util::MathUtil::MultiplyDegOnePoly(E[0][1], E[1][0])), E[2][2]);
    Matrix<double, 10, 20> constraints;
    constraints.block<9, 20>(0, 0) = traceConstraint;
    constraints.row(9) = determinantConstraint;

    Eigen::FullPivLU<Matrix<double, 10, 10>> LU(constraints.block<10, 10>(0, 0));
    Matrix<double, 10, 10> elim = LU.solve(constraints.block<10, 10>(0, 10));

    Matrix<double, 10, 10> action = Matrix<double, 10, 10>::Zero();
    action.block<3, 10>(0, 0) = elim.block<3, 10>(0, 0);
    action.row(3) = elim.row(4);
    action.row(4) = elim.row(5);
    action.row(5) = elim.row(7);
    action(6, 0) = -1.0;
    action(7, 1) = -1.0;
    action(8, 3) = -1.0;
    action(9, 6) = -1.0;

    Eigen::EigenSolver<Matrix<double, 10, 10>> eigensolver(action);
    const auto &eigenvectors = eigensolver.eigenvectors();
    const auto &eigenvalues = eigensolver.eigenvalues();

    Es.reserve(10);
    for (int i = 0; i < 10; i++)
    {
        if (eigenvalues(i).imag() != 0)
        {
            continue;
        }
        Matrix3d EMat;
        Eigen::Map<Matrix<double, 9, 1>>(EMat.data()) = basis * eigenvectors.col(i).tail<4>().real();
        Es.emplace_back(EMat.transpose());
    }
}

void FivePoint::EssentialToPoses(const Matrix3d &E, std::vector<Matrix3d> &rs, std::vector<Vector3d> &ts) const
{
    Eigen::JacobiSVD<Matrix3d> USV(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Matrix3d U = USV.matrixU();
    Matrix3d Vt = USV.matrixV().transpose();

    if (U.determinant() < 0)
    {
        U.col(2) *= -1;
    }
    if (Vt.determinant() < 0)
    {
        Vt.row(2) *= -1;
    }
    Matrix3d W;
    W << 0, -1, 0, 1, 0, 0, 0, 0, 1;

    Matrix3d UWVt = U * W * Vt;
    Matrix3d UWtVt = U * W.transpose() * Vt;

    rs.clear();
    rs.reserve(4);
    ts.clear();
    ts.reserve(4);
    rs.emplace_back(UWVt);
    ts.emplace_back(-UWVt.transpose() * U.col(2));
    rs.emplace_back(UWtVt);
    ts.emplace_back(-UWtVt.transpose() * -U.col(2));
    rs.emplace_back(UWVt);
    ts.emplace_back(-UWVt.transpose() * -U.col(2));
    rs.emplace_back(UWtVt);
    ts.emplace_back(-UWtVt.transpose() * U.col(2));
}

int FivePoint::ComputeTranslation(const std::vector<Vector3d> &xs, const std::vector<std::pair<const data::Feature*, const data::Feature*>> &ys, const Matrix3d &R, Vector3d &t, const Vector3d &tvec, std::vector<int> &inlier_indices) const
{
    int inliers = TranslationRANSAC(xs, ys, R, t, tvec);
    inlier_indices = GetTranslationInlierIndices(xs, ys, R, t);
    std::vector<Vector3d> xs_inliers;
    std::vector<std::pair<const data::Feature*, const data::Feature*>> ys_inliers;
    xs_inliers.reserve(inlier_indices.size());
    ys_inliers.reserve(inlier_indices.size());
    for (int inx : inlier_indices)
    {
        xs_inliers.push_back(xs[inx]);
        ys_inliers.push_back(ys[inx]);
    }
    OptimizeTranslation(xs_inliers, ys_inliers, R, t, tvec);
    return inliers;
}

int FivePoint::TranslationRANSAC(const std::vector<Vector3d> &xs, const std::vector<std::pair<const data::Feature*, const data::Feature*>> &ys, const Matrix3d &R, Vector3d &t, const Vector3d &tvec) const
{
    int bestInliers = 0;
    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_int_distribution<> distr(0, xs.size() - 1);
    std::unordered_set<int> randset;
    while (randset.size() < std::min(transIterations_, (int)xs.size()))
    {
        randset.insert(distr(eng));
    }
    std::vector<int> indices;
    std::copy(randset.begin(), randset.end(), std::back_inserter(indices));

    #pragma omp parallel for
    for (int i = 0; i < indices.size(); i++)
    {
        std::vector<Vector3d> x{xs[indices[i]]};
        std::vector<std::pair<const data::Feature*, const data::Feature*>> y{ys[indices[i]]};
        Vector3d tmpT = t;
        if (OptimizeTranslation(x, y, R, tmpT, tvec))
        {
            int inliers = GetTranslationInlierIndices(xs, ys, R, tmpT).size();
            #pragma omp critical
            {
                if (inliers > bestInliers)
                {
                    t = tmpT;
                    bestInliers = inliers;
                }
            }
        }
    }
    return bestInliers;
}

bool FivePoint::OptimizeTranslation(const std::vector<Vector3d> &xs, const std::vector<std::pair<const data::Feature*, const data::Feature*>> &ys, const Matrix3d &R, Vector3d &t, const Vector3d &tvec) const
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function = nullptr;
    std::vector<double> landmarks;
    landmarks.reserve(3 * xs.size());
    Quaterniond quat(R);
    quat.normalize();
    std::vector<double> quatData(quat.coeffs().data(), quat.coeffs().data() + 4);
    std::vector<double> tData(t.data(), t.data() + 3);
    for (int i = 0; i < xs.size(); i++)
    {
        const Vector3d &x = xs[i];
        landmarks.push_back(x(0));
        landmarks.push_back(x(1));
        landmarks.push_back(x(2));
        ceres::CostFunction *cost_function = nullptr;
        if (ys[i].first->GetFrame().GetCameraModel().GetType() == camera::CameraModel<>::kPerspective)
        {
            cost_function = optimization::ReprojectionError<camera::Perspective>::Create(*ys[i].first, *ys[i].second);
        }
        else if (ys[i].first->GetFrame().GetCameraModel().GetType() == camera::CameraModel<>::kDoubleSphere)
        {
            cost_function = optimization::ReprojectionError<camera::DoubleSphere>::Create(*ys[i].first, *ys[i].second);
        }
        else if (ys[i].first->GetFrame().GetCameraModel().GetType() == camera::CameraModel<>::kUnified)
        {
            cost_function = optimization::ReprojectionError<camera::Unified>::Create(*ys[i].first, *ys[i].second);
        }
        if (cost_function != nullptr)
        {
            problem.AddResidualBlock(cost_function, loss_function, &quatData[0], &tData[0], &landmarks[landmarks.size() - 3]);
            problem.SetParameterBlockConstant(&landmarks[landmarks.size() - 3]);
        }
    }
    problem.SetParameterBlockConstant(&quatData[0]);
    if (fixTransVec_)
    {
        problem.SetParameterization(&tData[0], optimization::ScaleParameterization<3>::Create(tvec));
    }
    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.function_tolerance = 1e-6;
    options.gradient_tolerance = 1e-6;
    options.num_threads = numCeresThreads_;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (!summary.IsSolutionUsable())
    {
        return false;
    }
    const Vector3d tRes = Map<const Vector3d>(&tData[0]);
    t = tRes;
    return true;
}

std::vector<int> FivePoint::GetTranslationInlierIndices(const std::vector<Vector3d> &xs, const std::vector<std::pair<const data::Feature*, const data::Feature*>> &ys, const Matrix3d &R, const Vector3d &t) const
{
    std::vector<int> indices;
    double thresh = reprojectionThreshold_ * reprojectionThreshold_;
    for (int i = 0; i < xs.size(); i++)
    {
        Matrix<double, 3, 4> pose;
        pose.block<3, 3>(0, 0) = R;
        pose.block<3, 1>(0, 3) = t;
        const Vector3d camPt = util::TFUtil::WorldFrameToCameraFrame(util::TFUtil::TransformPoint(pose, xs[i]));
        Vector2d reprojPoint;
        ys[i].first->GetFrame().GetCameraModel().ProjectToImage(camPt, reprojPoint);
        Vector2d stereoReprojPoint;
        ys[i].second->GetFrame().GetStereoCameraModel().ProjectToImage(util::TFUtil::TransformPoint(ys[i].second->GetFrame().GetStereoPose(), camPt), stereoReprojPoint);
        Vector2d pt;
        pt << ys[i].first->GetKeypoint().pt.x, ys[i].first->GetKeypoint().pt.y;
        Vector2d stereoPt;
        stereoPt << ys[i].second->GetKeypoint().pt.x, ys[i].second->GetKeypoint().pt.y;
        double reprojError = (reprojPoint - pt).squaredNorm();
        double stereoReprojError = (stereoReprojPoint - stereoPt).squaredNorm();
        if (reprojError < thresh && stereoReprojError < thresh)
        {
            indices.push_back(i);
        }
    }
    return indices;
}

Vector3d FivePoint::TriangulateDLT(const Vector3d &x1, const Vector3d &x2, const Matrix<double, 3, 4> &pose1, const Matrix<double, 3, 4> &pose2) const
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
