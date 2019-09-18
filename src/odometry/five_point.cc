#include "five_point.h"

#include "util/math_util.h"
#include <set>
#include <random>

namespace omni_slam
{
namespace odometry
{

FivePoint::FivePoint(int ransac_iterations, double epipolar_threshold)
    : ransacIterations_(ransac_iterations),
    epipolarThreshold_(epipolar_threshold)
{
}

int FivePoint::Compute(const std::vector<data::Landmark> &landmarks, const data::Frame &frame1, const data::Frame &frame2, Matrix3d &E, std::vector<int> &inlier_indices) const
{
    std::vector<Vector3d> x1;
    std::vector<Vector3d> x2;
    std::map<int, int> indexToLandmarkIndex;
    int i = 0;
    for (const data::Landmark &landmark : landmarks)
    {
        if (landmark.IsObservedInFrame(frame2.GetID()) && landmark.IsObservedInFrame(frame1.GetID()))
        {
            const data::Feature *feat1 = landmark.GetObservationByFrameID(frame1.GetID());
            const data::Feature *feat2 = landmark.GetObservationByFrameID(frame2.GetID());
            x1.push_back(feat1->GetBearing().normalized());
            x2.push_back(feat2->GetBearing().normalized());
            indexToLandmarkIndex[x1.size() - 1] = i;
        }
        i++;
    }
    int inliers = RANSAC(x1, x2, E);
    std::vector<int> indices = GetInlierIndices(x1, x2, E);
    inlier_indices.clear();
    inlier_indices.reserve(indices.size());
    for (int inx : indices)
    {
        inlier_indices.push_back(indexToLandmarkIndex[inx]);
    }
    return inliers;
}

int FivePoint::Compute(const std::vector<data::Landmark> &landmarks, const data::Frame &frame1, const data::Frame &frame2, Matrix3d &E) const
{
    std::vector<int> temp;
    return Compute(landmarks, frame1, frame2, E, temp);
}

int FivePoint::RANSAC(const std::vector<Vector3d> &x1, const std::vector<Vector3d> &x2, Matrix3d &E) const
{
    int maxInliers = 0;
    #pragma omp parallel for
    for (int i = 0; i < ransacIterations_; i++)
    {
        std::random_device rd;
        std::mt19937 eng(rd());
        std::uniform_int_distribution<> distr(0, x1.size() - 1);
        std::set<int> randset;
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
            int inliers = GetInlierIndices(x1, x2, e).size();
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

std::vector<int> FivePoint::GetInlierIndices(const std::vector<Vector3d> &x1, const std::vector<Vector3d> &x2, const Matrix3d &E) const
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


}
}
