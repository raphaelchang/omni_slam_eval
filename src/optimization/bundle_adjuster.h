#ifndef _BUNDLE_ADJUSTER_H_
#define _BUNDLE_ADJUSTER_H_

#include <ceres/ceres.h>
#include "data/landmark.h"

namespace omni_slam
{
namespace optimization
{

class BundleAdjuster
{
public:
    BundleAdjuster(int max_iterations = 500, double loss_coeff = 0.1, int num_threads = 1, bool log = false);

    bool Optimize(std::vector<data::Landmark> &landmarks);

private:
    std::unique_ptr<ceres::Problem> problem_;
    ceres::Solver::Options solverOptions_;

    double lossCoeff_;
};

}
}

#endif /* _BUNDLE_ADJUSTER_H_ */
