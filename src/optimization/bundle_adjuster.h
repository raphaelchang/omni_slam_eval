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
    BundleAdjuster(int max_iterations = 500, bool log = false);

    bool Optimize(std::vector<data::Landmark> &landmarks);

    std::unique_ptr<ceres::Problem> problem_;
    ceres::Solver::Options solverOptions_;
};

}
}

#endif /* _BUNDLE_ADJUSTER_H_ */
