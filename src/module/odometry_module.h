#ifndef _ODOMETRY_MODULE_H_
#define _ODOMETRY_MODULE_H_

#include <vector>
#include <set>
#include <memory>

#include "odometry/pose_estimator.h"
#include "optimization/bundle_adjuster.h"
#include "data/landmark.h"

#include <pcl/common/projection_matrix.h>
#include <pcl/point_types.h>

using namespace Eigen;

namespace omni_slam
{
namespace module
{

class OdometryModule
{
public:
    struct Stats
    {
    };

    OdometryModule(std::unique_ptr<odometry::PoseEstimator> &pose_estimator, std::unique_ptr<optimization::BundleAdjuster> &bundle_adjuster);
    OdometryModule(std::unique_ptr<odometry::PoseEstimator> &&pose_estimator, std::unique_ptr<optimization::BundleAdjuster> &&bundle_adjuster);

    void Update(std::vector<data::Landmark> &landmarks, std::unique_ptr<data::Frame> &cur_frame, const data::Frame *keyframe);
    void BundleAdjust(std::vector<data::Landmark> &landmarks);

    Stats& GetStats();

private:
    std::shared_ptr<odometry::PoseEstimator> poseEstimator_;
    std::shared_ptr<optimization::BundleAdjuster> bundleAdjuster_;

    Stats stats_;
};

}
}

#endif /* _ODOMETRY_MODULE_H_ */
