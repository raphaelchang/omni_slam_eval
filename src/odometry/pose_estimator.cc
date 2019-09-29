#include "pose_estimator.h"

namespace omni_slam
{
namespace odometry
{

int PoseEstimator::Compute(const std::vector<data::Landmark> &landmarks, data::Frame &cur_frame, data::Frame &prev_frame) const
{
    std::vector<int> temp;
    return Compute(landmarks, cur_frame, prev_frame, temp);
}

}
}
