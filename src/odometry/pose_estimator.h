#ifndef _POSE_ESTIMATOR_H_
#define _POSE_ESTIMATOR_H_

#include <vector>
#include "data/landmark.h"

namespace omni_slam
{
namespace odometry
{

class PoseEstimator
{
public:
    virtual int Compute(const std::vector<data::Landmark> &landmarks, data::Frame &cur_frame, const data::Frame &prev_frame, std::vector<int> &inlier_indices) const = 0;
    int Compute(const std::vector<data::Landmark> &landmarks, data::Frame &cur_frame, const data::Frame &prev_frame) const;
};

}
}

#endif /* _POSE_ESTIMATOR_H_ */
