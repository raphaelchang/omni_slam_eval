#ifndef _SLAM_EVAL_H_
#define _SLAM_EVAL_H_

#include "odometry_eval.h"
#include "reconstruction_eval.h"
#include "stereo_eval.h"
#include <ros/ros.h>
#include <vector>

namespace omni_slam
{
namespace ros
{

class SLAMEval : public OdometryEval<true>, ReconstructionEval<true>, StereoEval
{
public:
    SLAMEval(const ::ros::NodeHandle &nh, const ::ros::NodeHandle &nh_private);
    SLAMEval() : SLAMEval(::ros::NodeHandle(), ::ros::NodeHandle("~")) {}

private:
    void InitPublishers();

    void ProcessFrame(std::unique_ptr<data::Frame> &&frame);
    void Finish();
    void GetResultsData(std::map<std::string, std::vector<std::vector<double>>> &data);
    void Visualize(cv_bridge::CvImagePtr &base_img);
    void Visualize(cv_bridge::CvImagePtr &base_img, cv_bridge::CvImagePtr &base_stereo_img);

    int baSlidingWindow_;
    int frameNum_{0};
};

}
}

#endif /* _SLAM_EVAL_H_ */
