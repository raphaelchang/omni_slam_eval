#ifndef _ODOMETRY_EVAL_H_
#define _ODOMETRY_EVAL_H_

#include "tracking_eval.h"
#include <ros/ros.h>
#include <vector>

#include "module/odometry_module.h"

namespace omni_slam
{
namespace ros
{

class OdometryEval : public TrackingEval
{
public:
    OdometryEval(const ::ros::NodeHandle &nh, const ::ros::NodeHandle &nh_private);
    OdometryEval() : OdometryEval(::ros::NodeHandle(), ::ros::NodeHandle("~")) {}

private:
    void InitPublishers();

    void ProcessFrame(std::unique_ptr<data::Frame> &&frame);
    void Finish();
    void GetResultsData(std::map<std::string, std::vector<std::vector<double>>> &data);
    void Visualize(cv_bridge::CvImagePtr &base_img);

    std::unique_ptr<module::OdometryModule> odometryModule_;

    ::ros::Publisher odometryPublisher_;
    ::ros::Publisher odometryGndPublisher_;
};

}
}

#endif /* _ODOMETRY_EVAL_H_ */
