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

template <bool Stereo = false>
class OdometryEval : public virtual TrackingEval<Stereo>
{
public:
    OdometryEval(const ::ros::NodeHandle &nh, const ::ros::NodeHandle &nh_private);
    OdometryEval() : OdometryEval(::ros::NodeHandle(), ::ros::NodeHandle("~")) {}

protected:
    virtual void InitPublishers();

    virtual void ProcessFrame(std::unique_ptr<data::Frame> &&frame);
    virtual void Finish();
    virtual void GetResultsData(std::map<std::string, std::vector<std::vector<double>>> &data);
    virtual void Visualize(cv_bridge::CvImagePtr &base_img);

    void PublishOdometry();

    std::unique_ptr<module::OdometryModule> odometryModule_;

private:
    ::ros::Publisher odometryPublisher_;
    ::ros::Publisher odometryGndPublisher_;
    ::ros::Publisher pathPublisher_;
    ::ros::Publisher pathGndPublisher_;

    std::string cameraFrame_;
};

}
}

#endif /* _ODOMETRY_EVAL_H_ */
