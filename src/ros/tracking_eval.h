#ifndef _TRACKING_EVAL_H_
#define _TRACKING_EVAL_H_

#include "eval_base.h"
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <vector>

#include "module/tracking_module.h"

namespace omni_slam
{
namespace ros
{

template <bool Stereo = false>
class TrackingEval : public virtual EvalBase<Stereo>
{
public:
    TrackingEval(const ::ros::NodeHandle &nh, const ::ros::NodeHandle &nh_private);
    TrackingEval() : TrackingEval(::ros::NodeHandle(), ::ros::NodeHandle("~")) {}

protected:
    virtual void InitPublishers();

    virtual void ProcessFrame(std::unique_ptr<data::Frame> &&frame);
    virtual void GetResultsData(std::map<std::string, std::vector<std::vector<double>>> &data);
    virtual void Visualize(cv_bridge::CvImagePtr &base_img);

    std::unique_ptr<module::TrackingModule> trackingModule_;

private:
    image_transport::Publisher trackedImagePublisher_;
};

}
}

#endif /* _TRACKING_EVAL_H_ */
