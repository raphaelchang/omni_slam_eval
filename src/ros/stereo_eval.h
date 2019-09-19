#ifndef _STEREO_EVAL_H_
#define _STEREO_EVAL_H_

#include "tracking_eval.h"
#include "module/stereo_module.h"

#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <vector>

namespace omni_slam
{
namespace ros
{

class StereoEval : public virtual TrackingEval<true>
{
public:
    StereoEval(const ::ros::NodeHandle &nh, const ::ros::NodeHandle &nh_private);
    StereoEval() : StereoEval(::ros::NodeHandle(), ::ros::NodeHandle("~")) {}

protected:
    virtual void InitPublishers();

    virtual void ProcessFrame(std::unique_ptr<data::Frame> &&frame);
    virtual void GetResultsData(std::map<std::string, std::vector<std::vector<double>>> &data);
    virtual void Visualize(cv_bridge::CvImagePtr &base_img, cv_bridge::CvImagePtr &base_stereo_img);

    std::unique_ptr<module::StereoModule> stereoModule_;

private:
    image_transport::Publisher matchedImagePublisher_;
};

}
}

#endif /* _STEREO_EVAL_H_ */
