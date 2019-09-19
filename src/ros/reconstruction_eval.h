#ifndef _RECONSTRUCTION_EVAL_H_
#define _RECONSTRUCTION_EVAL_H_

#include "tracking_eval.h"
#include <ros/ros.h>
#include <vector>

#include "module/reconstruction_module.h"

namespace omni_slam
{
namespace ros
{

template <bool Stereo = false>
class ReconstructionEval : public virtual TrackingEval<Stereo>
{
public:
    ReconstructionEval(const ::ros::NodeHandle &nh, const ::ros::NodeHandle &nh_private);
    ReconstructionEval() : ReconstructionEval(::ros::NodeHandle(), ::ros::NodeHandle("~")) {}

protected:
    virtual void InitPublishers();

    virtual void ProcessFrame(std::unique_ptr<data::Frame> &&frame);
    virtual void Finish();
    virtual void GetResultsData(std::map<std::string, std::vector<std::vector<double>>> &data);
    virtual void Visualize(cv_bridge::CvImagePtr &base_img);

    std::unique_ptr<module::ReconstructionModule> reconstructionModule_;

private:
    ::ros::Publisher pointCloudPublisher_;
};

}
}

#endif /* _RECONSTRUCTION_EVAL_H_ */
